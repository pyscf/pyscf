#!/usr/bin/env python
# Copyright 2018-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Pseudo-spectral methods (COSX, PS, SN-K)
'''

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.scf import _vhf
from pyscf.lib import logger
from pyscf.sgx import sgx_jk
from pyscf.df import df_jk
from pyscf import __config__

def sgx_fit(mf, auxbasis=None, with_df=None, pjs=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding SGX or density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        with_df : SGX
            Existing SGX object for the system.
        pjs: bool
            Whether to perform P-junction screening (screening matrix elements
            by the density matrix). Default False. If True, dfj is set to True
            automatically at the beginning of the calculation, as this screening
            is only for K-matrix elements.

    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = sgx_fit(scf.RHF(mol))
    >>> mf.scf()
    -100.00978770917165

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = sgx_fit(scf.UHF(mol))
    >>> mf.scf()
    -100.00978770951018
    '''
    from pyscf.df.addons import predefined_auxbasis
    assert (isinstance(mf, scf.hf.SCF))

    if with_df is None:
        mol = mf.mol
        if auxbasis is None:
            if isinstance(mf, scf.hf.KohnShamDFT):
                xc = mf.xc
            else:
                xc = 'HF'
            if xc == 'LDA,VWN':
                # This is likely the default xc setting of a KS instance.
                # Postpone the auxbasis assignment to with_df.build().
                auxbasis = None
            else:
                auxbasis = predefined_auxbasis(mol, mol.basis, xc)
        with_df = SGX(mol, auxbasis=auxbasis, pjs=pjs)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose

    if isinstance(mf, _SGXHF):
        mf = mf.copy()
        mf.with_df = with_df
        return mf

    dfmf = _SGXHF(mf, with_df, auxbasis)
    return lib.set_class(dfmf, (_SGXHF, mf.__class__))

# A tag to label the derived SCF class
class _SGXHF:

    __name_mixin__ = 'SGX'

    _keys = {
        'auxbasis', 'with_df', 'direct_scf_sgx', 'rebuild_nsteps'
    }

    def __init__(self, mf, df=None, auxbasis=None):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.auxbasis = auxbasis
        self.with_df = df

        # Grids/Integral quality varies during SCF. VHF cannot be
        # constructed incrementally through standard direct SCF.
        self.direct_scf = False
        # Set direct_scf_sgx True to use direct SCF for each
        # grid size with SGX.
        self.direct_scf_sgx = False
        # Set rebuild_nsteps to control how many direct SCF steps
        # are taken between resets of the SGX JK matrix.
        # Default 5, only used if direct_scf_sgx = True
        self.rebuild_nsteps = 5

        self._last_dm = 0
        self._last_vj = 0
        self._last_vk = 0
        self._in_scf = False

    def undo_sgx(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _SGXHF))
        del obj.auxbasis
        del obj.with_df
        del obj.direct_scf_sgx
        del obj.rebuild_nsteps
        del obj._in_scf
        return obj

    def build(self, mol=None, **kwargs):
        if self.direct_scf_sgx:
            self._nsteps_direct = 0
            self._last_dm = 0
            self._last_vj = 0
            self._last_vk = 0
        if self.direct_scf:
            self.with_df.build(level=self.with_df.grids_level_f)
        else:
            self.with_df.build(level=self.with_df.grids_level_i)
        if self.with_df.pjs:
            if not self.with_df.dfj:
                import warnings
                msg = '''
                P-junction screening is not compatible with SGX J-matrix.
                Setting dfj = True. If you want to use SGX J-matrix,
                set pjs = False to turn off P-junction screening.
                '''
                warnings.warn(msg)
            self.with_df.dfj = True # no SGX-J allowed if P-junction screening on
        return super().build(mol, **kwargs)

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def pre_kernel(self, envs):
        self.direct_scf = False # should always be False
        if self.with_df.grids_level_i != self.with_df.grids_level_f:
            self._in_scf = True

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if dm is None: dm = self.make_rdm1()
        with_df = self.with_df
        if not with_df:
            return super().get_jk(self, mol, dm, hermi, with_j, with_k, omega)
        if (self._opt.get(omega) is None and
            self.with_df.direct_j and (not self.with_df.dfj)):
            with mol.with_range_coulomb(omega):
                self._opt[omega] = self.init_direct_scf(mol)
        vhfopt = self._opt.get(omega)

        if self._in_scf and not self.direct_scf:
            if numpy.linalg.norm(dm - self._last_dm) < with_df.grids_switch_thrd \
                    and with_df.grids_level_f != with_df.grids_level_i:
                # only reset if grids_level_f and grids_level_i differ
                logger.debug(self, 'Switching SGX grids')
                with_df.build(level=with_df.grids_level_f)
                self._nsteps_direct = 0
                self._in_scf = False
                self._last_dm = 0
                self._last_vj = 0
                self._last_vk = 0

        if self.direct_scf_sgx:
            vj, vk = with_df.get_jk(dm-self._last_dm, hermi, vhfopt,
                                    with_j, with_k,
                                    self.direct_scf_tol, omega)
            vj += self._last_vj
            vk += self._last_vk
            self._last_dm = numpy.asarray(dm)
            self._last_vj = vj.copy()
            self._last_vk = vk.copy()
            self._nsteps_direct += 1
            if self.rebuild_nsteps > 0 and \
                    self._nsteps_direct >= self.rebuild_nsteps:
                logger.debug(self, 'Resetting JK matrix')
                self._nsteps_direct = 0
                self._last_dm = 0
                self._last_vj = 0
                self._last_vk = 0
        else:
            self._last_dm = numpy.asarray(dm)
            vj, vk = with_df.get_jk(dm, hermi, vhfopt, with_j, with_k,
                                    self.direct_scf_tol, omega)

        return vj, vk

    def post_kernel(self, envs):
        self._in_scf = False
        self._last_dm = 0
        self._last_vj = 0
        self._last_vk = 0

    def to_gpu(self):
        raise NotImplementedError

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError
    nuc_grad_method = Gradients = method_not_implemented
    Hessian = method_not_implemented
    NMR = method_not_implemented
    NSR = method_not_implemented
    Polarizability = method_not_implemented
    RotationalGTensor = method_not_implemented
    MP2 = method_not_implemented
    CISD = method_not_implemented
    CCSD = method_not_implemented
    CASCI = method_not_implemented
    CASSCF = method_not_implemented

scf.hf.SCF.COSX = sgx_fit
mcscf.casci.CASBase.COSX = sgx_fit


def _make_opt(mol, pjs=False,
              direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)):
    '''Optimizer to generate 3-center 2-electron integrals'''
    if pjs:
        vhfopt = _vhf.SGXOpt(mol, 'int1e_grids', 'SGXnr_ovlp_prescreen',
                             dmcondname='SGXnr_dm_cond',
                             direct_scf_tol=direct_scf_tol)
    else:
        vhfopt = _vhf._VHFOpt(mol, 'int1e_grids', 'SGXnr_ovlp_prescreen',
                              direct_scf_tol=direct_scf_tol)
    vhfopt.init_cvhf_direct(mol, 'int1e_ovlp', 'SGXnr_q_cond')
    return vhfopt


class SGX(lib.StreamObject):
    _keys = {
        'mol', 'grids_thrd', 'grids_level_i', 'grids_level_f',
        'grids_switch_thrd', 'dfj', 'direct_j', 'pjs', 'debug', 'grids',
        'blockdim', 'auxmol',
    }

    def __init__(self, mol, auxbasis=None, pjs=False):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.grids_thrd = 1e-10
        self.grids_level_i = 0  # initial grids level
        self.grids_level_f = 1  # final grids level
        self.grids_switch_thrd = 0.03
        # compute J matrix using DF and K matrix using SGX. It's identical to
        # the RIJCOSX method in ORCA
        self.dfj = False
        self.direct_j = False
        self._auxbasis = auxbasis
        self.pjs = pjs

        # debug=True generates a dense tensor of the Coulomb integrals at each
        # grids. debug=False utilizes the sparsity of the integral tensor and
        # contracts the sparse tensor and density matrices on the fly.
        self.debug = False

        self.grids = None
        self.blockdim = 1200
        self.auxmol = None
        self._vjopt = None
        self._opt = None
        self._last_dm = 0
        self._rsh_df = {}  # Range separated Coulomb DF objects

    @property
    def auxbasis(self):
        return self._auxbasis
    @auxbasis.setter
    def auxbasis(self, x):
        if self._auxbasis != x:
            self._auxbasis = x
            self.auxmol = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('max_memory = %s', self.max_memory)
        log.info('grids_level_i = %s', self.grids_level_i)
        log.info('grids_level_f = %s', self.grids_level_f)
        log.info('grids_thrd = %s', self.grids_thrd)
        log.info('grids_switch_thrd = %s', self.grids_switch_thrd)
        log.info('dfj = %s', self.dfj)
        log.info('auxbasis = %s', self.auxbasis)
        return self

    # To mimic DF object, so that SGX can be used as in DF-SCF method by setting
    # mf.with_df = SGX(mol)
    @property
    def _cderi(self):
        return self.grids

    def build(self, level=None):
        if level is None:
            level = self.grids_level_f
        self.grids = sgx_jk.get_gridss(self.mol, level, self.grids_thrd)
        self._opt = _make_opt(self.mol, pjs=self.pjs)

        # In the RSH-integral temporary treatment, recursively rebuild SGX
        # objects in _rsh_df.
        if self._rsh_df:
            for k, v in self._rsh_df.items():
                v.build(level)
        return self

    def kernel(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self.grids = None
        self.auxmol = None
        self._vjopt = None
        self._opt = None
        self._last_dm = 0
        self._rsh_df = {}
        return self

    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is not None:
            # A temporary treatment for RSH integrals
            key = '%.6f' % omega
            if key in self._rsh_df:
                rsh_df = self._rsh_df[key]
            else:
                rsh_df = self.copy()
                rsh_df._rsh_df = None  # to avoid circular reference
                # Not all attributes need to be reset. Resetting _vjopt
                # because it is used by get_j method of regular DF object.
                rsh_df._vjopt = None
                self._rsh_df[key] = rsh_df
                logger.info(self, 'Create RSH-SGX object %s for omega=%s', rsh_df, omega)

            with rsh_df.mol.with_range_coulomb(omega):
                return rsh_df.get_jk(dm, hermi, with_j, with_k,
                                     direct_scf_tol)

        if with_j and self.dfj:
            vj = df_jk.get_j(self, dm, hermi, direct_scf_tol)
            if with_k:
                vk = sgx_jk.get_jk(self, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        elif with_j and self.direct_j:
            vj, _ = _vhf.direct(dm, self.mol._atm, self.mol._bas, self.mol._env,
                                vhfopt, hermi, self.mol.cart, True, False)
            if with_k:
                vk = sgx_jk.get_jk(self, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        else:
            vj, vk = sgx_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        return vj, vk

    to_gpu = lib.to_gpu
