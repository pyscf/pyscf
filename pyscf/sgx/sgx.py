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
import contextlib
import threading
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
        pjs : bool
            If True, the SGX object is set up to screen negligible integrals
            using the density matrix (i.e. P-junction screening), and density
            fitting is used for the J-matrix. If False, no P-junction
            screening is performed, and SGX is used for the J-matrix.
            Screening settings can also be adjusted after initialization.
            See dfj, optk, sgx_tol_energy, and sgx_tol_potential for details.

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
        with_df = SGX(mol, auxbasis=auxbasis)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose

    if pjs:
        with_df.optk = True
        with_df.dfj = True
    else:
        with_df.optk = False
        with_df.dfj = False

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
        'auxbasis', 'with_df', 'rebuild_nsteps'
    }

    def __init__(self, mf, df=None, auxbasis=None):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.auxbasis = auxbasis
        self.with_df = df

        # Set rebuild_nsteps to control how many direct SCF steps
        # are taken between resets of the SGX JK matrix. Default 5
        self.rebuild_nsteps = 5
        self._in_scf = False
        self._ctx_lock = None

    def undo_sgx(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _SGXHF))
        del obj.auxbasis
        del obj.with_df
        del obj.rebuild_nsteps
        del obj._in_scf
        return obj

    def build(self, mol=None, **kwargs):
        self._nsteps_direct = 0
        self._in_scf = False
        self.with_df.build(level=self.with_df.grids_level_i)
        return super().build(mol, **kwargs)

    def reset(self, mol=None):
        self._nsteps_direct = 0
        self._in_scf = False
        self.with_df.reset(mol)
        return super().reset(mol)

    def pre_kernel(self, envs):
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

        if (
                self._in_scf
                and with_df.grids_level_f != with_df.grids_level_i
                and self._grids_reset
        ):
            # only reset if grids_level_f and grids_level_i differ
            logger.debug(self, 'Switching SGX grids')
            with_df.build(level=with_df.grids_level_f)
            self._nsteps_direct = 0
            self._in_scf = False

        vj, vk = with_df.get_jk(dm, hermi, vhfopt, with_j, with_k,
                                self.direct_scf_tol, omega)
        self._nsteps_direct += 1
        if self.rebuild_nsteps > 0 and self._nsteps_direct >= self.rebuild_nsteps:
            self._nsteps_direct = 0

        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        with self.with_full_dm(dm, dm_last) as will_reset:
            if will_reset:
                dm_last = 0
                vhf_last = 0
            veff = super().get_veff(
                mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi
            )
        return veff

    @contextlib.contextmanager
    def with_full_dm(self, dm, dm_last):
        '''
        This context manager yields whether the density matrix should
        be reset, and it also saves the full density matrix dm
        to the SGX object so it can be used for energy screening
        if needed.
        '''
        haslock = self._ctx_lock
        sgx = self.with_df
        if haslock is None:
            self._ctx_lock = threading.RLock()

        with self._ctx_lock:
            try:
                will_reset = False
                self._grids_reset = False
                if sgx.grids_level_f != sgx.grids_level_i \
                        and numpy.linalg.norm(dm - dm_last) < sgx.grids_switch_thrd \
                        and self._in_scf:
                    self._grids_reset = True
                    will_reset = True
                elif self._nsteps_direct == 0:
                    will_reset = True
                if will_reset:
                    sgx._full_dm = None
                else:
                    sgx._full_dm = dm
                yield will_reset
            finally:
                sgx._full_dm = None
                if haslock is None:
                    self._ctx_lock = None

    def post_kernel(self, envs):
        self._in_scf = False

    def to_gpu(self):
        raise NotImplementedError

    def method_not_implemented(self, *args, **kwargs):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.sgx.grad import rhf, uhf, rks, uks
        if isinstance(self, (scf.uhf.UHF, scf.rohf.ROHF)):
            if isinstance(self, scf.hf.KohnShamDFT):
                return uks.Gradients(self)
            else:
                return uhf.Gradients(self)
        elif isinstance(self, scf.rhf.RHF):
            if isinstance(self, scf.hf.KohnShamDFT):
                return rks.Gradients(self)
            else:
                return rhf.Gradients(self)
        else:
            raise NotImplementedError

    Gradients = nuc_grad_method

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


def _make_opt(mol, grad=False,
              direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13)):
    '''Optimizer to generate 3-center 2-electron integrals'''
    if grad:
        intor_name = 'int1e_grids_ip'
    else:
        intor_name = 'int1e_grids'
    vhfopt = _vhf._VHFOpt(mol, intor_name, 'SGXnr_ovlp_prescreen',
                          direct_scf_tol=direct_scf_tol)
    vhfopt.init_cvhf_direct(mol, 'int1e_ovlp', 'SGXnr_q_cond')
    return vhfopt


class SGX(lib.StreamObject):
    _keys = {
        'mol', 'grids_thrd', 'grids_level_i', 'grids_level_f',
        'grids_switch_thrd', 'dfj', 'direct_j', 'debug', 'grids',
        'blockdim', 'auxmol', 'sgx_tol_potential',
        'sgx_tol_energy', 'use_opt_grids', 'fit_ovlp', 'bound_algo'
    }

    def __init__(self, mol, auxbasis=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        # BASIC SGX SETTINGS
        # Remove grids with small weights using this threshold.
        self.grids_thrd = 1e-10
        # initial grids level
        self.grids_level_i = 2
        # final grids level
        self.grids_level_f = 2
        # use optimized grids for SGX based on ORCA
        self.use_opt_grids = True
        # numerically fit overlap matrix to improve numerical precision
        self.fit_ovlp = True
        # threshold of density matrix convergence to switch from the coarse
        # initial grid (grids_level_i) to the denser final grid (grids_level_f)
        # Ignored if grids_level_i == grids_level_f
        self.grids_switch_thrd = 0.03
        # compute J matrix using DF and K matrix using SGX. It's identical to
        # the RIJCOSX method in ORCA
        self.dfj = True

        # OPTIMIZED SGX-K SETTINGS
        # Turn on optimization for evaluating the K-matrix with SGX.
        # Only has an effect is dfj is True
        self.optk = True
        # DM screening error tolerance for energy. "auto" means set
        # automatically based on direct_scf_tol.
        self.sgx_tol_energy = "auto"
        # DM screening error tolerance for the potential. "auto" means
        # set to sqrt(sgx_tol_energy), or sqrt(direct_scf_tol)
        # if sgx_tol_energy is None.
        self.sgx_tol_potential = "auto"
        # Note: If (sgx_tol_energy, sgx_tol_potential) is
        #   (float/"auto", float/"auto"): Bound energy and potential error
        #   (float/"auto", None): Bound energy error only
        #   (None, float/"auto"): Bound potential error only
        #   (None, None): Turn off DM screening (no energy or potential error)
        # It is recommended to bound both energy and potential error
        # for numerical stability

        # Bound algo determines how the three-center integral upper bounds
        # are estimated. Can be
        #   "ovlp": Screen integrals based on overlap of
        #       orbital pairs. Overlap serves as a rough
        #       approximation of the maximum ESP integral.
        #   "sample": Provide an approximate but accurate
        #       upper bound for the ESP integrals by sampling
        #       _nquad points for each shell pair.
        #   "sample_pos": Same as sample, but the ESP
        #       bounds are position-dependent, which gives
        #       a slight speed increase for large systems
        #       and a significant speed increase for
        #       short-range hybrids.
        # Default is "sample_pos" and is recommended for most cases.
        self.bound_algo = "sample_pos"

        self.grids = None
        self.auxmol = None

        # DEBUGGING SETTINGS
        # debug=True generates a dense tensor of the Coulomb integrals at each
        # grid. debug=False utilizes the sparsity of the integral tensor and
        # contracts the sparse tensor and density matrices on the fly.
        self.debug = False
        # max block size for grids when debug=True
        self.blockdim = 1200
        # Run the calculation with direct integral J-matrix.
        self.direct_j = False
        # perform a symmetric overlap fit when optk is True.
        # When fit_ovlp=True, _symm_ovlp_fit=True is required
        # for exact analytical gradients.
        # Note that symmetric overlap fitting is not "perfect"
        # overlap fitting. It fits the overlap correctly to
        # first order in the difference between the exact and
        # numerical overlap matrix. This difference is quite small
        # for any reasonable grid, so this approximation typically
        # works well.
        self._symm_ovlp_fit = True

        # private attributes
        self._auxbasis = auxbasis
        self._vjopt = None
        self._opt = None
        self._rsh_df = {}  # Range separated Coulomb DF objects
        self._overlap_correction_matrix = None
        self._sgx_block_cond = None
        self._full_dm = None
        self._pjs_data = None

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
        self._overlap_correction_matrix = None
        self._sgx_block_cond = None
        if level is None:
            level = self.grids_level_f
        self.grids = sgx_jk.get_gridss(
            self.mol, level, self.grids_thrd, self.use_opt_grids
        )
        self._opt = _make_opt(self.mol)
        self._pjs_data = None

        # In the RSH-integral temporary treatment, recursively rebuild SGX
        # objects in _rsh_df.
        if self._rsh_df:
            for k, v in self._rsh_df.items():
                v.build(level)
        return self

    def _build_pjs(self, direct_scf_tol):
        assert self._opt is not None
        self._pjs_data = sgx_jk.SGXData(
            self.mol,
            self.grids,
            fit_ovlp=self.fit_ovlp,
            sym_ovlp=self._symm_ovlp_fit,
            max_memory=self.max_memory,
            direct_scf_tol=direct_scf_tol,
            vtol=self.sgx_tol_potential,
            etol=self.sgx_tol_energy,
            hermi=1,
            bound_algo=self.bound_algo,
            sgxopt=self._opt,
        )
        self._pjs_data.build()

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
        self._pjs_data = None
        self._rsh_df = {}
        self._overlap_correction_matrix = None
        self._sgx_block_cond = None
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
                rsh_df._overlap_correction_matrix = None
                self._rsh_df[key] = rsh_df
                logger.info(self, 'Create RSH-SGX object %s for omega=%s', rsh_df, omega)

            with rsh_df.mol.with_range_coulomb(omega):
                return rsh_df.get_jk(dm, hermi, with_j, with_k,
                                     direct_scf_tol)

        if with_j and (self.dfj or self.direct_j):
            if self.dfj:
                vj = df_jk.get_j(self, dm, hermi, direct_scf_tol)
            else:
                vj, _ = _vhf.direct(dm, self.mol._atm, self.mol._bas, self.mol._env,
                                    vhfopt, hermi, self.mol.cart, True, False)
            if with_k:
                if self.optk:
                    vk = sgx_jk.get_k_only(self, dm, hermi, direct_scf_tol)
                else:
                    vk = sgx_jk.get_jk(self, dm, hermi, False, with_k, direct_scf_tol)[1]
            else:
                vk = None
        else:
            if (not with_j) and with_k and self.optk:
                vj = None
                vk = sgx_jk.get_k_only(self, dm, hermi, direct_scf_tol)
            else:
                vj, vk = sgx_jk.get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)
        return vj, vk

    to_gpu = lib.to_gpu
