#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''


import numpy
import pyscf.dft
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import hf as pbchf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import addons
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.dft import rks as mol_ks
from pyscf.pbc.dft import multigrid
from pyscf.pbc.lib.kpts import KPoints
from pyscf import __config__


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpts_band=None):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (logger.process_clock(), logger.perf_counter())

    ni = ks._numint
    if isinstance(ni, multigrid.MultiGridNumInt):
        if ks.do_nlc():
            raise NotImplementedError(f'MultiGrid for NLC functional {ks.xc} + {ks.nlc}')
        j_in_xc = ni.xc_with_j
    else:
        ks.initialize_grids(cell, dm, kpt)
        j_in_xc = False

    max_memory = ks.max_memory - lib.current_memory()[0]
    n, exc, vxc = ni.nr_rks(cell, ks.grids, ks.xc, dm, 0, hermi,
                            kpt, kpts_band, max_memory=max_memory)
    logger.info(ks, 'nelec by numeric integration = %s', n)
    if ks.do_nlc():
        if ni.libxc.is_nlc(ks.xc):
            xc = ks.xc
        else:
            assert ni.libxc.is_nlc(ks.nlc)
            xc = ks.nlc
        n, enlc, vnlc = ni.nr_nlc_vxc(cell, ks.nlcgrids, xc, dm, 0, hermi, kpt,
                                      max_memory=max_memory)
        exc += enlc
        vxc += vnlc
        logger.info(ks, 'nelec with nlc grids = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    ground_state = kpts_band is None
    vj, vk = _get_jk(ks, cell, dm, hermi, kpt, kpts_band, with_j=not j_in_xc)
    if j_in_xc:
        ecoul = vxc.ecoul
    else:
        vxc += vj
        ecoul = None
        if ground_state:
            ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    if ni.libxc.is_hybrid_xc(ks.xc):
        vxc -= .5 * vk
        if ground_state:
            exc -= numpy.einsum('ij,ji->', dm, vk).real * .25
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    logger.timer(ks, 'veff', *t0)
    return vxc

def _patch_df_beckegrids(density_fit):
    def new_df(self, auxbasis=None, with_df=None, *args, **kwargs):
        mf = density_fit(self, auxbasis, with_df, *args, **kwargs)
        mf.with_df._j_only = not self._numint.libxc.is_hybrid_xc(self.xc)
        mf.grids = gen_grid.BeckeGrids(self.cell)
        mf.grids.level = getattr(__config__, 'dft_rks_RKS_grids_level',
                                 mf.grids.level)
        mf.nlcgrids = gen_grid.BeckeGrids(self.cell)
        mf.nlcgrids.level = getattr(__config__, 'dft_rks_RKS_nlcgrids_level',
                                    mf.nlcgrids.level)
        return mf
    return new_df

NELEC_ERROR_TOL = getattr(__config__, 'pbc_dft_rks_prune_error_tol', 0.02)
def prune_small_rho_grids_(ks, cell, dm, grids, kpts):
    rho = ks.get_rho(dm, grids, kpts)
    n = numpy.dot(rho, grids.weights)
    if abs(n-cell.nelectron) < NELEC_ERROR_TOL*n:
        rho *= grids.weights
        idx = abs(rho) > ks.small_rho_cutoff / grids.weights.size
        logger.debug(ks, 'Drop grids %d',
                     grids.weights.size - numpy.count_nonzero(idx))
        grids.coords  = numpy.asarray(grids.coords [idx], order='C')
        grids.weights = numpy.asarray(grids.weights[idx], order='C')
        grids.non0tab = grids.make_mask(cell, grids.coords)
    return grids

@lib.with_doc(pbchf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpt=None):
    if dm is None: dm = mf.make_rdm1()
    if grids is None: grids = mf.grids
    if kpt is None: kpt = mf.kpt
    if dm[0].ndim == 2:  # the UKS density matrix
        dm = dm[0] + dm[1]
    return mf._numint.get_rho(mf.cell, dm, grids, kpt, mf.max_memory)

def gen_response(mf, mo_coeff=None, mo_occ=None,
                 singlet=None, hermi=0, max_memory=None, with_nlc=True):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpt = mf.kpt
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    j_in_xc = getattr(ni, 'xc_with_j', False)

    if with_nlc and mf.do_nlc():
        raise NotImplementedError

    if singlet is None:  # for newton solver
        spin = 0
    else:
        spin = 1
    rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                        mo_occ, spin, kpt)
    dm0 = None

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if singlet is None:  # Without specify singlet, general case
        def vind(dm1):
            # The singlet hessian
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, kpt, max_memory=max_memory)
            vj, vk = _get_jk(mf, cell, dm1, hermi, kpt, with_j=not j_in_xc)
            if not j_in_xc:
                v1 += vj
            if hybrid:
                v1 -= .5 * vk
            return v1

    elif singlet:
        fxc *= .5
        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                # nr_rks_fxc_st requires alpha of dm1
                v1 = ni.nr_rks_fxc_st(cell, mf.grids, mf.xc, dm0, dm1, hermi,
                                      True, rho0, vxc, fxc, kpt,
                                      max_memory=max_memory)
            vj, vk = _get_jk(mf, cell, dm1, hermi, kpt, with_j=not j_in_xc)
            if not j_in_xc:
                v1 += vj
            if hybrid:
                v1 -= .5 * vk
            return v1
    else:  # triplet
        fxc *= .5
        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                # nr_rks_fxc_st requires alpha of dm1
                v1 = ni.nr_rks_fxc_st(cell, mf.grids, mf.xc, dm0, dm1, hermi,
                                      False, rho0, vxc, fxc, kpt,
                                      max_memory=max_memory)
            vk = _get_jk(mf, cell, dm1, hermi, kpt, with_j=False)[1]
            if hybrid:
                v1 -= .5 * vk
            return v1
    return vind

def _get_jk(mf, cell, dm, hermi, kpt, kpts_band=None, with_j=True):
    '''J and Exx matrix. Note, Exx here is a scaled HF K term.'''
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    vj = vk = None
    if not hybrid:
        if hermi == 2 or not with_j:
            vj = 0
        else:
            vj = mf.get_j(cell, dm, hermi, kpt, kpts_band)
        return vj, vk

    if omega == 0:
        if hermi == 2 or not with_j:
            vk = mf.get_k(cell, dm, hermi, kpt, kpts_band)
        else:
            vj, vk = mf.get_jk(cell, dm, hermi, kpt, kpts_band)
        vk *= hyb
    elif alpha == 0: # LR=0, only SR exchange
        vk = mf.get_k(cell, dm, hermi, kpt, kpts_band, omega=-omega)
        vk *= hyb
        vj = mf.get_j(cell, dm, hermi, kpt, kpts_band)
    elif hyb == 0: # SR=0, only LR exchange
        vk = mf.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
        vk *= alpha
        vj = mf.get_j(cell, dm, hermi, kpt, kpts_band)
    else: # SR and LR exchange with different ratios
        if hermi == 2 or not with_j:
            vk = mf.get_k(cell, dm, hermi, kpt, kpts_band)
        else:
            vj, vk = mf.get_jk(cell, dm, hermi, kpt, kpts_band)
        vk *= hyb
        vklr = mf.get_k(cell, dm, hermi, kpt, kpts_band, omega=omega)
        vklr *= (alpha - hyb)
        vk += vklr
    return vj, vk

class KohnShamDFT(mol_ks.KohnShamDFT):
    '''PBC-KS'''

    _keys = {'xc', 'nlc', 'grids', 'nlcgrids', 'small_rho_cutoff'}
    # Use rho to filter grids
    small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

    get_rho = get_rho

    density_fit = _patch_df_beckegrids(pbchf.RHF.density_fit)
    rs_density_fit = _patch_df_beckegrids(pbchf.RHF.rs_density_fit)
    mix_density_fit = _patch_df_beckegrids(pbchf.RHF.mix_density_fit)
    gen_response = NotImplemented

    def __init__(self, xc='LDA,VWN'):
        self.xc = xc
        self.grids = gen_grid.UniformGrids(self.cell)
        self.nlc = ''
        self.nlcgrids = gen_grid.UniformGrids(self.cell)
##################################################
# don't modify the following attributes, they are not input options
        # Note Do not refer to .with_df._numint because mesh/coords may be different
        if isinstance(self, khf.KSCF):
            if isinstance(self.kpts, KPoints):
                self._numint = numint.KNumInt(self.kpts.kpts)
            else:
                self._numint = numint.KNumInt(self.kpts)
        else:
            self._numint = numint.NumInt()

    def dump_flags(self, verbose=None):
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags(verbose)
        return self

    def reset(self, cell=None):
        pbchf.SCF.reset(self, cell)
        self.grids.reset(cell)
        self.nlcgrids.reset(cell)
        self._numint.reset(cell)
        return self

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpts' in self.__dict__:
            self.kpts = self.__dict__.pop('kpts')
        elif 'kpt' in self.__dict__:
            self.kpt = self.__dict__.pop('kpt')

        kpts = self.kpts
        if self.rsjk:
            if not numpy.all(self.rsjk.kpts == self.kpt):
                self.rsjk = self.rsjk.__class__(cell, kpts)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    @lib.with_doc(pbchf.SCF.jk_method.__doc__)
    def jk_method(self, J='FFTDF', K=None):
        if K is None:
            K = J
        if (('RS' in J or 'RS' in K) and
            not isinstance(self.grids, gen_grid.BeckeGrids)):
            self.grids = gen_grid.BeckeGrids(self.cell)
        return pbchf.SCF.jk_method(self, J, K)

    def to_rks(self, xc=None):
        '''Convert the input mean-field object to a RKS/ROKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = addons.convert_to_rhf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc and isinstance(self, RKS)
        return mf

    def to_uks(self, xc=None):
        '''Convert the input mean-field object to a UKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = addons.convert_to_uhf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc
        return mf

    def to_gks(self, xc=None):
        '''Convert the input mean-field object to a GKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.pbc.dft.numint2c import NumInt2C, KNumInt2C
        mf = addons.convert_to_ghf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc
        if isinstance(mf, khf.KSCF):
            if not isinstance(mf._numint, KNumInt2C):
                mf._numint = KNumInt2C(mf._numint.kpts)
        else:
            if not isinstance(mf._numint, NumInt2C):
                mf._numint = NumInt2C()
        return mf

    def to_hf(self):
        raise NotImplementedError

    def initialize_grids(self, cell, dm, kpts, ground_state=True):
        '''Initialize self.grids the first time call get_veff'''
        if self.grids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.grids.build(with_non0tab=True)
            if (isinstance(self.grids, gen_grid.BeckeGrids) and
                self.small_rho_cutoff > 1e-20 and ground_state):
                self.grids = prune_small_rho_grids_(
                    self, self.cell, dm, self.grids, kpts)
            t0 = logger.timer(self, 'setting up grids', *t0)
        is_nlc = self.do_nlc()
        if is_nlc and self.nlcgrids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.nlcgrids.build(with_non0tab=True)
            if (isinstance(self.grids, gen_grid.BeckeGrids) and
                self.small_rho_cutoff > 1e-20 and ground_state):
                self.nlcgrids = prune_small_rho_grids_(
                    self, self.cell, dm, self.nlcgrids, kpts)
            t0 = logger.timer(self, 'setting up nlc grids', *t0)
        return self

    def to_gpu(self):
        raise NotImplementedError

# Update the KohnShamDFT label in pbc.scf.hf module
pbchf.KohnShamDFT = KohnShamDFT


class RKS(KohnShamDFT, pbchf.RHF):
    '''PBC-RKS at a single point (default: gamma point).

    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.
    '''

    get_vsap = mol_ks.get_vsap
    init_guess_by_vsap = mol_ks.init_guess_by_vsap
    get_veff = get_veff
    energy_elec = mol_ks.energy_elec
    gen_response = gen_response

    def __init__(self, cell, kpt=None, xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.RHF.__init__(self, cell, kpt, exxdiv=exxdiv)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        pbchf.RHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    def to_hf(self):
        '''Convert to RHF object.'''
        from pyscf.pbc import scf
        return self._transfer_attrs_(scf.RHF(self.cell, self.kpt))

    def multigrid_numint(self, mesh=None):
        '''Apply the MultiGrid algorithm for XC numerical integartion'''
        mf = self.copy()
        mf._numint = multigrid.MultiGridNumInt(self.cell)
        if mesh is not None:
            mf._numint.mesh = mesh
        return mf

    def Gradients(self):
        from pyscf.pbc.grad import rks
        from pyscf.pbc.dft.multigrid import MultiGridNumInt2
        if not isinstance(self._numint, MultiGridNumInt2):
            raise NotImplementedError('pbc-RKS must be computed with MultiGridNumInt2')
        return rks.Gradients(self)

    to_gpu = lib.to_gpu
