#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

'''Density expansion on plane waves'''


import copy
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo, estimate_ke_cutoff, error_for_ke_cutoff
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df import aft_ao2mo
from pyscf.pbc.df.incore import _IntNucBuilder, _compensate_nuccell
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf import __config__


CUTOFF = getattr(__config__, 'pbc_df_aft_estimate_eta_cutoff', 1e-12)
ETA_MIN = getattr(__config__, 'pbc_df_aft_estimate_eta_min', 0.2)
OMEGA_MIN = getattr(__config__, 'pbc_df_aft_estimate_omega_min', 0.3)
PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)
KE_SCALING = getattr(__config__, 'pbc_df_aft_ke_cutoff_scaling', 0.75)

def estimate_eta_min(cell, cutoff=CUTOFF):
    '''Given rcut the boundary of repeated images of the cell, estimates the
    minimal exponent of the smooth compensated gaussian model charge, requiring
    that at boundary, density ~ 4pi rmax^2 exp(-eta/2*rmax^2) < cutoff
    '''
    lmax = min(numpy.max(cell._bas[:,gto.ANG_OF]), 4)
    # If lmax=3 (r^5 for radial part), this expression guarantees at least up
    # to f shell the convergence at boundary
    rcut = cell.rcut
    eta = max(numpy.log(4*numpy.pi*rcut**(lmax+2)/cutoff)/rcut**2, ETA_MIN)
    return eta

estimate_eta = estimate_eta_min

def estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=PRECISION):
    '''Given ke_cutoff, the upper bound of eta to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    # search eta for interaction between GTO(eta) and point charge at the same
    # location so that
    # \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 GTO(eta, k) < precision
    # GTO(eta, k) = Fourier transform of Gaussian e^{-eta r^2}

    lmax = numpy.max(cell._bas[:,gto.ANG_OF])
    kmax = (ke_cutoff*2)**.5
    # The interaction between two s-type density distributions should be
    # enough for the error estimation.  Put lmax here to increate Ecut for
    # slightly better accuracy
    log_rest = numpy.log(precision / (32*numpy.pi**2 * kmax**max(0, lmax-1)))
    log_eta = -1
    eta = kmax**2/4 / (-log_eta - log_rest)
    return eta

def estimate_ke_cutoff_for_eta(cell, eta, precision=PRECISION):
    '''Given eta, the lower bound of ke_cutoff to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    # estimate ke_cutoff for interaction between GTO(eta) and point charge at
    # the same location so that
    # \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 GTO(eta, k) < precision
    # \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 GTO(eta, k)
    # ~ \int_kmax^infty 4*pi/k^2 GTO(eta,k) dk^3
    # = (4*pi)^2 *2*eta/kmax^{n-1} e^{-kmax^2/4eta} + ... < precision

    # The magic number 0.2 comes from AFTDF.__init__ and GDF.__init__
    # eta = max(eta, ETA_MIN)

    log_k0 = 3 + numpy.log(eta) / 2
    log_rest = numpy.log(precision / (32*numpy.pi**2*eta))
    # The interaction between two s-type density distributions should be
    # enough for the error estimation.  Put lmax here to increate Ecut for
    # slightly better accuracy
    lmax = numpy.max(cell._bas[:,gto.ANG_OF])
    Ecut = 2*eta * (log_k0*max(0, lmax-1) - log_rest)
    Ecut = max(Ecut, .5)
    return Ecut

def estimate_omega_min(cell, cutoff=CUTOFF):
    '''Given cell.rcut the boundary of repeated images of the cell, estimates
    the minimal omega for the attenuated Coulomb interactions, requiring that at
    boundary the Coulomb potential of a point charge < cutoff
    '''
    # erfc(z) = 2/\sqrt(pi) int_z^infty exp(-t^2) dt < exp(-z^2)/(z\sqrt(pi))
    # erfc(omega*rcut)/rcut < cutoff
    # ~ exp(-(omega*rcut)**2) / (omega*rcut**2*pi**.5) < cutoff
    rcut = cell.rcut
    omega = OMEGA_MIN
    omega = max((-numpy.log(cutoff * rcut**2 * omega))**.5 / rcut, OMEGA_MIN)
    return omega

estimate_omega = estimate_omega_min

# \sum_{k^2/2 > ke_cutoff} weight*4*pi/k^2 exp(-k^2/(4 omega^2)) rho(k) < precision
# ~ 16 pi^2 int_cutoff^infty exp(-k^2/(4*omega^2)) dk
# = 16 pi^{5/2} omega erfc(sqrt(ke_cutoff/(2*omega^2)))
# ~ 16 pi^2 exp(-ke_cutoff/(2*omega^2)))
def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff to converge attenuated Coulomb in moment space
    '''
    if precision is None:
        precision = cell.precision
    precision *= 1e-2
    lmax = numpy.max(cell._bas[:,gto.ANG_OF])
    ke_cutoff = -2*omega**2 * numpy.log(precision / (16*numpy.pi**2))
    ke_cutoff = -2*omega**2 * numpy.log(precision / (16*numpy.pi**2*(ke_cutoff*2)**(.5*lmax)))
    return ke_cutoff

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulombl given energy cutoff
    '''
    if precision is None:
        precision = cell.precision
    # esitimation based on \int dk 4pi/k^2 exp(-k^2/4omega) sometimes is not
    # enough to converge the 2-electron integrals. A penalty term here is to
    # reduce the error in integrals
    precision *= 1e-2
    # Consider l>0 basis here to increate Ecut for slightly better accuracy
    lmax = numpy.max(cell._bas[:,gto.ANG_OF])
    kmax = (ke_cutoff*2)**.5
    log_rest = numpy.log(precision / (16*numpy.pi**2 * kmax**lmax))
    omega = (-.5 * ke_cutoff / log_rest)**.5
    return omega

def get_pp_loc_part1(mydf, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    if mydf.eta != 0:
        dfbuilder = _IntNucBuilder(mydf.cell, kpts_lst)
        vj = dfbuilder.get_pp_loc_part1()
        if kpts is None or numpy.shape(kpts) == (3,):
            vj = vj[0]
        return numpy.asarray(vj)

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    cell = mydf.cell
    mesh = numpy.asarray(mydf.mesh)
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpt_allow = numpy.zeros(3)
    if cell.dimension > 0:
        ke_guess = estimate_ke_cutoff(cell, cell.precision)
        mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
        if numpy.any(mesh[:cell.dimension] < mesh_guess[:cell.dimension]*.8):
            logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                        'to get integral accuracy %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_guess)
    log.debug1('aft.get_pp_loc_part1 mesh = %s', mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

    vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)

    vpplocG *= kws
    vG = vpplocG
    vj = numpy.zeros((nkpts,nao_pair), dtype=numpy.complex128)

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for aoaoks, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts_lst,
                                       max_memory=max_memory, aosym='s2'):
        for k, aoao in enumerate(aoaoks):
            # rho_ij(G) nuc(-G) / G^2
            # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            if gamma_point(kpts_lst[k]):
                vj[k] += numpy.einsum('k,kx->x', vG[p0:p1].real, aoao.real)
                vj[k] += numpy.einsum('k,kx->x', vG[p0:p1].imag, aoao.imag)
            else:
                vj[k] += numpy.einsum('k,kx->x', vG[p0:p1].conj(), aoao)
        t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
    log.timer_debug1('contracting Vnuc', *t0)

    vj_kpts = []
    for k, kpt in enumerate(kpts_lst):
        if gamma_point(kpt):
            vj_kpts.append(lib.unpack_tril(vj[k].real))
        else:
            vj_kpts.append(lib.unpack_tril(vj[k]))

    if kpts is None or numpy.shape(kpts) == (3,):
        vj_kpts = vj_kpts[0]
    return numpy.asarray(vj_kpts)

def _int_nuc_vloc(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    dfbuilder = _IntNucBuilder(mydf.cell, kpts)
    return dfbuilder._int_nuc_vloc(nuccell, intor, aosym)

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    dfbuilder = _IntNucBuilder(mydf.cell, kpts)
    vpp = dfbuilder.get_pp(mydf.mesh)
    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    logger.timer(mydf, 'get_pp', *t0)
    return vpp

def get_nuc(mydf, kpts=None):
    t0 = (logger.process_clock(), logger.perf_counter())
    dfbuilder = _IntNucBuilder(mydf.cell, kpts)
    vj = dfbuilder.get_nuc(mydf.mesh)
    if kpts is None or numpy.shape(kpts) == (3,):
        vj = vj[0]
    logger.timer(mydf, 'get_nuc', *t0)
    return vj

def weighted_coulG(mydf, kpt=numpy.zeros(3), exx=False, mesh=None, omega=None):
    '''Weighted regular Coulomb kernel, applying cell.omega by default'''
    cell = mydf.cell
    if mesh is None:
        mesh = mydf.mesh
    if omega is None:
        omega = cell.omega
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = tools.get_coulG(cell, kpt, exx, mydf, mesh, Gv, omega=omega)
    coulG *= kws
    return coulG


class AFTDFMixin:

    weighted_coulG = weighted_coulG
    _int_nuc_vloc = _int_nuc_vloc
    get_nuc = get_nuc
    get_pp = get_pp

    def pw_loop(self, mesh=None, kpti_kptj=None, q=None, shls_slice=None,
                max_memory=2000, aosym='s1', blksize=None,
                intor='GTO_ft_ovlp', comp=1, bvk_kmesh=None):
        '''
        Fourier transform iterator for AO pair
        '''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj
        if q is None:
            q = kptj - kpti

        ao_loc = cell.ao_loc_nr()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert (shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj

        if blksize is None:
            blksize = min(max(64, int(max_memory*1e6*.75/(nij*16*comp))), 16384)
            sublk = int(blksize//4)
        else:
            sublk = blksize
        buf = numpy.empty(nij*blksize*comp, dtype=numpy.complex128)
        pqkRbuf = numpy.empty(nij*sublk*comp)
        pqkIbuf = numpy.empty(nij*sublk*comp)

        if bvk_kmesh is None:
            bvk_kmesh = k2gamma.kpts_to_kmesh(cell, [kpti, kptj])
        supmol_ft = ft_ao._ExtendedMole.from_cell(cell, bvk_kmesh).strip_basis()
        ft_kern = supmol_ft.gen_ft_kernel(aosym, intor=intor, comp=comp)

        for p0, p1 in self.prange(0, ngrids, blksize):
            aoaoR, aoaoI = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, q,
                                   kptj.reshape(1, 3), shls_slice, out=buf)
            aoaoR = aoaoR.reshape(comp,p1-p0,nij)
            aoaoI = aoaoI.reshape(comp,p1-p0,nij)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                if comp == 1:
                    pqkR = numpy.ndarray((nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoaoR[0,i0:i1].T
                    pqkI[:] = aoaoI[0,i0:i1].T
                else:
                    pqkR = numpy.ndarray((comp,nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((comp,nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoaoR[:,i0:i1].transpose(0,2,1)
                    pqkI[:] = aoaoI[:,i0:i1].transpose(0,2,1)
                yield (pqkR, pqkI, p0+i0, p0+i1)

    def ft_loop(self, mesh=None, q=numpy.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1,
                bvk_kmesh=None):
        '''
        Fourier transform iterator for all kpti which satisfy
            2pi*N = (kpts - kpti - q)*a,  N = -1, 0, 1
        '''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if kpts is None:
            assert (is_zero(q))
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        ao_loc = cell.ao_loc_nr()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert (shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj

        if bvk_kmesh is None:
            bvk_kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        # ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
        # rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell, ke_cutoff, ft_ao.RCUT_THRESHOLD)
        supmol_ft = ft_ao._ExtendedMole.from_cell(cell, bvk_kmesh).strip_basis()
        ft_kern = supmol_ft.gen_ft_kernel(aosym, intor=intor, comp=comp,
                                          return_complex=True)

        blksize = max(16, int(max_memory*.9e6/(nij*nkpts*16*comp)))
        blksize = min(blksize, ngrids, 16384)
        buf = numpy.empty(nkpts*nij*blksize*comp, dtype=numpy.complex128)

        for p0, p1 in self.prange(0, ngrids, blksize):
            dat = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, q, kpts, shls_slice, out=buf)
            yield dat, p0, p1


class AFTDF(lib.StreamObject, AFTDFMixin):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
# For nuclear attraction integrals using Ewald-like technique.
# Set to 0 to switch off Ewald tech and use the regular reciprocal space
# method (solving Poisson equation of nuclear charges in reciprocal space).
        if cell.dimension == 0:
            self.eta = 0.2
        else:
            ke_cutoff = tools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
            ke_cutoff = ke_cutoff[:cell.dimension].min()
            self.eta = max(estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision),
                           estimate_eta(cell, cell.precision))
        self.kpts = kpts

        # to mimic molecular DF object
        self.blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

        # The following attributes are not input options.
        self._rsh_df = {}  # Range separated Coulomb DF objects
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, numpy.prod(self.mesh))
        logger.info(self, 'eta = %s', self.eta)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)
        return self

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self._rsh_df = {}
        return self

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        cell = self.cell
        if not cell.has_ecp():
            logger.warn(self, 'AFTDF integrals are found in all-electron '
                        'calculation.  It often causes huge error.\n'
                        'Recommended methods are DF or MDF. In SCF calculation, '
                        'they can be initialized as\n'
                        '        mf = mf.density_fit()\nor\n'
                        '        mf = mf.mix_density_fit()')

        if cell.dimension > 0:
            if cell.ke_cutoff is None:
                ke_cutoff = tools.mesh_to_cutoff(cell.lattice_vectors(), self.mesh)
                ke_cutoff = ke_cutoff[:cell.dimension].min()
            else:
                ke_cutoff = numpy.min(cell.ke_cutoff)
            ke_guess = estimate_ke_cutoff(cell, cell.precision)
            mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
            if ke_cutoff < ke_guess * KE_SCALING:
                logger.warn(self, 'ke_cutoff/mesh (%g / %s) is not enough for AFTDF '
                            'to get integral accuracy %g.\nCoulomb integral error '
                            'is ~ %.2g Eh.\nRecommended ke_cutoff/mesh are %g / %s.',
                            ke_cutoff, self.mesh, cell.precision,
                            error_for_ke_cutoff(cell, ke_cutoff), ke_guess, mesh_guess)
        else:
            mesh_guess = numpy.copy(self.mesh)

        if cell.dimension < 3:
            err = numpy.exp(-0.436392335*min(self.mesh[cell.dimension:]) - 2.99944305)
            err *= cell.nelectron
            meshz = pbcgto.cell._mesh_inf_vaccum(cell)
            mesh_guess[cell.dimension:] = int(meshz)
            if err > cell.precision*10:
                logger.warn(self, 'mesh %s of AFTDF may not be enough to get '
                            'integral accuracy %g for %dD PBC system.\n'
                            'Coulomb integral error is ~ %.2g Eh.\n'
                            'Recommended mesh is %s.',
                            self.mesh, cell.precision, cell.dimension, err, mesh_guess)
            if any(x/meshz > 1.1 for x in cell.mesh[cell.dimension:]):
                meshz = pbcgto.cell._mesh_inf_vaccum(cell)
                logger.warn(self, 'setting mesh %s of AFTDF too high in non-periodic direction '
                            '(=%s) can result in an unnecessarily slow calculation.\n'
                            'For coulomb integral error of ~ %.2g Eh in %dD PBC, \n'
                            'a recommended mesh for non-periodic direction is %s.',
                            self.mesh, self.mesh[cell.dimension:], cell.precision,
                            cell.dimension, mesh_guess[cell.dimension:])
        return self

    def build(self):
        return self.check_sanity()

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            return _sub_df_jk_(self, dm, hermi, kpts, kpts_band,
                               with_j, with_k, omega, exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return aft_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                  with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = aft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = aft_ao2mo.get_eri
    ao2mo = get_mo_eri = aft_ao2mo.general
    ao2mo_7d = aft_ao2mo.ao2mo_7d
    get_ao_pairs_G = get_ao_pairs = aft_ao2mo.get_ao_pairs_G
    get_mo_pairs_G = get_mo_pairs = aft_ao2mo.get_mo_pairs_G

    def update_mf(self, mf):
        mf = copy.copy(mf)
        mf.with_df = self
        return mf

    def prange(self, start, stop, step):
        '''This is a hook for MPI parallelization. DO NOT use it out of the
        scope of AFTDF/GDF/MDF.
        '''
        return lib.prange(start, stop, step)

################################################################################
# With this function to mimic the molecular DF.loop function, the pbc gamma
# point DF object can be used in the molecular code
    def loop(self, blksize=None):
        cell = self.cell
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise RuntimeError('ERIs of PBC-2D systems are not positive '
                               'definite. Current API only supports postive '
                               'definite ERIs.')

        if blksize is None:
            blksize = self.blockdim
        # coulG of 1D and 2D has negative elements.
        coulG = self.weighted_coulG()
        Lpq = None
        for pqkR, pqkI, p0, p1 in self.pw_loop(aosym='s2', blksize=blksize):
            vG = numpy.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
            Lpq = lib.transpose(pqkR, out=Lpq)
            yield Lpq
            Lpq = lib.transpose(pqkI, out=Lpq)
            yield Lpq

    def get_naoaux(self):
        mesh = numpy.asarray(self.mesh)
        ngrids = numpy.prod(mesh)
        return ngrids * 2

def _sub_df_jk_(dfobj, dm, hermi=1, kpts=None, kpts_band=None,
                with_j=True, with_k=True, omega=None, exxdiv=None):
    key = '%.6f' % omega
    if key in dfobj._rsh_df:
        rsh_df = dfobj._rsh_df[key]
    else:
        rsh_df = dfobj._rsh_df[key] = copy.copy(dfobj).reset()
        logger.info(dfobj, 'Create RSH-%s object %s for omega=%s',
                    dfobj.__class__.__name__, rsh_df, omega)
    with rsh_df.cell.with_range_coulomb(omega):
        return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                             omega=None, exxdiv=exxdiv)

del(CUTOFF, PRECISION)
