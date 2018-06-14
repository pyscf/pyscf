#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import time
import copy
import numpy
import scipy.misc
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo, estimate_ke_cutoff, error_for_ke_cutoff
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import incore
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df import aft_ao2mo
from pyscf import __config__


CUTOFF = getattr(__config__, 'pbc_df_aft_estimate_eta_cutoff', 1e-12)
ETA_MIN = getattr(__config__, 'pbc_df_aft_estimate_eta_min', 0.2)
PRECISION = getattr(__config__, 'pbc_df_aft_estimate_eta_precision', 1e-8)
KE_SCALING = getattr(__config__, 'pbc_df_aft_ke_cutoff_scaling', 0.75)

def estimate_eta(cell, cutoff=CUTOFF):
    '''The exponent of the smooth gaussian model density, requiring that at
    boundary, density ~ 4pi rmax^2 exp(-eta/2*rmax^2) ~ 1e-12
    '''
    # r^5 to guarantee at least up to f shell converging at boundary
    eta = max(numpy.log(4*numpy.pi*cell.rcut**5/cutoff)/cell.rcut**2*2,
              ETA_MIN)
    return eta

def estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=PRECISION):
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = numpy.linalg.norm(b[0]) / (2*numpy.pi)
    elif cell.dimension == 2:
        w = numpy.linalg.norm(numpy.cross(b[0], b[1])) / (2*numpy.pi)**2
    else:
        w = abs(numpy.linalg.det(b)) / (2*numpy.pi)**3
    lmax = min(3, numpy.max(cell._bas[:,gto.ANG_OF]))
    fac = scipy.misc.factorial2(lmax*2+1)
    eta = ke_cutoff / ((4+2*lmax)*numpy.log(2*ke_cutoff) -
                       2*numpy.log(3*fac*precision/(32*numpy.pi**2*w)))
    return eta

def estimate_ke_cutoff_for_eta(cell, eta, precision=PRECISION):
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = numpy.linalg.norm(b[0]) / (2*numpy.pi)
    elif cell.dimension == 2:
        w = numpy.linalg.norm(numpy.cross(b[0], b[1])) / (2*numpy.pi)**2
    else:
        w = abs(numpy.linalg.det(b)) / (2*numpy.pi)**3
    lmax = min(3, numpy.max(cell._bas[:,gto.ANG_OF]))
    fac = scipy.misc.factorial2(lmax*2+1)
    Ecut = 2 * eta * (8+4*lmax - numpy.log(3*fac*precision*(4*eta)**lmax/(32*numpy.pi**2*w)))
    return Ecut

def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = t1 = (time.clock(), time.time())

    mesh = numpy.asarray(mydf.mesh)
    nkpts = len(kpts_lst)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    charges = cell.atom_charges()

    kpt_allow = numpy.zeros(3)
    if mydf.eta == 0:
        if cell.dimension > 0:
            ke_guess = estimate_ke_cutoff(cell, cell.precision)
            mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
            if numpy.any(mesh < mesh_guess*.8):
                logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                            'to get integral accuracy %g.\nRecommended mesh is %s.',
                            mesh, cell.precision, mesh_guess)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

        vpplocG = pseudo.pp_int.get_gth_vlocG_part1(cell, Gv)
        vpplocG = -numpy.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
        v1 = -vpplocG.copy()

        if cell.dimension == 1 or cell.dimension == 2:
            G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv)
            coulG = 4*numpy.pi / numpy.linalg.norm(Gv[G0idx], axis=1)**2
            vpplocG[G0idx] += charges.sum() * SI_on_z * coulG

        vpplocG *= kws
        vG = vpplocG
        vj = numpy.zeros((nkpts,nao_pair), dtype=numpy.complex128)

    else:
        if cell.dimension > 0:
            ke_guess = estimate_ke_cutoff_for_eta(cell, mydf.eta, cell.precision)
            mesh_guess = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_guess)
            if numpy.any(mesh < mesh_guess*.8):
                logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                            'to get integral accuracy %g.\nRecommended mesh is %s.',
                            mesh, cell.precision, mesh_guess)
            mesh_min = numpy.min((mesh_guess[:cell.dimension],
                                  mesh[:cell.dimension]), axis=0)
            mesh[:cell.dimension] = mesh_min.astype(int)
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

        nuccell = copy.copy(cell)
        half_sph_norm = .5/numpy.sqrt(numpy.pi)
        norm = half_sph_norm/gto.gaussian_int(2, mydf.eta)
        chg_env = [mydf.eta, norm]
        ptr_eta = cell._env.size
        ptr_norm = ptr_eta + 1
        chg_bas = [[ia, 0, 1, 1, 0, ptr_eta, ptr_norm, 0] for ia in range(cell.natm)]
        nuccell._atm = cell._atm
        nuccell._bas = numpy.asarray(chg_bas, dtype=numpy.int32)
        nuccell._env = numpy.hstack((cell._env, chg_env))

        # PP-loc part1 is handled by fakenuc in _int_nuc_vloc
        vj = lib.asarray(mydf._int_nuc_vloc(nuccell, kpts_lst))
        t0 = t1 = log.timer_debug1('vnuc pass1: analytic int', *t0)

        coulG = tools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv) * kws
        aoaux = ft_ao.ft_ao(nuccell, Gv)
        vG = numpy.einsum('i,xi->x', -charges, aoaux) * coulG

        if cell.dimension == 1 or cell.dimension == 2:
            G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv)
            vG[G0idx] += charges.sum() * SI_on_z * coulG[G0idx]

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
            vj_kpts.append(lib.unpack_tril(vj[k].real.copy()))
        else:
            vj_kpts.append(lib.unpack_tril(vj[k]))

    if kpts is None or numpy.shape(kpts) == (3,):
        vj_kpts = vj_kpts[0]
    return numpy.asarray(vj_kpts)

def _int_nuc_vloc(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    nkpts = len(kpts)

# Use the 3c2e code with steep s gaussians to mimic nuclear density
    fakenuc = _fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)

    kptij_lst = numpy.hstack((kpts,kpts)).reshape(-1,2,3)
    buf = incore.aux_e2(cell, fakenuc, intor, aosym=aosym, comp=comp,
                        kptij_lst=kptij_lst)

    charge = cell.atom_charges()
    charge = numpy.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    nao = cell.nao_nr()
    nchg = len(charge)
    if aosym == 's1':
        nao_pair = nao**2
    else:
        nao_pair = nao*(nao+1)//2
    if comp == 1:
        buf = buf.reshape(nkpts,nao_pair,nchg)
        mat = numpy.einsum('kxz,z->kx', buf, charge)
    else:
        buf = buf.reshape(nkpts,comp,nao_pair,nchg)
        mat = numpy.einsum('kcxz,z->kcx', buf, charge)

    if cell.dimension != 0 and intor in ('int3c2e', 'int3c2e_sph',
                                         'int3c2e_cart'):
        assert(comp == 1)
        charge = -cell.atom_charges()

        if cell.dimension == 1 or cell.dimension == 2:
            Gv, Gvbase, kws = cell.get_Gv_weights(mydf.mesh)
            G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv)
            ZSI = numpy.einsum("i,ix->x", charge, cell.get_SI(Gv[G0idx]))
            ZSI -= numpy.einsum('i,xi->x', charge, ft_ao.ft_ao(nuccell, Gv[G0idx]))
            coulG = 4*numpy.pi / numpy.linalg.norm(Gv[G0idx], axis=1)**2
            nucbar = numpy.einsum('i,i,i,i', ZSI.conj(), coulG, kws[G0idx], SI_on_z)
            if abs(kpts).sum() < 1e-9:
                nucbar = nucbar.real
        else: # cell.dimension == 3
            nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
            nucbar *= numpy.pi/cell.vol

        ovlp = cell.pbc_intor('int1e_ovlp', 1, lib.HERMITIAN, kpts)
        for k in range(nkpts):
            if aosym == 's1':
                mat[k] -= nucbar * ovlp[k].reshape(nao_pair)
            else:
                mat[k] -= nucbar * lib.pack_tril(ovlp[k])

    return mat

get_pp_loc_part1 = get_nuc

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    vloc1 = mydf.get_nuc(kpts_lst)
    vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    vpp = pseudo.pp_int.get_pp_nl(cell, kpts_lst)
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp

def weighted_coulG(mydf, kpt=numpy.zeros(3), exx=False, mesh=None):
    cell = mydf.cell
    if mesh is None:
        mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = tools.get_coulG(cell, kpt, exx, mydf, mesh, Gv)
    coulG *= kws
    return coulG


class AFTDF(lib.StreamObject):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
# For nuclear attraction integrals using Ewald-like technique.
# Set to 0 to swith off Ewald tech and use the regular reciprocal space
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

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, numpy.prod(self.mesh))
        logger.info(self, 'eta = %s', self.eta)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)
        return self

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        cell = self.cell
        if cell.low_dim_ft_type is not None:
            raise ValueError('AFTDF detected a non-None cell.low_dim_ft_type! '
                             'The cell.low_dim_ft_type should only be \nset when '
                             'using with_df = FFTDF. Please set mf.with_df equal '
                             'to FFTDF or set cell.low_dim_ft_type \n(= %s) to None. '
                              % (cell.low_dim_ft_type))
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
            meshz = (numpy.log(cell.nelectron/cell.precision)-2.99944305)/0.436392335
            mesh_guess[cell.dimension:] = int(meshz)
            if err > cell.precision*10:
                logger.warn(self, 'mesh %s of AFTDF may not be enough to get '
                            'integral accuracy %g for %dD PBC system.\n'
                            'Coulomb integral error is ~ %.2g Eh.\n'
                            'Recommended mesh is %s.',
                            self.mesh, cell.precision, cell.dimension, err, mesh_guess)
            if (cell.mesh[cell.dimension:]/(1.*meshz) > 1.1).any():
                meshz = (numpy.log(cell.nelectron/cell.precision)-2.99944305)/0.436392335
                logger.warn(self, 'setting mesh %s of AFTDF too high in non-periodic direction '
                            '(=%s) can result in an unnecessarily slow calculation.\n'
                            'For coulomb integral error of ~ %.2g Eh in %dD PBC, \n'
                            'a recommended mesh for non-periodic direction is %s.',
                            self.mesh, self.mesh[cell.dimension:], cell.precision,
                            cell.dimension, mesh_guess[cell.dimension:])
        return self

# TODO: Put Gv vector in the arguments
    def pw_loop(self, mesh=None, kpti_kptj=None, q=None, shls_slice=None,
                max_memory=2000, aosym='s1', blksize=None,
                intor='GTO_ft_ovlp', comp=1):
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
        b = cell.reciprocal_vectors()
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert(shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj

        if (abs(q).sum() < 1e-6 and (cell.dimension == 1 or cell.dimension == 2)):
            if aosym == 's2':
                s = lib.pack_tril(cell.pbc_intor('int1e_ovlp', kpt=kptj))
            else:
                s = cell.pbc_intor('int1e_ovlp', kpt=kptj).ravel()
        else:
            s = None

        if blksize is None:
            blksize = min(max(64, int(max_memory*1e6*.75/(nij*16*comp))), 16384)
            sublk = int(blksize//4)
        else:
            sublk = blksize
        buf = numpy.empty(nij*blksize*comp, dtype=numpy.complex128)
        pqkRbuf = numpy.empty(nij*sublk*comp)
        pqkIbuf = numpy.empty(nij*sublk*comp)

        for p0, p1 in self.prange(0, ngrids, blksize):
            #aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], shls_slice, aosym,
            #                       b, Gvbase, gxyz[p0:p1], mesh, (kpti, kptj), q)
            aoao = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                         b, gxyz[p0:p1], Gvbase, q,
                                         kptj.reshape(1,3), intor, comp, out=buf)[0]
            aoao = aoao.reshape(p1-p0,nij)
            if s is not None:  # to remove the divergent integrals
                G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv[p0:p1])
                aoao[G0idx] -= numpy.einsum('g,i->gi', SI_on_z, s)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                if comp == 1:
                    pqkR = numpy.ndarray((nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao[i0:i1].real.T
                    pqkI[:] = aoao[i0:i1].imag.T
                else:
                    pqkR = numpy.ndarray((comp,nij,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((comp,nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao[i0:i1].real.transpose(0,2,1)
                    pqkI[:] = aoao[i0:i1].imag.transpose(0,2,1)
                yield (pqkR, pqkI, p0+i0, p0+i1)

    def ft_loop(self, mesh=None, q=numpy.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1):
        '''
        Fourier transform iterator for all kpti which satisfy
            2pi*N = (kpts - kpti - q)*a,  N = -1, 0, 1
        '''
        cell = self.cell
        if mesh is None:
            mesh = self.mesh
        if kpts is None:
            assert(is_zero(q))
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        ao_loc = cell.ao_loc_nr()
        b = cell.reciprocal_vectors()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
        ngrids = gxyz.shape[0]

        if shls_slice is None:
            shls_slice = (0, cell.nbas, 0, cell.nbas)
        if aosym == 's2':
            assert(shls_slice[2] == 0)
            i0 = ao_loc[shls_slice[0]]
            i1 = ao_loc[shls_slice[1]]
            nij = i1*(i1+1)//2 - i0*(i0+1)//2
        else:
            ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
            nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
            nij = ni*nj

        if (abs(q).sum() < 1e-6 and intor[:11] == 'GTO_ft_ovlp' and
            (cell.dimension == 1 or cell.dimension == 2)):
            s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
            if aosym == 's2':
                s = [lib.pack_tril(x) for x in s]
        else:
            s = None

        blksize = max(16, int(max_memory*.9e6/(nij*nkpts*16*comp)))
        blksize = min(blksize, ngrids, 16384)
        buf = numpy.empty(nkpts*nij*blksize*comp, dtype=numpy.complex128)

        for p0, p1 in self.prange(0, ngrids, blksize):
            dat = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym,
                                        b, gxyz[p0:p1], Gvbase, q, kpts,
                                        intor, comp, out=buf)

            if s is not None:  # to remove the divergent part in 1D/2D systems
                G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv[p0:p1])
                if SI_on_z.size > 0:
                    for k, kpt in enumerate(kpts):
                        dat[k][G0idx] -= numpy.einsum('g,...->g...', SI_on_z, s[k])

            yield dat, p0, p1

    weighted_coulG = weighted_coulG
    _int_nuc_vloc = _int_nuc_vloc
    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts

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
        if blksize is None:
            blksize = self.blockdim
        Lpq = None
        coulG = self.weighted_coulG()
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


# Since the real-space lattice-sum for nuclear attraction is not implemented,
# use the 3c2e code with steep gaussians to mimic nuclear density
def _fake_nuc(cell):
    fakenuc = gto.Mole()
    fakenuc._atm = cell._atm.copy()
    fakenuc._atm[:,gto.PTR_COORD] = numpy.arange(gto.PTR_ENV_START,
                                                 gto.PTR_ENV_START+cell.natm*3,3)
    _bas = []
    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
    ptr = gto.PTR_ENV_START + cell.natm * 3
    half_sph_norm = .5/numpy.sqrt(numpy.pi)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            eta = .5 / rloc**2
        else:
            eta = 1e16
        norm = half_sph_norm/gto.gaussian_int(2, eta)
        _env.extend([eta, norm])
        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        ptr += 2
    fakenuc._bas = numpy.asarray(_bas, dtype=numpy.int32)
    fakenuc._env = numpy.asarray(numpy.hstack(_env), dtype=numpy.double)
    fakenuc.rcut = cell.rcut
    return fakenuc

del(CUTOFF, PRECISION)


if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1'
    cell.a = numpy.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [20]*3
    cell.build()
    k = numpy.ones(3)*.25
    v1 = AFTDF(cell).get_pp(k)
    print(abs(v1).sum(), 21.7504294462)
