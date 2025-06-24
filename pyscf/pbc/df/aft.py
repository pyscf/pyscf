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


import contextlib
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo, error_for_ke_cutoff
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.gto.pseudo import pp_int
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import aft_jk
from pyscf.pbc.df import aft_ao2mo
from pyscf.pbc.df.incore import Int3cBuilder
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools import pbc as pbctools
from pyscf import __config__


KE_SCALING = getattr(__config__, 'pbc_df_aft_ke_cutoff_scaling', 0.75)
RCUT_THRESHOLD = getattr(__config__, 'pbc_scf_rsjk_rcut_threshold', 2.0)

def estimate_eta_min(cell, cutoff=None):
    '''Given rcut the boundary of repeated images of the cell, estimates the
    minimal exponent of the smooth compensated gaussian model charge, requiring
    that at boundary, density ~ 4pi rmax^2 exp(-eta/2*rmax^2) < cutoff
    '''
    from pyscf.pbc.df.gdf_builder import estimate_eta_min
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.gdf_builder.estimate_eta_min instead.')
    return estimate_eta_min(cell, cutoff)

estimate_eta = estimate_eta_min

def estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''Given ke_cutoff, the upper bound of eta to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    from pyscf.pbc.df.gdf_builder import estimate_eta_for_ke_cutoff
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.gdf_builder.estimate_eta_for_ke_cutoff instead.')
    return estimate_eta_for_ke_cutoff(cell, ke_cutoff, precision)

def estimate_ke_cutoff_for_eta(cell, eta, precision=None):
    '''Given eta, the lower bound of ke_cutoff to produce the required
    precision in AFTDF Coulomb integrals.
    '''
    from pyscf.pbc.df.gdf_builder import estimate_ke_cutoff_for_eta
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.gdf_builder.estimate_ke_cutoff_for_eta instead.')
    return estimate_ke_cutoff_for_eta(cell, eta, precision)

def estimate_omega_min(cell, cutoff=None):
    '''Given cell.rcut the boundary of repeated images of the cell, estimates
    the minimal omega for the attenuated Coulomb interactions, requiring that at
    boundary the Coulomb potential of a point charge < cutoff
    '''
    from pyscf.pbc.df.rsdf_builder import estimate_omega_min
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.rsdf_builder.estimate_omega_min instead.')
    return estimate_omega_min(cell, cutoff)

estimate_omega = estimate_omega_min

def estimate_ke_cutoff_for_omega(cell, omega, precision=None):
    '''Energy cutoff for AFTDF to converge attenuated Coulomb in moment space
    '''
    from pyscf.pbc.df.rsdf_builder import estimate_ke_cutoff_for_omega
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.rsdf_builder.estimate_ke_cutoff_for_omega instead.')
    return estimate_ke_cutoff_for_omega(cell, omega, precision)

def estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision=None):
    '''The minimal omega in attenuated Coulombl given energy cutoff
    '''
    from pyscf.pbc.df.rsdf_builder import estimate_omega_for_ke_cutoff
    logger.warn(cell, 'Function deprecated. '
                'Call pbc.df.rsdf_builder.estimate_omega_for_ke_cutoff instead.')
    return estimate_omega_for_ke_cutoff(cell, ke_cutoff, precision)


def _get_pp_loc_part1(mydf, kpts=None, with_pseudo=True):
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    cell = mydf.cell
    mesh = np.asarray(mydf.mesh)
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    kpt_allow = np.zeros(3)
    if cell.dimension > 0:
        ke_guess = estimate_ke_cutoff(cell, cell.precision)
        mesh_guess = cell.cutoff_to_mesh(ke_guess)
        if np.any(mesh < mesh_guess*KE_SCALING):
            logger.warn(mydf, 'mesh %s is not enough for AFTDF.get_nuc function '
                        'to get integral accuracy %g.\nRecommended mesh is %s.',
                        mesh, cell.precision, mesh_guess)
    log.debug1('aft.get_pp_loc_part1 mesh = %s', mesh)
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)

    if with_pseudo:
        vpplocG = pp_int.get_gth_vlocG_part1(cell, Gv)
        vpplocG = -np.einsum('ij,ij->j', cell.get_SI(Gv), vpplocG)
    else:
        fakenuc = _fake_nuc(cell, with_pseudo=with_pseudo)
        aoaux = ft_ao.ft_ao(fakenuc, Gv)
        charges = cell.atom_charges()
        coulG = pbctools.get_coulG(cell, kpt_allow, mesh=mesh, Gv=Gv)
        vpplocG = np.einsum('i,xi,x->x', -charges, aoaux, coulG)

    vpplocG *= kws
    vGR = vpplocG.real
    vGI = vpplocG.imag

    vjR = np.zeros((nkpts, nao_pair))
    vjI = np.zeros((nkpts, nao_pair))
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for Gpq, p0, p1 in mydf.ft_loop(mesh, kpt_allow, kpts, aosym='s2',
                                    max_memory=max_memory, return_complex=False):
        # shape of Gpq (nkpts, nGv, nao_pair)
        for k, (GpqR, GpqI) in enumerate(zip(*Gpq)):
            # rho_ij(G) nuc(-G) / G^2
            # = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
            vjR[k] += np.einsum('k,kx->x', vGR[p0:p1], GpqR)
            vjR[k] += np.einsum('k,kx->x', vGI[p0:p1], GpqI)
            if not is_zero(kpts[k]):
                vjI[k] += np.einsum('k,kx->x', vGR[p0:p1], GpqI)
                vjI[k] -= np.einsum('k,kx->x', vGI[p0:p1], GpqR)
        t1 = log.timer_debug1('contracting Vnuc [%s:%s]'%(p0, p1), *t1)
    log.timer_debug1('contracting Vnuc', *t0)

    vj_kpts = []
    for k, kpt in enumerate(kpts):
        if is_zero(kpt):
            vj_kpts.append(lib.unpack_tril(vjR[k]))
        else:
            vj_kpts.append(lib.unpack_tril(vjR[k]+vjI[k]*1j))
    if is_single_kpt:
        vj_kpts = vj_kpts[0]
    return np.asarray(vj_kpts)

def _check_kpts(mydf, kpts):
    '''Check if the argument kpts is a single k-point'''
    if kpts is None:
        kpts = np.asarray(mydf.kpts)
        # mydf.kpts is initialized to np.zeros((1,3)). Here is only a guess
        # based on the value of mydf.kpts.
        is_single_kpt = kpts.ndim == 1 or is_zero(kpts)
    else:
        kpts = np.asarray(kpts)
        is_single_kpt = kpts.ndim == 1
    kpts = kpts.reshape(-1,3)
    return kpts, is_single_kpt

def _int_nuc_vloc(mydf, nuccell, kpts, intor='int3c2e', aosym='s2', comp=1):
    '''Vnuc - Vloc'''
    raise DeprecationWarning

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.

    Kwargs:
        mesh: custom mesh grids. By default mesh is determined by the
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    kpts, is_single_kpt = _check_kpts(mydf, kpts)
    cell = mydf.cell
    vpp = _get_pp_loc_part1(mydf, kpts, with_pseudo=True)
    t1 = logger.timer_debug1(mydf, 'get_pp_loc_part1', *t0)
    pp2builder = _IntPPBuilder(cell, kpts)
    vpp += pp2builder.get_pp_loc_part2()
    t1 = logger.timer_debug1(mydf, 'get_pp_loc_part2', *t1)
    vpp += pp_int.get_pp_nl(cell, kpts)
    t1 = logger.timer_debug1(mydf, 'get_pp_nl', *t1)
    if is_single_kpt:
        vpp = vpp[0]
    logger.timer(mydf, 'get_pp', *t0)
    return vpp


def get_nuc(mydf, kpts=None):
    '''Get the periodic nuc-el AO matrix, with G=0 removed.

    Kwargs:
        function _guess_eta from module pbc.df.gdf_builder.
    '''
    t0 = (logger.process_clock(), logger.perf_counter())
    nuc = _get_pp_loc_part1(mydf, kpts, with_pseudo=False)
    logger.timer(mydf, 'get_nuc', *t0)
    return nuc


def weighted_coulG(mydf, kpt=np.zeros(3), exx=False, mesh=None, omega=None):
    '''Weighted regular Coulomb kernel'''
    cell = mydf.cell
    if mesh is None:
        mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    coulG = tools.get_coulG(cell, kpt, exx, mydf, mesh, Gv, omega=omega)
    coulG *= kws
    return coulG


def _fake_nuc(cell, with_pseudo=True):
    '''A fake cell with steep gaussians to mimic nuclear density
    '''
    fakenuc = cell.copy(deep=False)
    fakenuc._atm = cell._atm.copy()
    fakenuc._atm[:,gto.PTR_COORD] = np.arange(gto.PTR_ENV_START,
                                              gto.PTR_ENV_START+cell.natm*3,3)
    _bas = []
    _env = [0]*gto.PTR_ENV_START + [cell.atom_coords().ravel()]
    ptr = gto.PTR_ENV_START + cell.natm * 3
    half_sph_norm = .5/np.sqrt(np.pi)
    for ia in range(cell.natm):
        symb = cell.atom_symbol(ia)
        if with_pseudo and symb in cell._pseudo:
            pp = cell._pseudo[symb]
            rloc, nexp, cexp = pp[1:3+1]
            eta = .5 / rloc**2
        else:
            eta = 1e16
        norm = half_sph_norm/gto.gaussian_int(2, eta)
        _env.extend([eta, norm])
        _bas.append([ia, 0, 1, 1, 0, ptr, ptr+1, 0])
        ptr += 2
    fakenuc._bas = np.asarray(_bas, dtype=np.int32)
    fakenuc._env = np.asarray(np.hstack(_env), dtype=np.double)
    fakenuc.rcut = 0.1
    return fakenuc


def _estimate_ke_cutoff(alpha, l, c, precision, omega=0):
    '''Energy cutoff estimation for 4-center Coulomb repulsion integrals'''
    norm_ang = ((2*l+1)/(4*np.pi))**2
    fac = 8*np.pi**5 * c**4*norm_ang / (2*alpha)**(4*l+2) / precision
    Ecut = 20.
    if omega <= 0:
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * 2*alpha
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * 2*alpha
    else:
        theta = 1./(1./(2*alpha) + 1./(2*omega**2))
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * theta
        Ecut = np.log(fac * (Ecut*.5)**(2*l-.5) + 1.) * theta
    return Ecut

def estimate_ke_cutoff(cell, precision=None):
    '''Energy cutoff estimation for 4-center Coulomb repulsion integrals'''
    if cell.nbas == 0:
        return 0.
    if precision is None:
        precision = cell.precision
    exps, cs = pbcgto.cell._extract_pgto_params(cell, 'max')
    ls = cell._bas[:,gto.ANG_OF]
    cs = gto.gto_norm(ls, exps)
    Ecut = _estimate_ke_cutoff(exps, ls, cs, precision)
    return Ecut.max()


class _IntPPBuilder(Int3cBuilder):
    '''3-center integral builder for pp loc part2 only
    '''
    def __init__(self, cell, kpts=np.zeros((1,3))):
        # cache ovlp_mask which are reused for different types of intor
        self._supmol = None
        self._ovlp_mask = None
        self._cell0_ovlp_mask = None
        Int3cBuilder.__init__(self, cell, None, kpts)

    def get_ovlp_mask(self, cutoff, supmol=None, cintopt=None):
        if self._ovlp_mask is None or supmol is not self._supmol:
            self._ovlp_mask, self._cell0_ovlp_mask = \
                    Int3cBuilder.get_ovlp_mask(self, cutoff, supmol, cintopt)
            self._supmol = supmol
        return self._ovlp_mask, self._cell0_ovlp_mask

    def build(self):
        pass

    def get_pp_loc_part2(self):
        log = logger.new_logger(self)
        cell = self.cell
        kpts = self.kpts
        nkpts = len(kpts)

        self.bvk_kmesh = kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        log.debug('kmesh for bvk-cell = %s', kmesh)

        self.rs_cell = rs_cell = ft_ao._RangeSeparatedCell.from_cell(
            cell, self.ke_cutoff, RCUT_THRESHOLD, verbose=log)

        intors = ('int3c2e', 'int3c1e', 'int3c1e_r2_origk',
                  'int3c1e_r4_origk', 'int3c1e_r6_origk')
        fake_cells = {}
        for cn in range(1, 5):
            fake_cell = pp_int.fake_cell_vloc(cell, cn)
            if fake_cell.nbas > 0:
                fake_cells[cn] = fake_cell

        if not fake_cells:
            if any(cell.atom_symbol(ia) in cell._pseudo for ia in range(cell.natm)):
                pass
            else:
                lib.logger.warn(cell, 'cell.pseudo was specified but its elements %s '
                                'were not found in the system.', cell._pseudo.keys())
            nao = cell.nao
            vpploc = np.zeros((nkpts, nao, nao))
            return vpploc

        rcut = self._estimate_rcut_3c1e(rs_cell, fake_cells)
        supmol = ft_ao.ExtendedMole.from_cell(rs_cell, kmesh, rcut.max(), log)
        self.supmol = supmol.strip_basis(rcut+1.)
        log.debug('sup-mol nbas = %d cGTO = %d pGTO = %d',
                  supmol.nbas, supmol.nao, supmol.npgto_nr())

        bufR = 0
        bufI = 0
        for cn, fake_cell in fake_cells.items():
            int3c = self.gen_int3c_kernel(
                intors[cn], 's2', comp=1, j_only=True, auxcell=fake_cell)
            vR, vI = int3c()
            bufR += np.einsum('...i->...', vR)
            if vI is not None:
                bufI += np.einsum('...i->...', vI)

        buf = (bufR + bufI * 1j).reshape(nkpts,-1)
        vpploc = []
        for k, kpt in enumerate(kpts):
            v = lib.unpack_tril(buf[k])
            if is_zero(kpt):  # gamma_point:
                v = v.real
            vpploc.append(v)
        return vpploc

    def _estimate_rcut_3c1e(self, cell, fake_cells):
        '''Estimate rcut for pp-loc part2 based on 3-center overlap integrals.
        '''
        precision = cell.precision
        exps = np.array([e.min() for e in cell.bas_exps()])
        if exps.size == 0:
            return np.zeros(1)

        ls = cell._bas[:,gto.ANG_OF]
        cs = gto.gto_norm(ls, exps)
        ai_idx = exps.argmin()
        ai = exps[ai_idx]
        li = cell._bas[ai_idx,gto.ANG_OF]
        ci = cs[ai_idx]

        r0 = cell.rcut  # initial guess
        rcut = []
        for lk, fake_cell in fake_cells.items():
            nuc_exps = np.hstack(fake_cell.bas_exps())
            ak_idx = nuc_exps.argmin()
            ak = nuc_exps[ak_idx]
            ck = abs(fake_cell._env[fake_cell._bas[ak_idx,gto.PTR_COEFF]])

            aij = ai + exps
            ajk = exps + ak
            aijk = aij + ak
            aijk1 = aijk**-.5
            theta = 1./(1./aij + 1./ak)
            norm_ang = ((2*li+1)*(2*ls+1))**.5/(4*np.pi)
            c1 = ci * cs * ck * norm_ang
            sfac = aij*exps/(aij*exps + ai*theta)
            rfac = ak / (aij * ajk)
            fl = 2
            fac = 2**(li+1)*np.pi**2.5 * aijk1**3 * c1 / theta * fl / precision

            r0 = (np.log(fac * r0 * (rfac*exps*r0+aijk1)**li *
                         (rfac*ai*r0+aijk1)**ls + 1.) / (sfac*theta))**.5
            r0 = (np.log(fac * r0 * (rfac*exps*r0+aijk1)**li *
                         (rfac*ai*r0+aijk1)**ls + 1.) / (sfac*theta))**.5
            rcut.append(r0)
        return np.max(rcut, axis=0)


class AFTDFMixin:

    weighted_coulG = weighted_coulG
    _int_nuc_vloc = _int_nuc_vloc

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
            kpti = kptj = np.zeros(3)
        else:
            kpti, kptj = kpti_kptj
        if q is None:
            q = kptj - kpti

        ao_loc = cell.ao_loc_nr()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
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
        buf = np.empty(nij*blksize*comp, dtype=np.complex128)
        pqkRbuf = np.empty(nij*sublk*comp)
        pqkIbuf = np.empty(nij*sublk*comp)

        if bvk_kmesh is None:
            bvk_kmesh = k2gamma.kpts_to_kmesh(cell, [kpti, kptj])
        rcut = ft_ao.estimate_rcut(cell)
        supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
        supmol = supmol.strip_basis(rcut)
        ft_kern = supmol.gen_ft_kernel(aosym, intor=intor, comp=comp)

        for p0, p1 in self.prange(0, ngrids, blksize):
            aoaoR, aoaoI = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, q,
                                   kptj.reshape(1, 3), shls_slice, out=buf)
            aoaoR = aoaoR.reshape(comp,p1-p0,nij)
            aoaoI = aoaoI.reshape(comp,p1-p0,nij)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                if comp == 1:
                    pqkR = np.ndarray((nij,nG), buffer=pqkRbuf)
                    pqkI = np.ndarray((nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoaoR[0,i0:i1].T
                    pqkI[:] = aoaoI[0,i0:i1].T
                else:
                    pqkR = np.ndarray((comp,nij,nG), buffer=pqkRbuf)
                    pqkI = np.ndarray((comp,nij,nG), buffer=pqkIbuf)
                    pqkR[:] = aoaoR[:,i0:i1].transpose(0,2,1)
                    pqkI[:] = aoaoI[:,i0:i1].transpose(0,2,1)
                yield (pqkR, pqkI, p0+i0, p0+i1)

    def ft_loop(self, mesh=None, q=np.zeros(3), kpts=None, shls_slice=None,
                max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1,
                bvk_kmesh=None, return_complex=True):
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
        kpts = np.asarray(kpts)
        nkpts = len(kpts)

        ao_loc = cell.ao_loc_nr()
        Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
        gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
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
        #TODO:
        # ke_cutoff = pbctools.mesh_to_cutoff(cell.lattice_vectors(), mesh)
        # rs_cell = ft_ao._RangeSeparatedCell.from_cell(cell, ke_cutoff, ft_ao.RCUT_THRESHOLD)

        rcut = ft_ao.estimate_rcut(cell)
        supmol = ft_ao.ExtendedMole.from_cell(cell, bvk_kmesh, rcut.max())
        supmol = supmol.strip_basis(rcut)
        ft_kern = supmol.gen_ft_kernel(aosym, intor=intor, comp=comp,
                                       return_complex=return_complex)

        blksize = max(16, int(max_memory*.9e6/(nij*nkpts*16*comp)))
        blksize = min(blksize, ngrids, 16384)
        buf = np.empty(nkpts*nij*blksize*comp, dtype=np.complex128)

        for p0, p1 in self.prange(0, ngrids, blksize):
            dat = ft_kern(Gv[p0:p1], gxyz[p0:p1], Gvbase, q, kpts, shls_slice, out=buf)
            yield dat, p0, p1

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        '''Creates a temporary density fitting object for RSH-DF integrals.
        In this context, only LR or SR integrals for mol and auxmol are computed.
        '''
        key = '%.6f' % omega
        if key in self._rsh_df:
            rsh_df = self._rsh_df[key]
        else:
            rsh_df = self._rsh_df[key] = self.copy().reset()
            logger.info(self, 'Create RSH-DF object %s for omega=%s', rsh_df, omega)

        cell = self.cell
        auxcell = getattr(self, 'auxcell', None)

        cell_omega = cell.omega
        cell.omega = omega
        auxcell_omega = None
        if auxcell is not None:
            auxcell_omega = auxcell.omega
            auxcell.omega = omega

        assert rsh_df.cell.omega == omega
        if getattr(rsh_df, 'auxcell', None) is not None:
            assert rsh_df.auxcell.omega == omega

        try:
            yield rsh_df
        finally:
            cell.omega = cell_omega
            if auxcell_omega is not None:
                auxcell.omega = auxcell_omega


class AFTDF(lib.StreamObject, AFTDFMixin):
    '''Density expansion on plane waves
    '''

    _keys = {
        'cell', 'mesh', 'kpts', 'time_reversal_symmetry', 'blockdim',
    }

    # to mimic molecular DF object
    blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

    def __init__(self, cell, kpts=np.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory
        self.mesh = cell.mesh
        if cell.omega > 0:
            ke_cutoff = estimate_ke_cutoff_for_omega(cell, cell.omega)
            self.mesh = cell.cutoff_to_mesh(ke_cutoff)
        self.kpts = kpts
        self.time_reversal_symmetry = True

        # The following attributes are not input options.
        self._rsh_df = {}  # Range separated Coulomb DF objects

    def dump_flags(self, verbose=None):
        logger.info(self, '\n')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'mesh = %s (%d PWs)', self.mesh, np.prod(self.mesh))
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
                ke_cutoff = np.min(cell.ke_cutoff)
            ke_guess = estimate_ke_cutoff(cell, cell.precision)
            mesh_guess = cell.cutoff_to_mesh(ke_guess)
            if ke_cutoff < ke_guess * KE_SCALING:
                logger.warn(self, 'ke_cutoff/mesh (%g / %s) is not enough for AFTDF '
                            'to get integral accuracy %g.\nCoulomb integral error '
                            'is ~ %.2g Eh.\nRecommended ke_cutoff/mesh are %g / %s.',
                            ke_cutoff, self.mesh, cell.precision,
                            error_for_ke_cutoff(cell, ke_cutoff), ke_guess, mesh_guess)
        else:
            mesh_guess = np.copy(self.mesh)

        if cell.dimension < 3:
            err = np.exp(-0.436392335*min(self.mesh[cell.dimension:]) - 2.99944305)
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

    get_nuc = get_nuc
    get_pp = get_pp

    # Note: Special exxdiv by default should not be used for an arbitrary
    # input density matrix. When the df object was used with the molecular
    # post-HF code, get_jk was often called with an incomplete DM (e.g. the
    # core DM in CASCI). An SCF level exxdiv treatment is inadequate for
    # post-HF methods.
    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                     omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)
        if is_single_kpt:
            return aft_jk.get_jk(self, dm, hermi, kpts[0], kpts_band, with_j,
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
        mf = mf.copy()
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
                               'definite. Current API only supports positive '
                               'definite ERIs.')

        if blksize is None:
            blksize = self.blockdim
        # coulG of 1D and 2D has negative elements.
        coulG = self.weighted_coulG()
        Lpq = None
        for pqkR, pqkI, p0, p1 in self.pw_loop(aosym='s2', blksize=blksize):
            vG = np.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
            Lpq = lib.transpose(pqkR, out=Lpq)
            yield Lpq
            Lpq = lib.transpose(pqkI, out=Lpq)
            yield Lpq

    def get_naoaux(self):
        mesh = np.asarray(self.mesh)
        ngrids = np.prod(mesh)
        return ngrids * 2

def _sub_df_jk_(dfobj, dm, hermi=1, kpts=None, kpts_band=None,
                with_j=True, with_k=True, omega=None, exxdiv=None):
    with dfobj.range_coulomb(omega) as rsh_df:
        return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                             omega=None, exxdiv=exxdiv)
