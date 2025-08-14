#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

r'''
The energy derivatives for the strain tensor e_ij is

                1  d E
    sigma_ij = --- ------
                V  d e_ij

The strain tesnor e_ij describes the transformation for real space coordinates
in the crystal

    \sum_j [\deta_ij + e_ij] R_j  [for j = x, y, z]

Due to numerical errors, the strain tensor may slightly break the symmetry
within the stress tensor. The 6 independent components of the stress tensor

    [e1   e6/2 e5/2]
    [e6/2 e2   e4/2]
    [e5/2 e4/2 e3  ]

is constructed by symmetrizing the strain tensor as follows:

    e1 = e_11
    e2 = e_22
    e3 = e_33
    e6 = e_12 + e_21
    e5 = e_13 + e_31
    e4 = e_32 + e_23

See K. Doll, Mol Phys (2010), 108, 223
'''

import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF, ft_ao
from pyscf.pbc.dft.numint import NumInt, _contract_rho
from pyscf.pbc.grad import rks as rks_grad

def strain_tensor_dispalcement(x, y, disp):
    E_strain = np.eye(3)
    E_strain[x,y] += disp
    return E_strain

def _finite_diff_cells(cell, x, y, disp=1e-4, precision=None):
    if precision is not None:
        cell = cell.copy()
        cell.precision = precision
    a = cell.lattice_vectors()
    r = cell.atom_coords()
    if not gto.mole.is_au(cell.unit):
        a *= lib.param.BOHR
        r *= lib.param.BOHR
    e_strain = strain_tensor_dispalcement(x, y, disp)
    cell1 = cell.set_geom_(r.dot(e_strain.T), inplace=False)
    cell1.a = a.dot(e_strain.T)

    e_strain = strain_tensor_dispalcement(x, y, -disp)
    cell2 = cell.set_geom_(r.dot(e_strain.T), inplace=False)
    cell2.a = a.dot(e_strain.T)

    if cell.space_group_symmetry:
        cell1.build(False, False)
        cell2.build(False, False)
    return cell1, cell2

def get_ovlp(cell):
    '''Strain derivatives for overlap matrix
    '''
    disp = 1e-5
    s = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            s1 = cell1.pbc_intor('int1e_ovlp')
            s2 = cell2.pbc_intor('int1e_ovlp')
            s.append((s1 - s2) / (2*disp))
    return s

def get_kin(cell):
    '''Strain derivatives for kinetic matrix
    '''
    disp = 1e-5
    t = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = cell1.pbc_intor('int1e_kin')
            t2 = cell2.pbc_intor('int1e_kin')
            t.append((t1 - t2) / (2*disp))
    return t

def _get_coulG_strain_derivatives(cell, Gv):
    '''derivatives of 4pi/G^2'''
    G2 = np.einsum('gx,gx->g', Gv, Gv)
    G2[0] = np.inf
    coulG_0 = 4 * np.pi / G2
    coulG_1 = np.einsum('gx,gy->xyg', Gv, Gv)
    coulG_1 *= coulG_0 * 2/G2
    return coulG_0, coulG_1

def _get_weight_strain_derivatives(cell, grids):
    ngrids = grids.size
    weight_0 = cell.vol / ngrids
    weight_1 = np.eye(3) * weight_0
    return weight_0, weight_1

def _eval_ao_strain_derivatives(cell, coords, kpts=None, deriv=0, out=None):
    '''
    Returns:
        ao_kpts: (nkpts, 3, 3, comp, ngrids, nao) ndarray
            AO values at each k-point
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    comp_3x3 = comp * 9
    if cell.cart:
        feval = 'GTOval_cart_deriv%d_strain_tensor' % deriv
    else:
        feval = 'GTOval_sph_deriv%d_strain_tensor' % deriv
    out = cell.pbc_eval_gto(feval, coords, comp_3x3, kpts, out=out)
    ngrids = len(coords)

    if isinstance(out, np.ndarray):
        out = [out.reshape(3,3,comp,ngrids,-1)]
    else:
        out = [x.reshape(3,3,comp,ngrids,-1) for x in out]
    return out

def get_vxc(ks_grad, cell, dm, with_j=False, with_nuc=False):
    '''Strain derivatives for Coulomb and XC at gamma point

    Kwargs:
        with_j : Whether to include the electron-electron Coulomb interactions
        with_nuc : Whether to include the electron-nuclear Coulomb interactions
    '''
    mf = ks_grad.base
    if dm is None: dm = mf.make_rdm1()
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1

    ni = mf._numint
    assert isinstance(ni, NumInt)
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    assert isinstance(grids, UniformGrids)

    xc_code = mf.xc
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
        nvar = 1
    elif xctype == 'GGA':
        deriv = 1
        nvar = 4
    elif xctype == 'MGGA':
        deriv = 1
        nvar = 5
    else:
        raise NotImplementedError

    assert dm.ndim == 2
    nao = dm.shape[-1]

    coords = grids.coords
    ngrids = len(coords)
    mesh = grids.mesh
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)
    out = np.zeros((3,3))
    rho0 = np.empty((nvar, ngrids))
    rho1 = np.empty((3,3, nvar, ngrids))

    def partial_dot(bra, ket):
        '''conj(ig),ig->g'''
        # Adapt to the _contract_rho function
        return _contract_rho(bra.T, ket.T)

    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9
    p1 = 0
    for ao, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, deriv+1):
        p0, p1 = p1, p1 + weight.size
        ao_strain = _eval_ao_strain_derivatives(cell, coords, deriv=deriv)[0]
        ao = ao.transpose(0,2,1)
        ao_strain = ao_strain.transpose(0,1,2,4,3)
        coordsT = np.asarray(coords.T, order='C')
        if xctype == 'LDA':
            ao1 = ao_strain[:,:,0]
            # Adding the response of the grids
            ao1 += np.einsum('xig,yg->xyig', ao[1:4], coordsT)
            c0 = dm.T.dot(ao[0])
            rho0[0,p0:p1] = partial_dot(ao[0], c0).real
            rho1[:,:,0,p0:p1] = np.einsum('xyig,ig->xyg', ao1, c0.conj()).real
        elif xctype == 'GGA':
            ao_strain[:,:,0] += np.einsum('xig,yg->xyig', ao[1:4], coordsT)
            ao_strain[:,:,1] += np.einsum('xig,yg->xyig', ao[4:7], coordsT)
            ao_strain[0,:,2] += np.einsum('ig,yg->yig', ao[XY], coordsT)
            ao_strain[1,:,2] += np.einsum('ig,yg->yig', ao[YY], coordsT)
            ao_strain[2,:,2] += np.einsum('ig,yg->yig', ao[ZY], coordsT)
            ao_strain[0,:,3] += np.einsum('ig,yg->yig', ao[XZ], coordsT)
            ao_strain[1,:,3] += np.einsum('ig,yg->yig', ao[YZ], coordsT)
            ao_strain[2,:,3] += np.einsum('ig,yg->yig', ao[ZZ], coordsT)
            c0 = lib.einsum('xig,ij->xjg', ao[:4], dm)
            for i in range(4):
                rho0[i,p0:p1] = partial_dot(ao[0], c0[i]).real
            rho1[:,:, : ,p0:p1]  = np.einsum('xynig,ig->xyng', ao_strain, c0[0].conj()).real
            rho1[:,:,1:4,p0:p1] += np.einsum('xyig,nig->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
        else: # MGGA
            ao_strain[:,:,0] += np.einsum('xig,yg->xyig', ao[1:4], coordsT)
            ao_strain[:,:,1] += np.einsum('xig,yg->xyig', ao[4:7], coordsT)
            ao_strain[0,:,2] += np.einsum('ig,yg->yig', ao[XY], coordsT)
            ao_strain[1,:,2] += np.einsum('ig,yg->yig', ao[YY], coordsT)
            ao_strain[2,:,2] += np.einsum('ig,yg->yig', ao[ZY], coordsT)
            ao_strain[0,:,3] += np.einsum('ig,yg->yig', ao[XZ], coordsT)
            ao_strain[1,:,3] += np.einsum('ig,yg->yig', ao[YZ], coordsT)
            ao_strain[2,:,3] += np.einsum('ig,yg->yig', ao[ZZ], coordsT)
            c0 = lib.einsum('xig,ij->xjg', ao[:4], dm)
            for i in range(4):
                rho0[i,p0:p1] = partial_dot(ao[0], c0[i]).real
            rho0[4,p0:p1]  = partial_dot(ao[1], c0[1]).real
            rho0[4,p0:p1] += partial_dot(ao[2], c0[2]).real
            rho0[4,p0:p1] += partial_dot(ao[3], c0[3]).real
            rho1[:,:, :4,p0:p1]  = np.einsum('xynig,ig->xyng', ao_strain, c0[0].conj()).real
            rho1[:,:,1:4,p0:p1] += np.einsum('xyig,nig->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
            rho1[:,:,4,p0:p1] = np.einsum('xynig,nig->xyg', ao_strain[:,:,1:4], c0[1:4].conj()).real

    if xctype == 'LDA':
        pass
    elif xctype == 'GGA':
        rho0[1:4] *= 2 # dm should be hermitian
    else: # MGGA
        rho0[1:4] *= 2 # dm should be hermitian
        rho0[4] *= .5 # factor 1/2 for tau
        rho1[:,:,4] *= .5
    # *2 for rho1 because the derivatives were applied to the bra only
    rho1 *= 2.

    exc, vxc = ni.eval_xc_eff(xc_code, rho0, 1, xctype=xctype, spin=0)[:2]
    out += np.einsum('xyng,ng->xy', rho1, vxc).real * weight_0
    out += np.einsum('g,g->', rho0[0], exc).real * weight_1

    Gv = cell.get_Gv(mesh)
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
    rhoG = pbctools.fft(rho0[0], mesh)
    if with_j:
        vR = pbctools.ifft(rhoG * coulG_0, mesh)
        EJ = np.einsum('xyg,g->xy', rho1[:,:,0], vR).real * weight_0 * 2
        EJ += np.einsum('g,g->', rho0[0], vR).real * weight_1
        EJ += np.einsum('g,xyg,g->xy', rhoG.conj(), coulG_1, rhoG).real * (weight_0/ngrids)
        out += .5 * EJ

    if with_nuc:
        if cell._pseudo:
            vpplocG_0, vpplocG_1 = _get_vpplocG_strain_derivatives(cell, mesh)
            vpplocR = pbctools.ifft(vpplocG_0, mesh).real
            Ene = np.einsum('xyg,g->xy', rho1[:,:,0], vpplocR).real
            Ene += np.einsum('g,xyg->xy', rhoG.conj(), vpplocG_1).real * (1./ngrids)
            Ene += _get_pp_nonloc_strain_derivatives(cell, mesh, dm)
        else:
            charge = -cell.atom_charges()
            # SI corresponds to Fourier components of the fractional atomic
            # positions within the cell. It does not respond to the strain
            # transformation
            SI = cell.get_SI(mesh=mesh)
            ZG = np.dot(charge, SI)
            vR = pbctools.ifft(ZG * coulG_0, mesh).real
            Ene = np.einsum('xyg,g->xy', rho1[:,:,0], vR).real
            Ene += np.einsum('g,xyg,g->xy', rhoG.conj(), coulG_1, ZG).real * (1./ngrids)
        out += Ene
    return out

def _get_vpplocG_strain_derivatives(cell, mesh):
    disp = 1e-5
    ngrids = np.prod(mesh)
    v1 = np.empty((3,3, ngrids), dtype=np.complex128)
    SI = cell.get_SI(mesh=mesh)
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            vpplocG1 = pseudo.get_vlocG(cell1, cell1.get_Gv(mesh))
            vpplocG2 = pseudo.get_vlocG(cell2, cell2.get_Gv(mesh))
            vpplocG1 = -np.einsum('ij,ij->j', SI, vpplocG1)
            vpplocG2 = -np.einsum('ij,ij->j', SI, vpplocG2)
            v1[x,y] = (vpplocG1 - vpplocG2) / (2*disp)
    vpplocG = pseudo.get_vlocG(cell, cell.get_Gv(mesh))
    v0 = -np.einsum('ij,ij->j', SI, vpplocG)
    return v0, v1

def _get_pp_nonloc_strain_derivatives(cell, mesh, dm_kpts, kpts=None):
    if kpts is None:
        assert dm_kpts.ndim == 2
        dm_kpts = dm_kpts[None,:,:]
        kpts = np.zeros((1, 3))
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    ngrids = np.prod(mesh)
    buf = np.empty((48,ngrids), dtype=np.complex128)
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    def eval_pp_nonloc(cell):
        vol = cell.vol
        b = cell.reciprocal_vectors(norm_to=1)
        Gv = cell.get_Gv(mesh)
        SI = cell.get_SI(mesh=mesh)
        # buf for SPG_lmi upto l=0..3 and nl=3
        vppnl = 0
        for k, dm in enumerate(dm_kpts):
            kpt = scaled_kpts[k].dot(b)
            Gk = Gv + kpt
            G_rad = lib.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/vol)**.5
            for ia in range(cell.natm):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        fakemol._bas[0,gto.ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)

                        p0, p1 = p1, p1+nl*(l*2+1)
                        # pYlm is real, SI[ia] is complex
                        pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                if p1 > 0:
                    SPG_lmi = buf[:p1]
                    SPG_lmi *= SI[ia].conj()
                    SPG_lm_aoGs = SPG_lmi.dot(aokG)
                    rho = SPG_lm_aoGs.dot(dm).dot(SPG_lm_aoGs.conj().T).real
                    p1 = 0
                    for l, proj in enumerate(pp[5:]):
                        rl, nl, hl = proj
                        if nl > 0:
                            nf = l * 2 + 1
                            p0, p1 = p1, p1+nl*nf
                            hl = np.asarray(hl)
                            rho_sub = rho[p0:p1,p0:p1].reshape(nl, nf, nl, nf)
                            vppnl += np.einsum('ij,jmim->', hl, rho_sub)
        return vppnl / (nkpts*vol)

    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = eval_pp_nonloc(cell1)
            e2 = eval_pp_nonloc(cell2)
            out[i,j] = (e1 - e2) / (2*disp)
    return out

def ewald(cell):
    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(i+1):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = cell1.ewald()
            e2 = cell2.ewald()
            out[j,i] = out[i,j] = (e1 - e2) / (2*disp)
    return out

def kernel(mf_grad):
    '''Compute the energy derivatives for strain tensor (e_ij)

                1  d E
    sigma_ij = --- ------
                V  d e_ij

    sigma is a asymmetric 3x3 matrix. The symmetric stress tensor in the 6 Voigt
    notation can be transformed from the asymmetric stress tensor

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

    See K. Doll, Mol Phys (2010), 108, 223
    '''
    assert isinstance(mf_grad, rks_grad.Gradients)
    mf = mf_grad.base
    assert is_zero(mf.kpt)
    with_df = mf.with_df
    assert isinstance(with_df, FFTDF)
    ni = mf._numint
    if ni.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError('Stress tensor for hybrid DFT')
    if hasattr(mf, 'U_idx'):
        raise NotImplementedError('Stress tensor for DFT+U')

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    sigma = ewald(cell)

    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = cell1.pbc_intor('int1e_kin')
            t2 = cell2.pbc_intor('int1e_kin')
            t1 = np.einsum('ij,ji->', t1, dm0)
            t2 = np.einsum('ij,ji->', t2, dm0)
            sigma[x,y] += (t1 - t2) / (2*disp)
            s1 = cell1.pbc_intor('int1e_ovlp')
            s2 = cell2.pbc_intor('int1e_ovlp')
            s1 = np.einsum('ij,ji->', s1, dme0)
            s2 = np.einsum('ij,ji->', s2, dme0)
            sigma[x,y] -= (s1 - s2) / (2*disp)
    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_vxc(mf_grad, cell, dm0, with_j=True, with_nuc=True)
    t0 = log.timer_debug1('Vxc and Coulomb derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma

# TODO: DFT+U
