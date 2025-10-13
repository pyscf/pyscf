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

The strain tensor is generally not a symmetric tensor. Symmetrization

    [e1   e6/2 e5/2]
    [e6/2 e2   e4/2]
    [e5/2 e4/2 e3  ]

is applied to form 6 independent component.

    e1 = e_11
    e2 = e_22
    e3 = e_33
    e6 = e_12 + e_21
    e5 = e_13 + e_31
    e4 = e_32 + e_23

The 6 component strain is then used to define the symmetric stress tensor.

               1  d E
    sigma_i = --- ------  for i = 1 .. 6
               V  d e_i

The symmetric stress tensor represented in the 6 Voigt notation can be
transformed from the asymmetric stress tensor sigma_ij

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

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
from pyscf.pbc.dft.numint import KNumInt, _contract_rho
from pyscf.pbc.dft.krkspu import _set_U, _make_minao_lo, reference_mol
from pyscf.pbc.grad import krks as krks_grad
from pyscf.pbc.grad.rks_stress import (
    strain_tensor_dispalcement,
    _finite_diff_cells,
    _get_weight_strain_derivatives,
    _get_coulG_strain_derivatives,
    _eval_ao_strain_derivatives,
    _get_vpplocG_strain_derivatives,
    _get_pp_nonloc_strain_derivatives,
    ewald)

def get_ovlp(cell, kpts):
    '''Strain derivatives for overlap matrix
    '''
    disp = 1e-5
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    s = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=kpts1))
            s2 = np.asarray(cell2.pbc_intor('int1e_ovlp', kpts=kpts2))
            s.append((s1 - s2) / (2*disp))
    return s

def get_kin(cell, kpts):
    '''Strain derivatives for kinetic matrix
    '''
    disp = 1e-5
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    t = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            t1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=kpts1))
            t2 = np.asarray(cell2.pbc_intor('int1e_kin', kpts=kpts2))
            t.append((t1 - t2) / (2*disp))
    return t

def get_vxc(ks_grad, cell, dm_kpts, kpts, with_j=False, with_nuc=False):
    '''Strain derivatives for Coulomb and XC at gamma point

    Kwargs:
        with_j : Whether to include the electron-electron Coulomb interactions
        with_nuc : Whether to include the electron-nuclear Coulomb interactions
    '''
    mf = ks_grad.base
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1

    ni = mf._numint
    assert isinstance(ni, KNumInt)
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

    assert kpts.ndim == 2
    assert dm_kpts.ndim == 3
    nkpts, nao = dm_kpts.shape[:2]
    assert nkpts == len(kpts)

    coords = grids.coords
    ngrids = len(coords)
    mesh = grids.mesh
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)
    out = np.zeros((3,3))
    rho0 = np.zeros((nvar, ngrids))
    rho1 = np.zeros((3,3, nvar, ngrids))

    def partial_dot(bra, ket):
        '''conj(ig),ig->g'''
        # Adapt to the _contract_rho function
        return _contract_rho(bra.T, ket.T)

    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9
    p1 = 0
    for ao_ks, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, deriv+1, kpts=kpts):
        p0, p1 = p1, p1 + weight.size
        ao_ks_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv=deriv)
        coordsT = np.asarray(coords.T, order='C')
        for k, dm in enumerate(dm_kpts):
            ao = ao_ks[k].transpose(0,2,1)
            ao_strain = ao_ks_strain[k].transpose(0,1,2,4,3)
            if xctype == 'LDA':
                ao1 = ao_strain[:,:,0]
                # Adding grids' response
                ao1 += np.einsum('xig,yg->xyig', ao[1:4], coordsT)
                c0 = dm.T.dot(ao[0])
                rho0[0,p0:p1] += partial_dot(ao[0], c0).real
                rho1[:,:,0,p0:p1] += np.einsum('xyig,ig->xyg', ao1, c0.conj()).real
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
                    rho0[i,p0:p1] += partial_dot(ao[0], c0[i]).real
                rho1[:,:, : ,p0:p1] += np.einsum('xynig,ig->xyng', ao_strain, c0[0].conj()).real
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
                    rho0[i,p0:p1] += partial_dot(ao[0], c0[i]).real
                rho0[4,p0:p1] += partial_dot(ao[1], c0[1]).real
                rho0[4,p0:p1] += partial_dot(ao[2], c0[2]).real
                rho0[4,p0:p1] += partial_dot(ao[3], c0[3]).real
                rho1[:,:, :4,p0:p1] += np.einsum('xynig,ig->xyng', ao_strain, c0[0].conj()).real
                rho1[:,:,1:4,p0:p1] += np.einsum('xyig,nig->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
                rho1[:,:,4,p0:p1] += np.einsum('xynig,nig->xyg', ao_strain[:,:,1:4], c0[1:4].conj()).real

    if xctype == 'LDA':
        pass
    elif xctype == 'GGA':
        rho0[1:4] *= 2 # dm should be hermitian
    else: # MGGA
        rho0[1:4] *= 2 # dm should be hermitian
        rho0[4] *= .5 # factor 1/2 for tau
        rho1[:,:,4] *= .5

    rho0 *= 1./nkpts
    # *2 for rho1 because the derivatives were applied to the bra only
    rho1 *= 2./nkpts

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
            Ene += _get_pp_nonloc_strain_derivatives(cell, mesh, dm_kpts, kpts)
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
    assert isinstance(mf_grad, krks_grad.Gradients)
    mf = mf_grad.base
    with_df = mf.with_df
    assert isinstance(with_df, FFTDF)
    ni = mf._numint
    if ni.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError('Stress tensor for hybrid DFT')

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    sigma = ewald(cell)
    kpts = mf.kpts
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            t1 = cell1.pbc_intor('int1e_kin', kpts=kpts1)
            t2 = cell2.pbc_intor('int1e_kin', kpts=kpts2)
            t1 = sum(np.einsum('ij,ji->', x, d).real for x, d in zip(t1, dm0))
            t2 = sum(np.einsum('ij,ji->', x, d).real for x, d in zip(t2, dm0))
            sigma[x,y] += (t1 - t2) / (2*disp) / nkpts
            s1 = cell1.pbc_intor('int1e_ovlp', kpts=kpts1)
            s2 = cell2.pbc_intor('int1e_ovlp', kpts=kpts2)
            s1 = sum(np.einsum('ij,ji->', x, d).real for x, d in zip(s1, dme0))
            s2 = sum(np.einsum('ij,ji->', x, d).real for x, d in zip(s2, dme0))
            sigma[x,y] -= (s1 - s2) / (2*disp) / nkpts

    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_vxc(mf_grad, cell, dm0, kpts=kpts, with_j=True, with_nuc=True)
    t0 = log.timer_debug1('Vxc and Coulomb derivatives', *t0)

    if hasattr(mf, 'U_idx'):
        sigma += _hubbard_U_deriv1(mf, dm0, kpts)
        log.timer_debug1('DFT+U')

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma

def _get_first_order_local_orbitals(cell, minao_ref='MINAO', kpts=None):
    if isinstance(minao_ref, str):
        pcell = reference_mol(cell, minao_ref)
    else:
        pcell = minao_ref
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    nao = cell.nao
    naop = pcell.nao
    if is_zero(kpts):
        dtype = np.float64
    else:
        dtype = np.complex128
    C1_minao = np.empty((3, 3, nkpts, nao, naop), dtype=dtype)
    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            pcell1, pcell2 = _finite_diff_cells(pcell, x, y, disp)
            kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))
            kpts2 = scaled_kpts.dot(cell2.reciprocal_vectors(norm_to=1))
            C1 = np.asarray(_make_minao_lo(cell1, pcell1, kpts=kpts1), dtype=dtype)
            C2 = np.asarray(_make_minao_lo(cell2, pcell2, kpts=kpts2), dtype=dtype)
            C1_minao[x,y] = (C1 - C2) / (2*disp)
    return C1_minao

def _hubbard_U_deriv1(mf, dm=None, kpts=None):
    assert mf.alpha is None
    assert mf.C_ao_lo is None
    assert mf.minao_ref is not None
    if dm is None:
        dm = mf.make_rdm1()
    if kpts is None:
        kpts = mf.kpts
    nkpts = len(kpts)
    cell = mf.cell

    # Construct orthogonal minao local orbitals.
    pcell = reference_mol(cell, mf.minao_ref)
    C_ao_lo = _make_minao_lo(cell, pcell, kpts=kpts)
    U_idx, U_val = _set_U(cell, pcell, mf.U_idx, mf.U_val)[:2]
    U_idx_stack = np.hstack(U_idx)
    C0 = [C_k[:,U_idx_stack] for C_k in C_ao_lo]
    C1_ao_lo = _get_first_order_local_orbitals(cell, pcell, kpts)
    C1 = [C_k[:,:,:,U_idx_stack] for C_k in C1_ao_lo.transpose(2,0,1,3,4)]

    ovlp0 = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    ovlp1 = np.asarray(get_ovlp(cell, kpts))
    nao = cell.nao
    ovlp1 = ovlp1.reshape(3,3,nkpts,nao,nao).transpose(2,0,1,3,4)
    C_inv = [C_k.conj().T.dot(S_k) for C_k, S_k in zip(C0, ovlp0)]
    dm_deriv0 = [C_k.dot(dm_k).dot(C_k.conj().T) for C_k, dm_k in zip(C_inv, dm)]

    sigma = np.zeros((3, 3))
    weight = 1. / nkpts
    for k in range(nkpts):
        SC1 = lib.einsum('pq,xyqi->xypi', ovlp0[k], C1[k])
        SC1 += lib.einsum('xypq,qi->xypi', ovlp1[k], C0[k])
        dm_deriv1 = lib.einsum('pj,xyjq->xypq', C_inv[k].dot(dm[k]), SC1)
        i0 = i1 = 0
        for idx, val in zip(U_idx, U_val):
            i0, i1 = i1, i1 + len(idx)
            P0 = dm_deriv0[k][i0:i1,i0:i1]
            P1 = dm_deriv1[:,:,i0:i1,i0:i1]
            sigma += weight * (val * 0.5) * (
                np.einsum('xyii->xy', P1).real * 2 # *2 for P1+P1.T
                - np.einsum('xyij,ji->xy', P1, P0).real * 2)
    return sigma
