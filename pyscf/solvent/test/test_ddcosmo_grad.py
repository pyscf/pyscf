#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import unittest
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import cc
from pyscf import tdscf
from pyscf import dft
from pyscf import df
from pyscf import solvent
from pyscf.scf import cphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.solvent import ddcosmo
from pyscf.solvent.grad import ddcosmo_grad
from pyscf.solvent import _ddcosmo_tdscf_grad
from pyscf.symm import sph

def tda_grad(td, z):
    '''ddcosmo TDA gradients'''
    mol = td.mol
    mf = td._scf
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    nocc = (mo_occ>0).sum()
    nvir = nmo - nocc
    z = z[0].reshape(nocc,nvir).T * numpy.sqrt(2)
    orbv = mo_coeff[:,nocc:]
    orbo = mo_coeff[:,:nocc]

    r_vdw = ddcosmo.get_atomic_radii(td.with_solvent)
    fi = ddcosmo.make_fi(td.with_solvent, r_vdw)
    ui = 1 - fi
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(td.with_solvent.lebedev_order)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, td.with_solvent.lmax, True))
    grids = td.with_solvent.grids
    cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, td.with_solvent.lmax)
    L = ddcosmo.make_L(td.with_solvent, r_vdw, ylm_1sph, fi)

    def fvind(x):
        v_mo  = numpy.einsum('iabj,xai->xbj', g[:nocc,nocc:,nocc:,:nocc], x)
        v_mo += numpy.einsum('aibj,xai->xbj', g[nocc:,:nocc,nocc:,:nocc], x)
        return v_mo

    h1 = rhf_grad.get_hcore(mol)
    s1 = rhf_grad.get_ovlp(mol)

    eri1 = -mol.intor('int2e_ip1', aosym='s1', comp=3)
    eri1 = eri1.reshape(3,nao,nao,nao,nao)
    eri0 = ao2mo.kernel(mol, mo_coeff)
    eri0 = ao2mo.restore(1, eri0, nmo).reshape(nmo,nmo,nmo,nmo)
    g = eri0 * 2 - eri0.transpose(0,3,2,1)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[nocc:]

    dielectric = td.with_solvent.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1
    pcm_nuc = .5 * f_epsilon * nuc_part1(td.with_solvent, r_vdw, ui, ylm_1sph, cached_pol, L)
    B0      = .5 * f_epsilon * make_B(td.with_solvent, r_vdw, ui, ylm_1sph, cached_pol, L)
    B0 = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', B0, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
    g += B0 * 2
    B1      = .5 * f_epsilon * make_B1(td.with_solvent, r_vdw, ui, ylm_1sph, cached_pol, L)

    offsetdic = mol.offset_nr_by_atom()
    de = numpy.zeros((mol.natm,3))
    for ia in range(mol.natm):
        shl0, shl1, p0, p1 = offsetdic[ia]

        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = -mol.atom_charge(ia) * mol.intor('int1e_iprinv', comp=3)
        h1ao[:,p0:p1] += h1[:,p0:p1]
        h1ao = h1ao + h1ao.transpose(0,2,1)
        h1ao += pcm_nuc[ia]
        h1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff, h1ao, mo_coeff)
        s1mo = numpy.einsum('pi,xpq,qj->xij', mo_coeff[p0:p1], s1[:,p0:p1], mo_coeff)
        s1mo = s1mo + s1mo.transpose(0,2,1)

        f1 = h1mo - numpy.einsum('xpq,pq->xpq', s1mo, zeta)
        f1-= numpy.einsum('klpq,xlk->xpq', g[:nocc,:nocc], s1mo[:,:nocc,:nocc])

        eri1a = eri1.copy()
        eri1a[:,:p0] = 0
        eri1a[:,p1:] = 0
        eri1a = eri1a + eri1a.transpose(0,2,1,3,4)
        eri1a = eri1a + eri1a.transpose(0,3,4,1,2)
        g1 = lib.einsum('xpqrs,pi,qj,rk,sl->xijkl', eri1a, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
        tmp1 = lib.einsum('xpqrs,pi,qj,rk,sl->xijkl', B1[ia], mo_coeff, mo_coeff, mo_coeff, mo_coeff)
        g1 = g1 * 2 - g1.transpose(0,1,4,3,2)
        g1 += tmp1 * 2
        f1 += numpy.einsum('xkkpq->xpq', g1[:,:nocc,:nocc])
        f1ai = f1[:,nocc:,:nocc].copy()

        c1 = s1mo * -.5
        c1vo = cphf.solve(fvind, mo_energy, mo_occ, f1ai, max_cycle=50)[0]
        c1[:,nocc:,:nocc] = c1vo
        c1[:,:nocc,nocc:] = -(s1mo[:,nocc:,:nocc]+c1vo).transpose(0,2,1)
        f1 += numpy.einsum('kapq,xak->xpq', g[:nocc,nocc:], c1vo)
        f1 += numpy.einsum('akpq,xak->xpq', g[nocc:,:nocc], c1vo)

        e1  = numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)
        e1 += numpy.einsum('xab,ai,bi->x', f1[:,nocc:,nocc:], z, z)
        e1 -= numpy.einsum('xij,ai,aj->x', f1[:,:nocc,:nocc], z, z)

        g1  = numpy.einsum('pjkl,xpi->xijkl', g, c1)
        g1 += numpy.einsum('ipkl,xpj->xijkl', g, c1)
        g1 += numpy.einsum('ijpl,xpk->xijkl', g, c1)
        g1 += numpy.einsum('ijkp,xpl->xijkl', g, c1)
        e1 += numpy.einsum('xaijb,ai,bj->x', g1[:,nocc:,:nocc,:nocc,nocc:], z, z)

        de[ia] = e1

    return de

def nuc_part(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''0th order'''
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2
    nao = mol.nao
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids

    extern_point_idx = ui > 0
    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    v_phi = numpy.zeros((natm, ngrid_1sph))
    for ia in range(natm):
# Note (-) sign is not applied to atom_charges, because (-) is explicitly
# included in rhs and L matrix
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
        v_phi[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi.ravel())
    Xvec = Xvec.reshape(natm,nlm)

    i1 = 0
    scaled_weights = numpy.empty((grids.weights.size))
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        eta_nj = 0
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            eta_nj += fac * numpy.einsum('mn,m->n', fak_pol[l], Xvec[ia,p0:p1])
        scaled_weights[i0:i1] = eta_nj
    scaled_weights *= grids.weights

    ao = mol.eval_gto('GTOval', grids.coords)
    vmat = -lib.einsum('g,gi,gj->ij', scaled_weights, ao, ao)

# Contribution of nuclear charges to the total density
# The factor numpy.sqrt(4*numpy.pi) is due to the product of 4*pi * Y_0^0
    psi = numpy.zeros((natm, nlm))
    for ia in range(natm):
        psi[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)

    # <Psi, L^{-1}g> -> Psi = SL the adjoint equation to LX = g
    L_S = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi.ravel())
    L_S = L_S.reshape(natm,nlm)
    xi_jn = numpy.einsum('n,jn,xn,jx->jn', weights_1sph, ui, ylm_1sph, L_S)
    cav_coords = cav_coords[extern_point_idx]
    xi_jn = xi_jn[extern_point_idx]

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))

    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    fakemol = gto.fakemol_for_charges(cav_coords)
    v_nj = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s1', cintopt=cintopt)
    vmat += numpy.einsum('ijn,n->ij', v_nj, xi_jn)
    return vmat

def nuc_part1(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''1st order'''
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2
    nao = mol.nao
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids
    aoslices = mol.aoslice_by_atom()

    extern_point_idx = ui > 0
    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    vmat1 = numpy.zeros((natm,3,nao,nao))

    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    v_phi = numpy.zeros((natm, ngrid_1sph))
    for ia in range(natm):
# Note (-) sign is not applied to atom_charges, because (-) is explicitly
# included in rhs and L matrix
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
        v_phi[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi0 = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0.ravel())
    Xvec0 = Xvec0.reshape(natm,nlm)

    ngrid_1sph = weights_1sph.size
    v_phi0 = numpy.empty((natm,ngrid_1sph))
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords
        v_phi0[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
    phi1 = -numpy.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, v_phi0)

    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        for ja in range(natm):
            rs = atom_coords[ja] - cav_coords
            d_rs = lib.norm(rs, axis=1)
            v_phi = atom_charges[ja] * numpy.einsum('px,p->px', rs, 1./d_rs**3)
            tmp = numpy.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
            phi1[ja,:,ia] += tmp  # response of the other atoms
            phi1[ia,:,ia] -= tmp  # response of cavity grids

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)

    phi1 -= lib.einsum('aziljm,jm->azil', L1, Xvec0)
    Xvec1 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi1.reshape(-1,natm*nlm).T)
    Xvec1 = Xvec1.T.reshape(natm,3,natm,nlm)

    i1 = 0
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        i0, i1 = i1, i1 + weight.size
        ao = mol.eval_gto('GTOval_sph_deriv1', coords)
        aow = numpy.einsum('gi,g->gi', ao[0], weight)
        aopair1 = lib.einsum('xgi,gj->xgij', ao[1:], aow)
        aow = numpy.einsum('gi,zxg->zxgi', ao[0], weight1)
        aopair0 = lib.einsum('zxgi,gj->zxgij', aow, ao[0])

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        vmat1 -= numpy.einsum('m,mn,zxnij->zxij', Xvec0[ia], fac_pol, aopair0)
        vtmp = numpy.einsum('m,mn,xnij->xij', Xvec0[ia],fac_pol, aopair1)
        vmat1[ia,:] -= vtmp
        vmat1[ia,:] -= vtmp.transpose(0,2,1)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            vmat1[ja,:,p0:p1,:] += vtmp[:,p0:p1]
            vmat1[ja,:,:,p0:p1] += vtmp[:,p0:p1].transpose(0,2,1)

        scaled_weights = lib.einsum('azm,mn->azn', Xvec1[:,:,ia], fac_pol)
        scaled_weights *= weight
        aow = numpy.einsum('gi,azg->azgi', ao[0], scaled_weights)
        vmat1 -= numpy.einsum('gi,azgj->azij', ao[0], aow)

    psi0 = numpy.zeros((natm, nlm))
    for ia in range(natm):
        psi0[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)

    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0.ravel())
    LS0 = LS0.reshape(natm,nlm)

    LS1 = numpy.einsum('il,aziljm->azjm', LS0, L1)
    LS1 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, LS1.reshape(-1,natm*nlm).T)
    LS1 = LS1.T.reshape(natm,3,natm,nlm)

    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        wtmp = lib.einsum('l,n,ln->ln', LS0[ia], weights_1sph, ylm_1sph)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
        vmat1 -= numpy.einsum('azl,n,ln,n,pqn->azpq', LS1[:,:,ia], weights_1sph, ylm_1sph, ui[ia], v_nj)
        vmat1 += lib.einsum('ln,azn,ijn->azij', wtmp, ui1[:,:,ia], v_nj)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3, aosym='s1')
        vtmp = lib.einsum('ln,n,xijn->xij', wtmp, ui[ia], v_e1_nj)
        vmat1[ia] += vtmp
        vmat1[ia] += vtmp.transpose(0,2,1)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            vmat1[ja,:,p0:p1,:] -= vtmp[:,p0:p1]
            vmat1[ja,:,:,p0:p1] -= vtmp[:,p0:p1].transpose(0,2,1)

    return vmat1

def make_B(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''0th order'''
    mol = pcmobj.mol
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    mol = pcmobj.mol
    natm = mol.natm
    nao  = mol.nao
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids

    extern_point_idx = ui > 0
    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    cav_coords = cav_coords[extern_point_idx]
    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    fakemol = gto.fakemol_for_charges(cav_coords)
    v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
    nao_pair = v_nj.shape[0]
    v_phi = numpy.zeros((natm, ngrid_1sph, nao, nao))
    v_phi[extern_point_idx] += v_nj.transpose(2,0,1)

    phi = numpy.einsum('n,xn,jn,jnpq->jxpq', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi.reshape(natm*nlm,-1))
    Xvec = Xvec.reshape(natm,nlm,nao,nao)

    ao = mol.eval_gto('GTOval', grids.coords)
    aow = numpy.einsum('gi,g->gi', ao, grids.weights)
    aopair = numpy.einsum('gi,gj->gij', ao, aow)

    psi = numpy.zeros((natm, nlm, nao, nao))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            psi[ia,p0:p1] = -fac * numpy.einsum('mn,nij->mij', fak_pol[l], aopair[i0:i1])

    B = lib.einsum('nlpq,nlrs->pqrs', psi, Xvec)
    B = B + B.transpose(2,3,0,1)
    return B

def make_B1(pcmobj, r_vdw, ui, ylm_1sph, cached_pol, L):
    '''1st order'''
    mol = pcmobj.mol
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]
    mol = pcmobj.mol
    natm = mol.natm
    nao  = mol.nao
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    grids = pcmobj.grids

    extern_point_idx = ui > 0
    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    cav_coords = cav_coords[extern_point_idx]
    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                         mol._env, int3c2e)
    fakemol = gto.fakemol_for_charges(cav_coords)
    v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
    nao_pair = v_nj.shape[0]
    v_phi = numpy.zeros((natm, ngrid_1sph, nao, nao))
    v_phi[extern_point_idx] += v_nj.transpose(2,0,1)
    phi0 = numpy.einsum('n,xn,jn,jnpq->jxpq', weights_1sph, ylm_1sph, ui, v_phi)

    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0.reshape(natm*nlm,-1))
    Xvec0 = Xvec0.reshape(natm,nlm,nao,nao)

    ao = mol.eval_gto('GTOval', grids.coords)
    aow = numpy.einsum('gi,g->gi', ao, grids.weights)
    aopair = numpy.einsum('gi,gj->gij', ao, aow)

    psi0 = numpy.zeros((natm, nlm, nao, nao))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            psi0[ia,p0:p1] = -fac * numpy.einsum('mn,nij->mij', fak_pol[l], aopair[i0:i1])

    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    phi1 = numpy.zeros(ui1.shape[:3] + (nlm,nao,nao))
    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1')
        phi1[:,:,ia] += lib.einsum('n,ln,azn,ijn->azlij', weights_1sph, ylm_1sph, ui1[:,:,ia], v_nj)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3, aosym='s1')
        v_e2_nj = v_e1_nj + v_e1_nj.transpose(0,2,1,3)
        phi1[ia,:,ia] += lib.einsum('n,ln,n,xijn->xlij', weights_1sph, ylm_1sph, ui[ia], v_e2_nj)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            v = numpy.einsum('n,ln,n,xijn->xlij', weights_1sph, ylm_1sph, ui[ia], v_e1_nj[:,p0:p1])
            phi1[ja,:,ia,:,p0:p1,:] -= v
            phi1[ja,:,ia,:,:,p0:p1] -= v.transpose(0,1,3,2)

    psi1 = numpy.zeros((natm,3,natm,nlm,nao,nao))
    i1 = 0
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        i0, i1 = i1, i1 + weight.size
        ao = mol.eval_gto('GTOval_sph_deriv1', coords)
        aow = numpy.einsum('gi,g->gi', ao[0], weight)
        aopair1 = lib.einsum('xgi,gj->xgij', ao[1:], aow)
        aow = numpy.einsum('gi,zxg->zxgi', ao[0], weight1)
        aopair0 = lib.einsum('zxgi,gj->zxgij', aow, ao[0])

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            psi1[: ,:,ia,p0:p1] -= fac*numpy.einsum('mn,zxnij->zxmij', fak_pol[l], aopair0)
            vtmp = fac*numpy.einsum('mn,xnij->xmij', fak_pol[l], aopair1)
            psi1[ia,:,ia,p0:p1] -= vtmp
            psi1[ia,:,ia,p0:p1] -= vtmp.transpose(0,1,3,2)

            for ja in range(natm):
                shl0, shl1, q0, q1 = aoslices[ja]
                psi1[ja,:,ia,p0:p1,q0:q1,:] += vtmp[:,:,q0:q1]
                psi1[ja,:,ia,p0:p1,:,q0:q1] += vtmp[:,:,q0:q1].transpose(0,1,3,2)

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)

    Xvec1 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi1.transpose(2,3,0,1,4,5).reshape(natm*nlm,-1))
    Xvec1 = Xvec1.reshape(natm,nlm,natm,3,nao,nao).transpose(2,3,0,1,4,5)
    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0.reshape(natm*nlm,-1))
    LS0 = LS0.reshape(natm,nlm,nao,nao)

    B = lib.einsum('ixnlpq,nlrs->ixpqrs', psi1, Xvec0)
    B+= lib.einsum('nlpq,ixnlrs->ixpqrs', psi0, Xvec1)
    B-= lib.einsum('ilpq,aziljm,jmrs->azpqrs', LS0, L1, Xvec0)
    B = B + B.transpose(0,1,4,5,2,3)
    return B

def B1_dot_x(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, L):
    mol = pcmobj.mol
    mol = pcmobj.mol
    natm = mol.natm
    nao  = mol.nao
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    dms = numpy.asarray(dm)
    is_single_dm = dms.ndim == 2
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    aoslices = mol.aoslice_by_atom()
    grids = pcmobj.grids
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]

    extern_point_idx = ui > 0
    fi0 = ddcosmo.make_fi(pcmobj, r_vdw)
    fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    Bx = numpy.zeros((natm,3,nao,nao))

    ao = mol.eval_gto('GTOval', grids.coords)
    aow = numpy.einsum('gi,g->gi', ao, grids.weights)
    aopair = numpy.einsum('gi,gj->gij', ao, aow)
    den = numpy.einsum('gij,ij->g', aopair, dm)
    psi0 = numpy.zeros((natm, nlm))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        i0, i1 = i1, i1 + fac_pol.shape[1]
        psi0[ia] = -numpy.einsum('mn,n->m', fac_pol, den[i0:i1])
    LS0 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi0.ravel())
    LS0 = LS0.reshape(natm,nlm)

    phi0 = numpy.zeros((natm,nlm))
    phi1 = numpy.zeros((natm,3,natm,nlm))
    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    cintopt_ip1 = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)

        v_phi = numpy.einsum('pqg,pq->g', v_nj, dm)
        phi0[ia] = numpy.einsum('n,ln,n,n->l', weights_1sph, ylm_1sph, ui[ia], v_phi)
        phi1[:,:,ia] += lib.einsum('n,ln,azn,n->azl', weights_1sph, ylm_1sph, ui1[:,:,ia], v_phi)
        Bx += lib.einsum('l,n,ln,azn,ijn->azij', LS0[ia], weights_1sph, ylm_1sph, ui1[:,:,ia], v_nj)

        wtmp = lib.einsum('n,ln,n->ln', weights_1sph, ylm_1sph, ui[ia])

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3,
                                   aosym='s1', cintopt=cintopt_ip1)
        vtmp = lib.einsum('l,ln,xijn->xij', LS0[ia], wtmp, v_e1_nj)
        Bx[ia] += vtmp
        Bx[ia] += vtmp.transpose(0,2,1)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            Bx[ja,:,p0:p1,:] -= vtmp[:,p0:p1]
            Bx[ja,:,:,p0:p1] -= vtmp[:,p0:p1].transpose(0,2,1)
            tmp  = numpy.einsum('xijn,ij->xn', v_e1_nj[:,p0:p1], dm[p0:p1])
            tmp += numpy.einsum('xijn,ji->xn', v_e1_nj[:,p0:p1], dm[:,p0:p1])
            phitmp = numpy.einsum('ln,xn->xl', wtmp, tmp)
            phi1[ja,:,ia] -= phitmp
            phi1[ia,:,ia] += phitmp

    Xvec0 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi0.ravel())
    Xvec0 = Xvec0.reshape(natm,nlm)

    L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi0)

    phi1 -= lib.einsum('aziljm,jm->azil', L1, Xvec0)
    Xvec1 = numpy.linalg.solve(L.reshape(natm*nlm,-1), phi1.reshape(-1,natm*nlm).T)
    Xvec1 = Xvec1.T.reshape(natm,3,natm,nlm)

    psi1 = numpy.zeros((natm,3,natm,nlm))
    i1 = 0
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        i0, i1 = i1, i1 + weight.size
        ao = mol.eval_gto('GTOval_sph_deriv1', coords)
        aow = numpy.einsum('gi,g->gi', ao[0], weight)
        aopair1 = lib.einsum('xgi,gj->xgij', ao[1:], aow)
        aow = numpy.einsum('gi,zxg->zxgi', ao[0], weight1)
        aopair0 = lib.einsum('zxgi,gj->zxgij', aow, ao[0])
        den0 = numpy.einsum('zxgij,ij->zxg', aopair0, dm)
        den1 = numpy.empty((natm,3,weight.size))
        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            den1[ja] = numpy.einsum('xgij,ij->xg', aopair1[:,:,p0:p1], dm[p0:p1,:])
            den1[ja]+= numpy.einsum('xgij,ji->xg', aopair1[:,:,p0:p1], dm[:,p0:p1])

        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        scaled_weights = lib.einsum('azm,mn->azn', Xvec1[:,:,ia], fac_pol)
        scaled_weights *= weight
        aow = numpy.einsum('gi,azg->azgi', ao[0], scaled_weights)
        Bx -= numpy.einsum('gi,azgj->azij', ao[0], aow)

        tmp = numpy.einsum('mn,zxn->zxm', fac_pol, den1)
        psi1[: ,:,ia] -= numpy.einsum('mn,zxn->zxm', fac_pol, den0)
        psi1[ia,:,ia] -= tmp.sum(axis=0)
        for ja in range(natm):
            psi1[ja,:,ia] += tmp[ja]

        eta_nj = lib.einsum('mn,m->n', fac_pol, Xvec0[ia])
        Bx -= lib.einsum('n,zxnpq->zxpq', eta_nj, aopair0)
        vtmp = lib.einsum('n,xnpq->xpq', eta_nj, aopair1)
        Bx[ia] -= vtmp
        Bx[ia] -= vtmp.transpose(0,2,1)
        for ja in range(natm):
            shl0, shl1, q0, q1 = aoslices[ja]
            Bx[ja,:,q0:q1,:] += vtmp[:,q0:q1]
            Bx[ja,:,:,q0:q1] += vtmp[:,q0:q1].transpose(0,2,1)

    psi1 -= numpy.einsum('il,aziljm->azjm', LS0, L1)
    LS1 = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi1.reshape(-1,natm*nlm).T)
    LS1 = LS1.T.reshape(natm,3,natm,nlm)

    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))
    cav_coords = cav_coords[extern_point_idx]
    fakemol = gto.fakemol_for_charges(cav_coords)
    v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
    v_phi = numpy.zeros((natm, ngrid_1sph, nao, nao))
    v_phi[extern_point_idx] += v_nj.transpose(2,0,1)
    Bx += lib.einsum('azjx,n,xn,jn,jnpq->azpq', LS1, weights_1sph, ylm_1sph, ui, v_phi)

    return Bx

def setUpModule():
    global dx, mol0, mol1, mol2, nao, dm
    dx = 0.0001
    mol0 = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1', unit='B')
    mol1 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%(-dx), unit='B')
    mol2 = gto.M(atom='H 0 0 %g; H 0 1 1.2; H 1. .1 0; H .5 .5 1'%dx, unit='B')
    dx = dx * 2
    nao = mol0.nao_nr()
    numpy.random.seed(1)
    dm = numpy.random.random((nao,nao))
    dm = dm + dm.T

def tearDownModule():
    global dx, mol0, mol1, mol2, nao, dm
    del dx, mol0, mol1, mol2, nao, dm

class KnownValues(unittest.TestCase):
    def test_e_psi1(self):
        def get_e_psi1(pcmobj):
            pcmobj.grids.build()
            mol = pcmobj.mol
            natm = mol.natm
            lmax = pcmobj.lmax

            r_vdw = ddcosmo.get_atomic_radii(pcmobj)
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

            fi = ddcosmo.make_fi(pcmobj, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            nexposed = numpy.count_nonzero(ui==1)
            nbury = numpy.count_nonzero(ui==0)
            on_shell = numpy.count_nonzero(ui>0) - nexposed

            nlm = (lmax+1)**2
            Lmat = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
            Lmat = Lmat.reshape(natm*nlm,-1)

            cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

            phi = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui, ylm_1sph)
            L_X = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
            psi, vmat, L_S = \
                    ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph,
                                          cached_pol, L_X, Lmat)
            psi1 = ddcosmo_grad.make_e_psi1(pcmobj, dm, r_vdw, ui, ylm_1sph,
                                            cached_pol, L_X, Lmat)
            return L_X, psi, psi1

        pcmobj = ddcosmo.DDCOSMO(mol0)
        L_X, psi0, psi1 = get_e_psi1(pcmobj)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        L_X1, psi = get_e_psi1(pcmobj)[:2]
        e1 = numpy.einsum('jx,jx', psi, L_X)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        L_X2, psi = get_e_psi1(pcmobj)[:2]
        e2 = numpy.einsum('jx,jx', psi, L_X)
        self.assertAlmostEqual(abs((e2-e1)/dx - psi1[0,2]).max(), 0, 7)

    def test_phi(self):
        def get_phi1(pcmojb):
            pcmobj.grids.build()
            mol = pcmobj.mol
            natm = mol.natm
            lmax = pcmobj.lmax

            r_vdw = pcmobj.get_atomic_radii()
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

            fi = ddcosmo.make_fi(pcmobj, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            nexposed = numpy.count_nonzero(ui==1)
            nbury = numpy.count_nonzero(ui==0)
            on_shell = numpy.count_nonzero(ui>0) - nexposed

            nlm = (lmax+1)**2
            Lmat = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
            Lmat = Lmat.reshape(natm*nlm,-1)

            cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

            phi = ddcosmo.make_phi(pcmobj, dm, r_vdw, ui, ylm_1sph)
            L_X = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
            psi, vmat, L_S = \
                    ddcosmo.make_psi_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph,
                                          cached_pol, L_X, Lmat)
            phi1 = ddcosmo_grad.make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph)
            phi1 = numpy.einsum('izjx,jx->iz', phi1, L_S)
            return L_S, phi, phi1

        pcmobj = ddcosmo.DDCOSMO(mol0)
        L_S, phi0, phi1 = get_phi1(pcmobj)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        L_S1, phi = get_phi1(pcmobj)[:2]
        e1 = numpy.einsum('jx,jx', phi, L_S)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        L_S2, phi = get_phi1(pcmobj)[:2]
        e2 = numpy.einsum('jx,jx', phi, L_S)
        self.assertAlmostEqual(abs((e2-e1)/dx - phi1[0,2]).max(), 0, 6)

    def test_fi(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        fi1 = ddcosmo_grad.make_fi1(pcmobj, pcmobj.get_atomic_radii())
        ui1 = -fi1
        fi = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui = 1 - fi
        ui1[:,:,ui<0] = 0

        pcmobj = ddcosmo.DDCOSMO(mol1)
        fi_1 = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui_1 = 1 - fi_1
        ui_1[ui_1<0] = 0

        pcmobj = ddcosmo.DDCOSMO(mol2)
        fi_2 = ddcosmo.make_fi(pcmobj, pcmobj.get_atomic_radii())
        ui_2 = 1 - fi_2
        ui_2[ui_2<0] = 0
        self.assertAlmostEqual(abs((fi_2-fi_1)/dx - fi1[0,2]).max(), 0, 5)
        self.assertAlmostEqual(abs((ui_2-ui_1)/dx - ui1[0,2]).max(), 0, 5)

    def test_L1(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        r_vdw = pcmobj.get_atomic_radii()
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcmobj.lmax, True))

        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L1 = ddcosmo_grad.make_L1(pcmobj, r_vdw, ylm_1sph, fi)

        pcmobj = ddcosmo.DDCOSMO(mol1)
        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L_1 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)

        pcmobj = ddcosmo.DDCOSMO(mol2)
        fi = ddcosmo.make_fi(pcmobj, r_vdw)
        L_2 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
        self.assertAlmostEqual(abs((L_2-L_1)/dx - L1[0,2]).max(), 0, 6)

    def test_e_cosmo_grad(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        de = ddcosmo_grad.kernel(pcmobj, dm)
        pcmobj = ddcosmo.DDCOSMO(mol1)
        e1 = pcmobj.energy(dm)
        pcmobj = ddcosmo.DDCOSMO(mol2)
        e2 = pcmobj.energy(dm)
        self.assertAlmostEqual(abs((e2-e1)/dx - de[0,2]).max(), 0, 6)

    def test_scf_grad(self):
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol0)).run()
        # solvent only
        de_cosmo = ddcosmo_grad.kernel(mf.with_solvent, mf.make_rdm1())
        self.assertAlmostEqual(lib.fp(de_cosmo), 0.000902640319, 5)
        # solvent + solute
        de = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(de), -0.191856565, 5)

        dm1 = mf.make_rdm1()

        mf1 = ddcosmo.ddcosmo_for_scf(scf.RHF(mol1)).run()
        e1 = mf1.e_tot
        e1_cosmo = mf1.with_solvent.energy(dm1)

        mf2 = ddcosmo.ddcosmo_for_scf(scf.RHF(mol2)).run()
        e2 = mf2.e_tot
        e2_cosmo = mf2.with_solvent.energy(dm1)
        self.assertAlmostEqual(abs((e2-e1)/dx - de[0,2]).max(), 0, 7)
        self.assertAlmostEqual(abs((e2_cosmo-e1_cosmo)/dx - de_cosmo[0,2]).max(), 0, 7)

        sc = mf.nuc_grad_method().as_scanner()
        e, g = sc('H 0 1 0; H 0 1 1.2; H 1. 0 0; H .5 .5 0')
        self.assertAlmostEqual(e, -0.83152362, 5)
        self.assertAlmostEqual(lib.fp(g), 0.068317954, 5)

        mol3 = gto.M(atom='H 0 1 0; H 0 1 1.2; H 1. 0 0; H .5 .5 0', unit='B')
        mf = ddcosmo.ddcosmo_for_scf(scf.RHF(mol3)).run()
        de = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(de), 0.0683179013, 5)

    def test_casci_grad(self):
        mf = scf.RHF(mol0).ddCOSMO().run()
        mc = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2))
        e, de = mc.nuc_grad_method().as_scanner()(mol0)
        self.assertAlmostEqual(e, -1.18433554, 5)
        self.assertAlmostEqual(lib.fp(de), -0.18543118, 5)

        mf = scf.RHF(mol1).run()
        mc1 = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2)).run()
        e1 = mc1.e_tot
        mf = scf.RHF(mol2).run()
        mc2 = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2)).run()
        e2 = mc2.e_tot
        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 3)

        ## FIXME: seems working?
        ## frozen dm in CASCI
        #mf = scf.RHF(mol0).ddCOSMO().run()
        #mc = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2), dm=mf.make_rdm1())
        #e, de = mc.nuc_grad_method().as_scanner()(mol0)
        #self.assertAlmostEqual(e, -1.1845042661517311, 7)
        #self.assertAlmostEqual(lib.fp(de), -0.18563349186388467, 5)

        #mf = scf.RHF(mol1).run()
        #mc1 = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2), dm=mf.make_rdm1()).run()
        #e1 = mc1.e_tot
        #mf = scf.RHF(mol2).run()
        #mc2 = solvent.ddCOSMO(mcscf.CASCI(mf, 2, 2), dm=mf.make_rdm1()).run()
        #e2 = mc2.e_tot
        #self.assertAlmostEqual((e2-e1)/dx, de[0,2], 4)

    def test_casscf_grad(self):
        mf = scf.RHF(mol0).ddCOSMO().run()
        mc = solvent.ddCOSMO(mcscf.CASSCF(mf, 2, 2)).set(conv_tol=1e-9)
        mc_g = mc.nuc_grad_method().as_scanner()
        e, de = mc_g(mol0)
        self.assertAlmostEqual(e, -1.19627418, 5)
        self.assertAlmostEqual(lib.fp(de), -0.1831184, 4)

        mf = scf.RHF(mol1).run()
        mc1 = solvent.ddCOSMO(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e1 = mc1.e_tot
        mf = scf.RHF(mol2).run()
        mc2 = solvent.ddCOSMO(mcscf.CASSCF(mf, 2, 2)).run(conv_tol=1e-9)
        e2 = mc2.e_tot
        # ddcosmo-CASSCF is not fully variational. Errors will be found large
        # in this test.
        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 2)

    def test_ccsd_grad(self):
        mf = scf.RHF(mol0).ddCOSMO().run()
        mycc = cc.CCSD(mf).ddCOSMO()
        e, de = mycc.nuc_grad_method().as_scanner()(mol0)
        self.assertAlmostEqual(e, -1.2060391657, 5)
        self.assertAlmostEqual(lib.fp(de), -0.1794318433, 5)

        mf = scf.RHF(mol1).run()
        mycc1 = solvent.ddCOSMO(cc.CCSD(mf)).run()
        e1 = mycc1.e_tot
        mf = scf.RHF(mol2).run()
        mycc2 = solvent.ddCOSMO(cc.CCSD(mf)).run()
        e2 = mycc2.e_tot
        self.assertAlmostEqual((e2-e1)/dx, de[0,2], 4)

    def test_tda_grad(self):
        mol0 = gto.M(atom='H 0 0 0    ; H .5 .5 .1', unit='B', basis='321g')
        mol1 = gto.M(atom='H 0 0 -.001; H .5 .5 .1', unit='B', basis='321g')
        mol2 = gto.M(atom='H 0 0 0.001; H .5 .5 .1', unit='B', basis='321g')
        mf = scf.RHF(mol0).ddCOSMO().run()
        td = solvent.ddCOSMO(tdscf.TDA(mf)).run(equilibrium_solvation=True)
        ref = tda_grad(td, td.xy[0]) + mf.nuc_grad_method().kernel()

        e, de = td.nuc_grad_method().as_scanner(state=1)(mol0)
        de = td.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(ref - de).max(), 0, 12)

        td1 = mol1.RHF().ddCOSMO().run().TDA().ddCOSMO().run(equilibrium_solvation=True)
        td2 = mol2.RHF().ddCOSMO().run().TDA().ddCOSMO().run(equilibrium_solvation=True)
        e1 = td1.e_tot[0]
        e2 = td2.e_tot[0]
        self.assertAlmostEqual((e2-e1)/0.002, de[0,2], 5)

    def test_solvent_nuc(self):
        def get_nuc(mol):
            pcm = ddcosmo.DDCOSMO(mol)
            pcm.lmax = 2
            pcm.eps = 0
            natm = mol.natm
            nao = mol.nao
            nlm = (pcm.lmax+1)**2
            r_vdw = ddcosmo.get_atomic_radii(pcm)
            fi = ddcosmo.make_fi(pcm, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            pcm.grids = grids = ddcosmo.Grids(mol).run(level=0)
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
            cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)
            L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
            return nuc_part(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)

        pcm = ddcosmo.DDCOSMO(mol0)
        pcm.lmax = 2
        pcm.eps = 0
        natm = mol0.natm
        nao = mol0.nao
        nlm = (pcm.lmax+1)**2
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        pcm.grids = grids = ddcosmo.Grids(mol0).run(level=0)
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)
        L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
        dvmat = nuc_part1(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)

        vmat1 = get_nuc(mol1)
        vmat2 = get_nuc(mol2)
        self.assertAlmostEqual(abs((vmat2-vmat1)/dx - dvmat[0,2]).max(), 0, 8)

        nao = mol0.nao
        numpy.random.seed(19)
        dm = numpy.random.random((nao,nao))
        vref = pcm._get_vind(dm)[1]
        vmat = 0.5 * get_nuc(mol0)
        vmat += pcm._B_dot_x(dm)
        self.assertAlmostEqual(abs(vmat-vref).max(), 0, 14)

        dm1 = numpy.random.random((2,nao,nao))
        de = _ddcosmo_tdscf_grad._grad_ne(pcm, dm1, r_vdw, ui, ylm_1sph, cached_pol, L)
        ref = numpy.einsum('azij,nij->naz', dvmat, dm1)
        self.assertAlmostEqual(abs(de - ref).max(), 0, 12)

    def test_B1(self):
        def getB(mol):
            pcm = ddcosmo.DDCOSMO(mol)
            pcm.lmax = 2
            pcm.eps = 0
            natm = mol.natm
            nao = mol.nao
            nlm = (pcm.lmax+1)**2
            r_vdw = ddcosmo.get_atomic_radii(pcm)
            fi = ddcosmo.make_fi(pcm, r_vdw)
            ui = 1 - fi
            ui[ui<0] = 0
            pcm.grids = grids = ddcosmo.Grids(mol).run(level=0)
            coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
            ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
            cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)
            L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
            return make_B(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)

        pcm = ddcosmo.DDCOSMO(mol0)
        pcm.lmax = 2
        pcm.eps = 0
        natm = mol0.natm
        nao = mol0.nao
        nlm = (pcm.lmax+1)**2
        r_vdw = ddcosmo.get_atomic_radii(pcm)
        fi = ddcosmo.make_fi(pcm, r_vdw)
        ui = 1 - fi
        ui[ui<0] = 0
        pcm.grids = grids = ddcosmo.Grids(mol0).run(level=0)
        coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcm.lebedev_order)
        ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcm.lmax, True))
        cached_pol = ddcosmo.cache_fake_multipoles(grids, r_vdw, pcm.lmax)
        L = ddcosmo.make_L(pcm, r_vdw, ylm_1sph, fi)
        dB = make_B1(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)

        B1 = getB(mol1)
        B2 = getB(mol2)
        self.assertAlmostEqual(abs((B2-B1)/dx - dB[0,2]).max(), 0, 8)

        nao = mol0.nao
        numpy.random.seed(1)
        dm1 = numpy.random.random((2,nao,nao))
        dm2 = numpy.random.random((2,nao,nao))
        dm = dm1[0]
        ref = numpy.einsum('azpqrs,npq->nazrs', dB, dm1)
        v = B1_dot_x(pcm, dm, r_vdw, ui, ylm_1sph, cached_pol, L)
        self.assertAlmostEqual(abs(v-ref[0]).max(), 0, 12)

        de = _ddcosmo_tdscf_grad._grad_ee(pcm, dm1, dm2, r_vdw, ui, ylm_1sph, cached_pol, L)
        ref = numpy.einsum('nazij,nij->naz', ref, dm2)
        self.assertAlmostEqual(abs(de - ref).max(), 0, 12)

        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        ref = ddcosmo_grad.kernel(pcm, dm)
        dielectric = pcm.eps
        if dielectric > 0:
            f_epsilon = (dielectric-1.)/dielectric
        else:
            f_epsilon = 1
        de = _ddcosmo_tdscf_grad._grad_nn(pcm, r_vdw, ui, ylm_1sph, cached_pol, L)
        de+= _ddcosmo_tdscf_grad._grad_ne(pcm, dm, r_vdw, ui, ylm_1sph, cached_pol, L)
        de+= .5*_ddcosmo_tdscf_grad._grad_ee(pcm, dm, dm, r_vdw, ui, ylm_1sph, cached_pol, L)
        de *= .5 * f_epsilon
        self.assertAlmostEqual(abs(de-ref).max(), 0, 12)

    def test_regularize_xt(self):
        pcmobj = ddcosmo.DDCOSMO(mol0)
        numpy.random.seed(2)
        t = numpy.random.rand(4)
        eta = 0.8
        L1 = ddcosmo_grad.regularize_xt1(t, eta)
        L_1 = ddcosmo.regularize_xt(t-1e-4, eta)
        L_2 = ddcosmo.regularize_xt(t+1e-4, eta)
        self.assertAlmostEqual(abs((L_2-L_1)/2e-4 - L1).max(), 0, 6)

if __name__ == "__main__":
    print("Full Tests for ddcosmo gradients")
    unittest.main()
