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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

'''
PBC spin-restricted G0W0-CD QP eigenvalues with k-point sampling
This implementation has the same scaling (N^4) as GW-AC, more robust but slower.
GW-CD is particularly recommended for accurate core and high-energy states.

Method:
    See T. Zhu and G.K.-L. Chan, arxiv:2007.03148 (2020) for details
    Compute Sigma directly on real axis with density fitting
    through a contour deformation method
'''

from functools import reduce
import numpy
import numpy as np
import h5py
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, orbs=None,
           kptlist=None, nw=None, verbose=logger.NOTE):
    '''GW-corrected quasiparticle orbital energies

    Returns:
        A list :  converged, mo_energy, mo_coeff
    '''
    mf = gw._scf
    if gw.frozen is None:
        frozen = 0
    else:
        frozen = gw.frozen
    assert (frozen == 0)

    if orbs is None:
        orbs = range(gw.nmo)
    if kptlist is None:
        kptlist = range(gw.nkpts)
    nkpts = gw.nkpts
    nklist = len(kptlist)

    # v_xc
    dm = np.array(mf.make_rdm1())
    v_mf = np.array(mf.get_veff()) - np.array(mf.get_j(dm_kpts=dm))
    for k in range(nkpts):
        v_mf[k] = reduce(numpy.dot, (mo_coeff[k].T.conj(), v_mf[k], mo_coeff[k]))

    nocc = gw.nocc
    nmo = gw.nmo

    # v_hf from DFT/HF density
    if gw.fc:
        exxdiv = 'ewald'
    else:
        exxdiv = None
    rhf = scf.KRHF(gw.mol, gw.kpts, exxdiv=exxdiv)
    rhf.with_df = gw.with_df
    if getattr(gw.with_df, '_cderi', None) is None:
        raise RuntimeError('Found incompatible integral scheme %s.'
                           'KGWCD can be only used with GDF integrals' %
                           gw.with_df.__class__)
    if rhf.with_df._j_only:
        logger.debug(gw, 'Rebuild CDERI for exchange integrals')
        rhf.with_df.build(j_only=False)

    vk = rhf.get_veff(gw.mol,dm_kpts=dm) - rhf.get_j(gw.mol,dm_kpts=dm)
    for k in range(nkpts):
        vk[k] = reduce(numpy.dot, (mo_coeff[k].T.conj(), vk[k], mo_coeff[k]))

    # Grids for integration on imaginary axis
    freqs,wts = _get_scaled_legendre_roots(nw)

    logger.debug(gw, "Computing the imaginary part")
    Wmn, Del_00, Del_P0, qij, q_abs = get_WmnI_diag(gw, orbs, kptlist, freqs)

    conv = True
    mo_energy = np.zeros_like(np.array(mf.mo_energy))
    for k in range(nklist):
        kn = kptlist[k]
        for p in orbs:
            if p < nocc:
                delta = -2e-2
            else:
                delta = 2e-2
            if gw.linearized:
                # FIXME
                logger.warn(gw,'linearization with CD leads to wrong quasiparticle energy')
                raise NotImplementedError
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    if gw.fc:
                        sigmaR = get_sigma_diag(gw, omega, kn, p, Wmn[:,k,:,p-orbs[0],:],
                                                Del_00, Del_P0[k,p-orbs[0],:], freqs, wts, qij, q_abs).real
                    else:
                        sigmaR = get_sigma_diag(gw, omega, kn, p, Wmn[:,k,:,p-orbs[0],:],
                                                Del_00, Del_P0, freqs, wts, qij, q_abs).real
                    return omega - mf.mo_energy[kn][p] - (sigmaR.real + vk[kn,p,p].real - v_mf[kn,p,p].real)
                try:
                    e = newton(quasiparticle, mf.mo_energy[kn][p]+delta, tol=1e-6, maxiter=50)
                    logger.debug(gw, "Computing poles for QP (k: %s, orb: %s)"%(kn,p))
                    mo_energy[kn,p] = e
                except RuntimeError:
                    conv = False
    mo_coeff = mf.mo_coeff

    if gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy @ k%d =\n%s', k,mo_energy[k])
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff

def get_sigma_diag(gw, ep, kp, p, Wmn, Del_00, Del_P0, freqs, wts, qij, q_abs):
    '''
    Compute self-energy on real axis using contour deformation
    '''
    nocc = gw.nocc
    nkpts = gw.nkpts
    # This code does not support metals
    homo = -99.
    lumo = 99.
    for k in range(nkpts):
        if homo < gw._scf.mo_energy[k][nocc-1]:
            homo = gw._scf.mo_energy[k][nocc-1]
        if lumo > gw._scf.mo_energy[k][nocc]:
            lumo = gw._scf.mo_energy[k][nocc]
    ef = (homo+lumo)/2.

    nmo = gw.nmo
    sign = np.zeros((nkpts,nmo),dtype=np.int64)
    for k in range(nkpts):
        sign[k] = np.sign(ef-gw._scf.mo_energy[k])
    sigmaI = get_sigmaI_diag(gw, ep, kp, p, Wmn, Del_00, Del_P0, sign, freqs, wts)
    sigmaR = get_sigmaR_diag(gw, ep, kp, p, ef, freqs, qij, q_abs)
    return sigmaI + sigmaR

def get_rho_response(gw, omega, mo_energy, Lpq, kL, kidx):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    nkpts, naux, nmo, nmo = Lpq.shape
    nocc = gw.nocc
    kpts = gw.kpts
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux,naux),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = mo_energy[i,:nocc,None] - mo_energy[a,None,nocc:]
        eia = eia/(omega**2+eia*eia)
        Pia = einsum('Pia,ia->Pia',Lpq[i][:,:nocc,nocc:],eia)
        # Response from both spin-up and spin-down density
        Pi += 4./nkpts * einsum('Pia,Qia->PQ',Pia,Lpq[i][:,:nocc,nocc:].conj())
    return Pi

def get_WmnI_diag(gw, orbs, kptlist, freqs, max_memory=8000):
    '''
    Compute GW correlation self-energy (diagonal elements)
    in MO basis on imaginary axis
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    nklist = len(kptlist)
    nw = len(freqs)
    norbs = len(orbs)
    mydf = gw.with_df

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    Del_00, Del_P0, qij, q_abs = None, None, None, None
    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        q_pts = np.array([1e-3,0,0]).reshape(1,3)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = get_qij(gw, q_abs[0], mo_coeff)

    Wmn = np.zeros((nkpts,nklist,nmo,norbs,nw),dtype=np.complex128)
    if gw.fc:
        Del_P0 = np.zeros((nklist,norbs,nw),dtype=np.complex128)
        Del_00 = np.zeros(nw,dtype=np.complex128)
    for kL in range(nkpts):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, "Read Lpq (kL: %s / %s, ki: %s, kj: %s)"%(kL+1, nkpts, i, j))
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign \
                            in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support unequal naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nmo**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                    Lij.append(Lij_out.reshape(-1,nmo,nmo))
        Lij = np.asarray(Lij)
        naux = Lij.shape[1]

        if kL == 0:
            for w in range(nw):
                # body dielectric matrix eps_body
                Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                eps_body_inv = np.linalg.inv(np.eye(naux)-Pi)

                if gw.fc:
                    # head dielectric matrix eps_00
                    Pi_00 = get_rho_response_head(gw, freqs[w], mo_energy, qij)
                    eps_00 = 1. - 4. * np.pi/np.linalg.norm(q_abs[0])**2 * Pi_00

                    # wings dielectric matrix eps_P0
                    Pi_P0 = get_rho_response_wing(gw, freqs[w], mo_energy, Lij, qij)
                    eps_P0 = -np.sqrt(4.*np.pi) / np.linalg.norm(q_abs[0]) * Pi_P0

                    # inverse dielectric matrix
                    eps_inv_00 = 1./(eps_00 - np.dot(np.dot(eps_P0.conj(),eps_body_inv),eps_P0))
                    eps_inv_P0 = -eps_inv_00 * np.dot(eps_body_inv, eps_P0)

                    # head correction
                    Del_00[w] = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)
                    wings_const = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.)

                eps_inv_PQ = eps_body_inv

                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Wmn[km,k,:,:,w] = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])

                    if gw.fc:
                        # compute wing correction
                        Wn_P0 = einsum('Pnm,P->nm',Lij[kn],eps_inv_P0).diagonal()
                        Wn_P0 = Wn_P0.real * 2.
                        Del_P0[k,:,w] = wings_const * Wn_P0[orbs]
        else:
            for w in range(nw):
                Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),Pi_inv)
                    Wmn[km,k,:,:,w] = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])

    return Wmn, Del_00, Del_P0, qij, q_abs

def get_sigmaI_diag(gw, omega, kp, p, Wmn, Del_00, Del_P0, sign, freqs, wts):
    '''
    Compute self-energy by integrating on imaginary axis
    '''
    mo_energy = gw._scf.mo_energy
    nkpts = gw.nkpts
    sigma = 0j
    for k in range(nkpts):
        emo = omega - 1j*gw.eta*sign[k] - mo_energy[k]
        g0 = wts[None,:]*emo[:,None] / ((emo**2)[:,None]+(freqs**2)[None,:])
        sigma += -einsum('mw,mw',g0,Wmn[k])/np.pi
        if gw.fc and k == kp:
            sigma += -einsum('w,w->',Del_00,g0[p])/np.pi
            sigma += -einsum('w,w->',Del_P0,g0[p])/np.pi

    return sigma

def get_rho_response_R(gw, omega, mo_energy, Lpq, kL, kidx):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    nkpts, naux, nmo, nmo = Lpq.shape
    nocc = gw.nocc
    kpts = gw.kpts
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux,naux),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = mo_energy[i,:nocc,None] - mo_energy[a,None,nocc:]
        eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
        Pia = einsum('Pia,ia->Pia',Lpq[i][:,:nocc,nocc:],eia)
        # Response from both spin-up and spin-down density
        Pi += 2./nkpts * einsum('Pia,Qia->PQ',Pia,Lpq[i][:,:nocc,nocc:].conj())
    return Pi

def get_sigmaR_diag(gw, omega, kn, orbp, ef, freqs, qij, q_abs):
    '''
    Compute self-energy for poles inside contour
    (more and more expensive away from Fermi surface)
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    mydf = gw.with_df

    # possible kpts shift center
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    idx = []
    for k in range(nkpts):
        if omega > ef:
            fm = 1.0
            idx.append(np.where((mo_energy[k]<omega) & (mo_energy[k]>ef))[0])
        else:
            fm = -1.0
            idx.append(np.where((mo_energy[k]>omega) & (mo_energy[k]<ef))[0])

    sigmaR = 0j
    for kL in range(nkpts):
        # Lij: (ki, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
        km = kidx_r[kn]
        if len(idx[km]) > 0:
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                    if is_kconserv:
                        kidx[i] = j
                        kidx_r[j] = i
                        #logger.debug(gw, "Read Lpq (kL: %s / %s, ki: %s, kj: %s)"%(kL+1, nkpts, i, j))
                        Lij_out = None
                        # Read (L|pq) and ao2mo transform to (L|ij)
                        Lpq = []
                        for LpqR, LpqI, sign \
                                in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                            Lpq.append(LpqR+LpqI*1.0j)
                        # support unequal naux on different k points
                        Lpq = np.vstack(Lpq).reshape(-1,nmo**2)
                        tao = []
                        ao_loc = None
                        moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)
                        Lij.append(Lij_out.reshape(-1,nmo,nmo))
            Lij = np.asarray(Lij)
            naux = Lij.shape[1]

            if kL == 0:
                km = kidx_r[kn]
                if len(idx[km]) > 0:
                    for m in idx[km]:
                        em = mo_energy[km][m] - omega

                        # body dielectric matrix eps_body
                        Pi = get_rho_response_R(gw, abs(em), mo_energy, Lij, kL, kidx)
                        eps_body_inv = np.linalg.inv(np.eye(naux)-Pi)

                        if gw.fc and m == orbp:
                            # head dielectric matrix eps_00
                            Pi_00 = get_rho_response_head_R(gw, abs(em), mo_energy, qij)
                            eps_00 = 1. - 4. * np.pi/np.linalg.norm(q_abs[0])**2 * Pi_00

                            # wings dielectric matrix eps_P0
                            Pi_P0 = get_rho_response_wing_R(gw, abs(em), mo_energy, Lij, qij)
                            eps_P0 = -np.sqrt(4.*np.pi) / np.linalg.norm(q_abs[0]) * Pi_P0

                            # inverse dielectric matrix
                            eps_inv_00 = 1./(eps_00 - np.dot(np.dot(eps_P0.conj(),eps_body_inv),eps_P0))
                            eps_inv_P0 = -eps_inv_00 * np.dot(eps_body_inv, eps_P0)

                        eps_inv_PQ = eps_body_inv

                        # body
                        Qmn = einsum('P,PQ->Q',Lij[km][:,m,orbp].conj(),eps_inv_PQ-np.eye(naux))
                        Wmn = 1./nkpts * einsum('Q,Q->',Qmn,Lij[km][:,m,orbp])
                        sigmaR += fm * Wmn

                        if gw.fc and m == orbp:
                            # head correction
                            Del_00 = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)
                            sigmaR += fm * Del_00

                            # wings correction
                            wings_const = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.)
                            Wn_P0 = einsum('P,P->',Lij[kn][:,m,orbp].conj(),eps_inv_P0)
                            Wn_P0 = Wn_P0.real * 2.
                            sigmaR += fm * wings_const * Wn_P0
            else:
                km = kidx_r[kn]
                if len(idx[km]) > 0:
                    for m in idx[km]:
                        em = mo_energy[km][m] - omega
                        Pi = get_rho_response_R(gw, abs(em), mo_energy, Lij, kL, kidx)
                        Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                        Qmn = einsum('P,PQ->Q',Lij[km][:,m,orbp].conj(),Pi_inv)
                        Wmn = 1./nkpts * einsum('Q,Q->',Qmn,Lij[km][:,m,orbp])
                        sigmaR += fm * Wmn

    return sigmaR

def get_rho_response_head_R(gw, omega, mo_energy, qij):
    '''
    Compute head (G=0, G'=0) density response function in auxiliary basis at freq w
    '''
    nkpts, nocc, nvir = qij.shape
    nocc = gw.nocc
    kpts = gw.kpts

    # Compute Pi head
    Pi_00 = 0j
    for i, kpti in enumerate(kpts):
        eia = mo_energy[i,:nocc,None] - mo_energy[i,None,nocc:]
        eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
        Pi_00 += 2./nkpts * einsum('ia,ia->',eia,qij[i].conj()*qij[i])
    return Pi_00

def get_rho_response_wing_R(gw, omega, mo_energy, Lpq, qij):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    nkpts, naux, nmo, nmo = Lpq.shape
    nocc = gw.nocc
    kpts = gw.kpts

    # Compute Pi for kL
    Pi = np.zeros(naux,dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        eia = mo_energy[i,:nocc,None] - mo_energy[i,None,nocc:]
        eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
        eia_q = eia * qij[i].conj()
        Pi += 2./nkpts * einsum('Pia,ia->P',Lpq[i][:,:nocc,nocc:],eia_q)
    return Pi

def get_rho_response_head(gw, omega, mo_energy, qij):
    '''
    Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw
    '''
    nkpts, nocc, nvir = qij.shape
    nocc = gw.nocc
    kpts = gw.kpts

    # Compute Pi head
    Pi_00 = 0j
    for i, kpti in enumerate(kpts):
        eia = mo_energy[i,:nocc,None] - mo_energy[i,None,nocc:]
        eia = eia/(omega**2+eia*eia)
        Pi_00 += 4./nkpts * einsum('ia,ia->',eia,qij[i].conj()*qij[i])
    return Pi_00

def get_rho_response_wing(gw, omega, mo_energy, Lpq, qij):
    '''
    Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw
    '''
    nkpts, naux, nmo, nmo = Lpq.shape
    nocc = gw.nocc
    kpts = gw.kpts

    # Compute Pi wing
    Pi = np.zeros(naux,dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        eia = mo_energy[i,:nocc,None] - mo_energy[i,None,nocc:]
        eia = eia/(omega**2+eia*eia)
        eia_q = eia * qij[i].conj()
        Pi += 4./nkpts * einsum('Pia,ia->P',Lpq[i][:,:nocc,nocc:],eia_q)
    return Pi

def get_qij(gw, q, mo_coeff, uniform_grids=False):
    '''
    Compute qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2 at q: (nkpts, nocc, nvir)
    '''
    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo - nocc
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol
    mo_energy = gw._scf.mo_energy

    if uniform_grids:
        mydf = df.FFTDF(cell, kpts=kpts)
        coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        coords, weights = dft.gen_grid.get_becke_grids(cell,level=5)
    ngrid = len(coords)

    qij = np.zeros((nkpts,nocc,nvir),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpti, deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        if uniform_grids:
            ao_ao_grad = einsum('mg,xgn->xmn',ao.T.conj(),ao_grad) * cell.vol / ngrid
        else:
            ao_ao_grad = einsum('g,mg,xgn->xmn',weights,ao.T.conj(),ao_grad)
        q_ao_ao_grad = -1j * einsum('x,xmn->mn',q,ao_ao_grad)
        q_mo_mo_grad = np.dot(np.dot(mo_coeff[i][:,:nocc].T.conj(), q_ao_ao_grad), mo_coeff[i][:,nocc:])
        enm = 1./(mo_energy[i][nocc:,None] - mo_energy[i][None,:nocc])
        dens = enm.T * q_mo_mo_grad
        qij[i] = dens / np.sqrt(cell.vol)

    return qij

def _get_scaled_legendre_roots(nw):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    x0 = 0.5
    freqs_new = x0*(1.+freqs)/(1.-freqs)
    wts = wts*2.*x0/(1.-freqs)**2
    return freqs_new, wts

def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis quadrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)
    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w+1.0)/nw * np.pi/2.
        freqs[w] = a / np.tan(t)
        if w != nw-1:
            wts[w] = a*np.pi/2./nw/(np.sin(t)**2)
        else:
            wts[w] = a*np.pi/4./nw/(np.sin(t)**2)
    return freqs[::-1], wts[::-1]


class KRGWCD(lib.StreamObject):

    linearized = getattr(__config__, 'gw_gw_GW_linearized', False)
    eta = getattr(__config__, 'gw_gw_GW_eta', 1e-3)
    fc = getattr(__config__, 'gw_gw_GW_fc', True)

    _keys = {
        'linearized', 'eta', 'fc', 'frozen', 'mol', 'with_df',
        'kpts', 'nkpts', 'mo_energy', 'mo_coeff', 'mo_occ', 'sigma',
    }

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        #TODO: implement frozen orbs
        if frozen is not None and frozen > 0:
            raise NotImplementedError
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.sigma = None

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        logger.info(self, 'GW finite size corrections = %s', self.fc)
        return self

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, orbs=None, kptlist=None, nw=100):
        """
        Input:
            kptlist: self-energy k-points
            orbs: self-energy orbs
            nw: grid number
        Output:
            mo_energy: GW quasiparticle energy
        """
        if mo_coeff is None:
            mo_coeff = np.array(self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = np.array(self._scf.mo_energy)

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2*nkpts*nmo**2*naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
                kernel(self, mo_energy, mo_coeff, orbs=orbs,
                       kptlist=kptlist, nw=nw, verbose=self.verbose)

        logger.warn(self, 'GW QP energies may not be sorted from min to max')
        logger.timer(self, 'GW', *cput0)
        return self.mo_energy
