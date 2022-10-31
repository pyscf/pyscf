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
PBC spin-unrestricted G0W0-AC QP eigenvalues with k-point sampling
GW-AC is recommended for valence states only, and is inaccuarate for core states.

Method:
    See T. Zhu and G.K.-L. Chan, arxiv:2007.03148 (2020) for details
    Compute Sigma on imaginary frequency with density fitting,
    then analytically continued to real frequency.
    Gaussian density fitting must be used (FFTDF and MDF are not supported).
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
from pyscf.pbc.cc.kccsd_uhf import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, orbs=None,
           kptlist=None, nw=None, verbose=logger.NOTE):
    '''
    GW-corrected quasiparticle orbital energies
    Returns:
        A list :  converged, mo_energy, mo_coeff
    '''
    mf = gw._scf
    if gw.frozen is None:
        frozen = 0
    else:
        frozen = gw.frozen
    assert (frozen == 0)

    nmoa, nmob = gw.nmo
    nocca, noccb = gw.nocc

    if orbs is None:
        orbs = range(nmoa)
    if kptlist is None:
        kptlist = range(gw.nkpts)

    nkpts = gw.nkpts
    nklist = len(kptlist)

    # v_xc
    dm = np.array(mf.make_rdm1())
    v_mf = np.array(mf.get_veff())
    vj = np.array(mf.get_j(dm_kpts=dm))
    v_mf[0] = v_mf[0] - (vj[0] + vj[1])
    v_mf[1] = v_mf[1] - (vj[0] + vj[1])
    for s in range(2):
        for k in range(nkpts):
            v_mf[s,k] = reduce(numpy.dot, (mo_coeff[s,k].T.conj(), v_mf[s,k], mo_coeff[s,k]))

    # v_hf from DFT/HF density
    if gw.fc:
        exxdiv = 'ewald'
    else:
        exxdiv = None
    uhf = scf.KUHF(gw.mol, gw.kpts, exxdiv=exxdiv)
    uhf.with_df = gw.with_df
    uhf.with_df._cderi = gw.with_df._cderi
    vk = uhf.get_veff(gw.mol,dm_kpts=dm)
    vj = uhf.get_j(gw.mol,dm_kpts=dm)
    vk[0] = vk[0] - (vj[0] + vj[1])
    vk[1] = vk[1] - (vj[0] + vj[1])
    for s in range(2):
        for k in range(nkpts):
            vk[s,k] = reduce(numpy.dot, (mo_coeff[s,k].T.conj(), vk[s,k], mo_coeff[s,k]))

    # Grids for integration on imaginary axis
    freqs,wts = _get_scaled_legendre_roots(nw)

    # Compute self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI, omega = get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=5.)

    # Analytic continuation
    coeff_a = []
    coeff_b = []
    if gw.ac == 'twopole':
        for k in range(nklist):
            coeff_a.append(AC_twopole_diag(sigmaI[0,k], omega[0], orbs, nocca))
            coeff_b.append(AC_twopole_diag(sigmaI[1,k], omega[1], orbs, noccb))
    elif gw.ac == 'pade':
        for k in range(nklist):
            coeff_a_tmp, omega_fit_a = AC_pade_thiele_diag(sigmaI[0,k], omega[0])
            coeff_b_tmp, omega_fit_b = AC_pade_thiele_diag(sigmaI[1,k], omega[1])
            coeff_a.append(coeff_a_tmp)
            coeff_b.append(coeff_b_tmp)
        omega_fit = np.asarray((omega_fit_a, omega_fit_b))
    coeff = np.asarray((coeff_a, coeff_b))

    conv = True
    # This code does not support metals
    homo = -99.
    lumo = 99.
    mo_energy = np.asarray(mf.mo_energy)
    for k in range(nkpts):
        if homo < max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1]):
            homo = max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1])
        if lumo > min(mo_energy[0,k][nocca],mo_energy[1,k][noccb]):
            lumo = min(mo_energy[0,k][nocca],mo_energy[1,k][noccb])
    ef = (homo+lumo)/2.

    mo_energy = np.zeros_like(np.array(mf.mo_energy))
    for s in range(2):
        for k in range(nklist):
            kn = kptlist[k]
            for p in orbs:
                if gw.linearized:
                    # linearized G0W0
                    de = 1e-6
                    ep = mf.mo_energy[s][kn][p]
                    #TODO: analytic sigma derivative
                    if gw.ac == 'twopole':
                        sigmaR = two_pole(ep-ef, coeff[s,k,:,p-orbs[0]]).real
                        dsigma = two_pole(ep-ef+de, coeff[s,k,:,p-orbs[0]]).real - sigmaR.real
                    elif gw.ac == 'pade':
                        sigmaR = pade_thiele(ep-ef, omega_fit[s,p-orbs[0]], coeff[s,k,:,p-orbs[0]]).real
                        dsigma = pade_thiele(ep-ef+de, omega_fit[s,p-orbs[0]],
                                             coeff[s,k,:,p-orbs[0]]).real - sigmaR.real
                    zn = 1.0/(1.0-dsigma/de)
                    e = ep + zn*(sigmaR.real + vk[s,kn,p,p].real - v_mf[s,kn,p,p].real)
                    mo_energy[s,kn,p] = e
                else:
                    # self-consistently solve QP equation
                    def quasiparticle(omega):
                        if gw.ac == 'twopole':
                            sigmaR = two_pole(omega-ef, coeff[s,k,:,p-orbs[0]]).real
                        elif gw.ac == 'pade':
                            sigmaR = pade_thiele(omega-ef, omega_fit[s,p-orbs[0]], coeff[s,k,:,p-orbs[0]]).real
                        return omega - mf.mo_energy[s][kn][p] - (sigmaR.real + vk[s,kn,p,p].real - v_mf[s,kn,p,p].real)
                    try:
                        e = newton(quasiparticle, mf.mo_energy[s][kn][p], tol=1e-6, maxiter=100)
                        mo_energy[s,kn,p] = e
                    except RuntimeError:
                        conv = False
    mo_coeff = mf.mo_coeff

    if gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmoa)
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy spin-up @ k%d =\n%s', k,mo_energy[0,k])
        for k in range(nkpts):
            logger.debug(gw, '  GW mo_energy spin-down @ k%d =\n%s', k,mo_energy[1,k])
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff

def get_rho_response(gw, omega, mo_energy, Lpq, kL, kidx):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    spin, nkpts, naux, nmo, nmo = Lpq.shape
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Compute Pi for kL
    Pi = np.zeros((naux,naux),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,a,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,a,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        Pia_a = einsum('Pia,ia->Pia',Lpq[0,i][:,:nocca,nocca:],eia_a)
        Pia_b = einsum('Pia,ia->Pia',Lpq[1,i][:,:noccb,noccb:],eia_b)
        # Response from both spin-up and spin-down density
        Pi += 2./nkpts * (einsum('Pia,Qia->PQ',Pia_a,Lpq[0,i][:,:nocca,nocca:].conj()) +
                          einsum('Pia,Qia->PQ',Pia_b,Lpq[1,i][:,:noccb,noccb:].conj()))
    return Pi

def get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=None, max_memory=8000):
    '''
    Compute GW correlation self-energy (diagonal elements) in MO basis
    on imaginary axis
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts
    nklist = len(kptlist)
    nw = len(freqs)
    norbs = len(orbs)
    mydf = gw.with_df

    # possible kpts shift
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # This code does not support metals
    homo = -99.
    lumo = 99.
    for k in range(nkpts):
        if homo < max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1]):
            homo = max(mo_energy[0,k][nocca-1],mo_energy[1,k][noccb-1])
        if lumo > min(mo_energy[0,k][nocca],mo_energy[1,k][noccb]):
            lumo = min(mo_energy[0,k][nocca],mo_energy[1,k][noccb])
    if (lumo-homo)<1e-3:
        logger.warn(gw, 'Current KUGW is not supporting metals!')
    ef = (homo+lumo)/2.

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    # Compute occ for -iw and vir for iw separately
    # to avoid branch cuts in analytic continuation
    omega_occ = np.zeros((nw_sigma),dtype=np.complex128)
    omega_vir = np.zeros((nw_sigma),dtype=np.complex128)
    omega_occ[1:] = -1j*freqs[:(nw_sigma-1)]
    omega_vir[1:] = 1j*freqs[:(nw_sigma-1)]
    orbs_occ_a = [i for i in orbs if i < nocca]
    orbs_occ_b = [i for i in orbs if i < noccb]
    norbs_occ_a = len(orbs_occ_a)
    norbs_occ_b = len(orbs_occ_b)

    emo_occ_a = np.zeros((nkpts,nmoa,nw_sigma),dtype=np.complex128)
    emo_occ_b = np.zeros((nkpts,nmob,nw_sigma),dtype=np.complex128)
    emo_vir_a = np.zeros((nkpts,nmoa,nw_sigma),dtype=np.complex128)
    emo_vir_b = np.zeros((nkpts,nmob,nw_sigma),dtype=np.complex128)
    for k in range(nkpts):
        emo_occ_a[k] = omega_occ[None,:] + ef - mo_energy[0,k][:,None]
        emo_occ_b[k] = omega_occ[None,:] + ef - mo_energy[1,k][:,None]
        emo_vir_a[k] = omega_vir[None,:] + ef - mo_energy[0,k][:,None]
        emo_vir_b[k] = omega_vir[None,:] + ef - mo_energy[1,k][:,None]

    sigma = np.zeros((2,nklist,norbs,nw_sigma),dtype=np.complex128)
    omega = np.zeros((2,norbs,nw_sigma),dtype=np.complex128)
    for s in range(2):
        for p in range(norbs):
            orbp = orbs[p]
            if orbp < gw.nocc[s]:
                omega[s,p] = omega_occ.copy()
            else:
                omega[s,p] = omega_vir.copy()

    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        q_pts = np.array([1e-3,0,0]).reshape(1,3)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = get_qij(gw, q_abs[0], mo_coeff)

    for kL in range(nkpts):
        # Lij: (2, ki, L, i, j) for looping every kL
        #Lij = np.zeros((2,nkpts,naux,nmoa,nmoa),dtype=np.complex128)
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
                    Lij_out_a = None
                    Lij_out_b = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign \
                            in mydf.sr_loop([kpti, kptj], max_memory=0.1*gw._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    Lpq = np.vstack(Lpq).reshape(-1,nmoa**2)
                    moija, ijslicea = _conc_mos(mo_coeff[0,i], mo_coeff[0,j])[2:]
                    moijb, ijsliceb = _conc_mos(mo_coeff[1,i], mo_coeff[1,j])[2:]
                    tao = []
                    ao_loc = None
                    Lij_out_a = _ao2mo.r_e2(Lpq, moija, ijslicea, tao, ao_loc, out=Lij_out_a)
                    tao = []
                    ao_loc = None
                    Lij_out_b = _ao2mo.r_e2(Lpq, moijb, ijsliceb, tao, ao_loc, out=Lij_out_b)
                    Lij.append(np.asarray((Lij_out_a.reshape(-1,nmoa,nmoa),Lij_out_b.reshape(-1,nmob,nmob))))

        Lij = np.asarray(Lij)
        Lij = Lij.transpose(1,0,2,3,4)
        naux = Lij.shape[2]

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
                    Del_00 = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)

                eps_inv_PQ = eps_body_inv
                g0_occ_a = wts[w] * emo_occ_a / (emo_occ_a**2+freqs[w]**2)
                g0_occ_b = wts[w] * emo_occ_b / (emo_occ_b**2+freqs[w]**2)
                g0_vir_a = wts[w] * emo_vir_a / (emo_vir_a**2+freqs[w]**2)
                g0_vir_b = wts[w] * emo_vir_b / (emo_vir_b**2+freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn_a = einsum('Pmn,PQ->Qmn',Lij[0,km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Qmn_b = einsum('Pmn,PQ->Qmn',Lij[1,km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Wmn_a = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_a,Lij[0,km][:,:,orbs])
                    Wmn_b = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_b,Lij[1,km][:,:,orbs])

                    sigma[0,k][:norbs_occ_a] += -einsum('mn,mw->nw',Wmn_a[:,:norbs_occ_a],g0_occ_a[km])/np.pi
                    sigma[1,k][:norbs_occ_b] += -einsum('mn,mw->nw',Wmn_b[:,:norbs_occ_b],g0_occ_b[km])/np.pi
                    sigma[0,k][norbs_occ_a:] += -einsum('mn,mw->nw',Wmn_a[:,norbs_occ_a:],g0_vir_a[km])/np.pi
                    sigma[1,k][norbs_occ_b:] += -einsum('mn,mw->nw',Wmn_b[:,norbs_occ_b:],g0_vir_b[km])/np.pi

                    if gw.fc:
                        # apply head correction
                        assert (kn == km)
                        sigma[0,k][:norbs_occ_a] += -Del_00 * g0_occ_a[kn][orbs][:norbs_occ_a] /np.pi
                        sigma[0,k][norbs_occ_a:] += -Del_00 * g0_vir_a[kn][orbs][norbs_occ_a:] /np.pi
                        sigma[1,k][:norbs_occ_b] += -Del_00 * g0_occ_b[kn][orbs][:norbs_occ_b] /np.pi
                        sigma[1,k][norbs_occ_b:] += -Del_00 * g0_vir_b[kn][orbs][norbs_occ_b:] /np.pi

                        # apply wing correction
                        Wn_P0_a = einsum('Pnm,P->nm',Lij[0,kn],eps_inv_P0).diagonal()
                        Wn_P0_b = einsum('Pnm,P->nm',Lij[1,kn],eps_inv_P0).diagonal()
                        Wn_P0_a = Wn_P0_a.real * 2.
                        Wn_P0_b = Wn_P0_b.real * 2.
                        Del_P0_a = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0_a[orbs]  # noqa: E501
                        Del_P0_b = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0_b[orbs]  # noqa: E501
                        sigma[0,k][:norbs_occ_a] += -einsum('n,nw->nw',Del_P0_a[:norbs_occ_a],g0_occ_a[kn][orbs][:norbs_occ_a]) /np.pi  # noqa: E501
                        sigma[0,k][norbs_occ_a:] += -einsum('n,nw->nw',Del_P0_a[norbs_occ_a:],g0_vir_a[kn][orbs][norbs_occ_a:]) /np.pi  # noqa: E501
                        sigma[1,k][:norbs_occ_b] += -einsum('n,nw->nw',Del_P0_b[:norbs_occ_b],g0_occ_b[kn][orbs][:norbs_occ_b]) /np.pi  # noqa: E501
                        sigma[1,k][norbs_occ_b:] += -einsum('n,nw->nw',Del_P0_b[norbs_occ_b:],g0_vir_b[kn][orbs][norbs_occ_b:]) /np.pi  # noqa: E501
        else:
            for w in range(nw):
                Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                g0_occ_a = wts[w] * emo_occ_a / (emo_occ_a**2+freqs[w]**2)
                g0_occ_b = wts[w] * emo_occ_b / (emo_occ_b**2+freqs[w]**2)
                g0_vir_a = wts[w] * emo_vir_a / (emo_vir_a**2+freqs[w]**2)
                g0_vir_b = wts[w] * emo_vir_b / (emo_vir_b**2+freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn_a = einsum('Pmn,PQ->Qmn',Lij[0,km][:,:,orbs].conj(),Pi_inv)
                    Qmn_b = einsum('Pmn,PQ->Qmn',Lij[1,km][:,:,orbs].conj(),Pi_inv)
                    Wmn_a = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_a,Lij[0,km][:,:,orbs])
                    Wmn_b = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn_b,Lij[1,km][:,:,orbs])

                    sigma[0,k][:norbs_occ_a] += -einsum('mn,mw->nw',Wmn_a[:,:norbs_occ_a],g0_occ_a[km])/np.pi
                    sigma[1,k][:norbs_occ_b] += -einsum('mn,mw->nw',Wmn_b[:,:norbs_occ_b],g0_occ_b[km])/np.pi
                    sigma[0,k][norbs_occ_a:] += -einsum('mn,mw->nw',Wmn_a[:,norbs_occ_a:],g0_vir_a[km])/np.pi
                    sigma[1,k][norbs_occ_b:] += -einsum('mn,mw->nw',Wmn_b[:,norbs_occ_b:],g0_vir_b[km])/np.pi

    return sigma, omega

def get_rho_response_head(gw, omega, mo_energy, qij):
    '''
    Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw
    '''
    qij_a, qij_b = qij
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    nkpts = len(kpts)

    # Compute Pi head
    Pi_00 = 0j
    for i, kpti in enumerate(kpts):
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,i,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,i,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        Pi_00 += 2./nkpts * (einsum('ia,ia->',eia_a,qij_a[i].conj()*qij_a[i]) +
                             einsum('ia,ia->',eia_b,qij_b[i].conj()*qij_b[i]))
    return Pi_00

def get_rho_response_wing(gw, omega, mo_energy, Lpq, qij):
    '''
    Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw
    '''
    qij_a, qij_b = qij
    spin, nkpts, naux, nmo, nmo = Lpq.shape
    nocca, noccb = gw.nocc
    kpts = gw.kpts
    nkpts = len(kpts)

    # Compute Pi wing
    Pi = np.zeros(naux,dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        eia_a = mo_energy[0,i,:nocca,None] - mo_energy[0,i,None,nocca:]
        eia_b = mo_energy[1,i,:noccb,None] - mo_energy[1,i,None,noccb:]
        eia_a = eia_a/(omega**2+eia_a*eia_a)
        eia_b = eia_b/(omega**2+eia_b*eia_b)
        eia_q_a = eia_a * qij_a[i].conj()
        eia_q_b = eia_b * qij_b[i].conj()
        Pi += 2./nkpts * (einsum('Pia,ia->P',Lpq[0,i][:,:nocca,nocca:],eia_q_a) +
                          einsum('Pia,ia->P',Lpq[1,i][:,:noccb,noccb:],eia_q_b))
    return Pi

def get_qij(gw, q, mo_coeff, uniform_grids=False):
    '''
    Compute qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2 at q: (nkpts, nocc, nvir)
    through kp perturbtation theory
    Ref: Phys. Rev. B 83, 245122 (2011)
    '''
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol
    mo_energy = np.asarray(gw._scf.mo_energy)

    if uniform_grids:
        mydf = df.FFTDF(cell, kpts=kpts)
        coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        coords, weights = dft.gen_grid.get_becke_grids(cell,level=4)
    ngrid = len(coords)

    qij_a = np.zeros((nkpts,nocca,nvira),dtype=np.complex128)
    qij_b = np.zeros((nkpts,noccb,nvirb),dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpti, deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        if uniform_grids:
            ao_ao_grad = einsum('mg,xgn->xmn',ao.T.conj(),ao_grad) * cell.vol / ngrid
        else:
            ao_ao_grad = einsum('g,mg,xgn->xmn',weights,ao.T.conj(),ao_grad)
        q_ao_ao_grad = -1j * einsum('x,xmn->mn',q,ao_ao_grad)
        q_mo_mo_grad_a = np.dot(np.dot(mo_coeff[0,i][:,:nocca].T.conj(), q_ao_ao_grad), mo_coeff[0,i][:,nocca:])
        q_mo_mo_grad_b = np.dot(np.dot(mo_coeff[1,i][:,:noccb].T.conj(), q_ao_ao_grad), mo_coeff[1,i][:,noccb:])
        enm_a = 1./(mo_energy[0,i][nocca:,None] - mo_energy[0,i][None,:nocca])
        enm_b = 1./(mo_energy[1,i][noccb:,None] - mo_energy[1,i][None,:noccb])
        dens_a = enm_a.T * q_mo_mo_grad_a
        dens_b = enm_b.T * q_mo_mo_grad_b
        qij_a[i] = dens_a / np.sqrt(cell.vol)
        qij_b[i] = dens_b / np.sqrt(cell.vol)

    return (qij_a, qij_b)

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
    Clenshaw-Curtis qaudrature on [0,inf)
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

def two_pole_fit(coeff, omega, sigma):
    cf = coeff[:5] + 1j*coeff[5:]
    f = cf[0] + cf[1]/(omega+cf[3]) + cf[2]/(omega+cf[4]) - sigma
    f[0] = f[0]/0.01
    return np.array([f.real,f.imag]).reshape(-1)

def two_pole(freqs, coeff):
    cf = coeff[:5] + 1j*coeff[5:]
    return cf[0] + cf[1]/(freqs+cf[3]) + cf[2]/(freqs+cf[4])

def AC_twopole_diag(sigma, omega, orbs, nocc):
    """
    Analytic continuation to real axis using a two-pole model
    Returns:
        coeff: 2D array (ncoeff, norbs)
    """
    norbs, nw = sigma.shape
    coeff = np.zeros((10,norbs))
    for p in range(norbs):
        if orbs[p] < nocc:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5])
        else:
            x0 = np.array([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5])
        #TODO: analytic gradient
        xopt = least_squares(two_pole_fit, x0, jac='3-point', method='trf', xtol=1e-10,
                             gtol = 1e-10, max_nfev=2000, verbose=0, args=(omega[p], sigma[p]))
        if xopt.success is False:
            print('WARN: 2P-Fit Orb %d not converged, cost function %e'%(p,xopt.cost))
        coeff[:,p] = xopt.x.copy()
    return coeff

def thiele(fn,zn):
    nfit = len(zn)
    g = np.zeros((nfit,nfit),dtype=np.complex128)
    g[:,0] = fn.copy()
    for i in range(1,nfit):
        g[i:,i] = (g[i-1,i-1]-g[i:,i-1])/((zn[i:]-zn[i-1])*g[i:,i-1])
    a = g.diagonal()
    return a

def pade_thiele(freqs,zn,coeff):
    nfit = len(coeff)
    X = coeff[-1]*(freqs-zn[-2])
    for i in range(nfit-1):
        idx = nfit-i-1
        X = coeff[idx]*(freqs-zn[idx-1])/(1.+X)
    X = coeff[0]/(1.+X)
    return X

def AC_pade_thiele_diag(sigma, omega):
    """
    Analytic continuation to real axis using a Pade approximation
    from Thiele's reciprocal difference method
    Reference: J. Low Temp. Phys. 29, 179 (1977)
    Returns:
        coeff: 2D array (ncoeff, norbs)
        omega: 2D array (norbs, npade)
    """
    idx = range(1,40,6)
    sigma1 = sigma[:,idx].copy()
    sigma2 = sigma[:,(idx[-1]+4)::4].copy()
    sigma = np.hstack((sigma1,sigma2))
    omega1 = omega[:,idx].copy()
    omega2 = omega[:,(idx[-1]+4)::4].copy()
    omega = np.hstack((omega1,omega2))
    norbs, nw = sigma.shape
    npade = nw // 2
    coeff = np.zeros((npade*2,norbs),dtype=np.complex128)
    for p in range(norbs):
        coeff[:,p] = thiele(sigma[p,:npade*2], omega[p,:npade*2])

    return coeff, omega[:,:npade*2]


class KUGWAC(lib.StreamObject):

    linearized = getattr(__config__, 'gw_gw_GW_linearized', False)
    # Analytic continuation: pade or twopole
    ac = getattr(__config__, 'gw_gw_GW_ac', 'pade')
    # Whether applying finite size corrections
    fc = getattr(__config__, 'gw_gw_GW_fc', True)

    def __init__(self, mf, frozen=0):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        #TODO: implement frozen orbs
        if frozen > 0:
            raise NotImplementedError
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

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

        keys = set(('linearized','ac','fc'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        nkpts = self.nkpts
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d), nkpts = %d',
                 nocca, noccb, nvira, nvirb, nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        logger.info(self, 'analytic continuation method = %s', self.ac)
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

        nmoa, nmob = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (3*nkpts*nmoa**2*naux) * 16/1e6
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

if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc.lib import chkfile
    import os
    cell = gto.Cell()
    cell.build(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = 'gth-dzvp',
        pseudo = 'gth-pade',
        verbose = 5,
        charge = 0,
        spin = 1)

    cell.spin = cell.spin * 3
    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'h3_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'h_311.chk'
    if os.path.isfile(chkfname):
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KUGWAC(kmf)
    gw.linearized = False
    gw.ac = 'pade'
    gw.fc = False
    nocca, noccb = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocca+3))
    print(gw.mo_energy)
    assert ((abs(gw.mo_energy[0][0][nocca-1]--0.28012813))<1e-5)
    assert ((abs(gw.mo_energy[0][0][nocca]-0.13748876))<1e-5)
    assert ((abs(gw.mo_energy[0][1][nocca-1]--0.29515851))<1e-5)
    assert ((abs(gw.mo_energy[0][1][nocca]-0.14128011))<1e-5)
    assert ((abs(gw.mo_energy[1][0][noccb-1]--0.33991721))<1e-5)
    assert ((abs(gw.mo_energy[1][0][noccb]-0.10578847))<1e-5)
    assert ((abs(gw.mo_energy[1][1][noccb-1]--0.33547973))<1e-5)
    assert ((abs(gw.mo_energy[1][1][noccb]-0.08053408))<1e-5)

    gw.fc = True
    nocca, noccb = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocca+3))
    print(gw.mo_energy)
    assert ((abs(gw.mo_energy[0][0][nocca-1]--0.40244058))<1e-5)
    assert ((abs(gw.mo_energy[0][0][nocca]-0.13618348))<1e-5)
    assert ((abs(gw.mo_energy[0][1][nocca-1]--0.41743063))<1e-5)
    assert ((abs(gw.mo_energy[0][1][nocca]-0.13997427))<1e-5)
    assert ((abs(gw.mo_energy[1][0][noccb-1]--0.46133481))<1e-5)
    assert ((abs(gw.mo_energy[1][0][noccb]-0.1044926))<1e-5)
    assert ((abs(gw.mo_energy[1][1][noccb-1]--0.4568894))<1e-5)
    assert ((abs(gw.mo_energy[1][1][noccb]-0.07922511))<1e-5)
