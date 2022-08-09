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
PBC spin-restricted G0W0-AC QP eigenvalues with k-point sampling
This implementation has N^4 scaling, and is faster than GW-CD (N^4)
and analytic GW (N^6) methods.
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
                           'KGWAC can be only used with GDF integrals' %
                           gw.with_df.__class__)

    vk = rhf.get_veff(gw.mol,dm_kpts=dm) - rhf.get_j(gw.mol,dm_kpts=dm)
    for k in range(nkpts):
        vk[k] = reduce(numpy.dot, (mo_coeff[k].T.conj(), vk[k], mo_coeff[k]))

    # Grids for integration on imaginary axis
    freqs,wts = _get_scaled_legendre_roots(nw)

    # Compute self-energy on imaginary axis i*[0,iw_cutoff]
    sigmaI, omega = get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=5.)

    # Analytic continuation
    coeff = []
    if gw.ac == 'twopole':
        for k in range(nklist):
            coeff.append(AC_twopole_diag(sigmaI[k], omega, orbs, nocc))
    elif gw.ac == 'pade':
        for k in range(nklist):
            coeff_tmp, omega_fit = AC_pade_thiele_diag(sigmaI[k], omega)
            coeff.append(coeff_tmp)
    coeff = np.array(coeff)

    conv = True
    # This code does not support metals
    homo = -99.
    lumo = 99.
    for k in range(nkpts):
        if homo < mf.mo_energy[k][nocc-1]:
            homo = mf.mo_energy[k][nocc-1]
        if lumo > mf.mo_energy[k][nocc]:
            lumo = mf.mo_energy[k][nocc]
    ef = (homo+lumo)/2.

    mo_energy = np.zeros_like(np.array(mf.mo_energy))
    for k in range(nklist):
        kn = kptlist[k]
        for p in orbs:
            if gw.linearized:
                # linearized G0W0
                de = 1e-6
                ep = mf.mo_energy[kn][p]
                #TODO: analytic sigma derivative
                if gw.ac == 'twopole':
                    sigmaR = two_pole(ep-ef, coeff[k,:,p-orbs[0]]).real
                    dsigma = two_pole(ep-ef+de, coeff[k,:,p-orbs[0]]).real - sigmaR.real
                elif gw.ac == 'pade':
                    sigmaR = pade_thiele(ep-ef, omega_fit[p-orbs[0]], coeff[k,:,p-orbs[0]]).real
                    dsigma = pade_thiele(ep-ef+de, omega_fit[p-orbs[0]], coeff[k,:,p-orbs[0]]).real - sigmaR.real
                zn = 1.0/(1.0-dsigma/de)
                e = ep + zn*(sigmaR.real + vk[kn,p,p].real - v_mf[kn,p,p].real)
                mo_energy[kn,p] = e
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    if gw.ac == 'twopole':
                        sigmaR = two_pole(omega-ef, coeff[k,:,p-orbs[0]]).real
                    elif gw.ac == 'pade':
                        sigmaR = pade_thiele(omega-ef, omega_fit[p-orbs[0]], coeff[k,:,p-orbs[0]]).real
                    return omega - mf.mo_energy[kn][p] - (sigmaR.real + vk[kn,p,p].real - v_mf[kn,p,p].real)
                try:
                    e = newton(quasiparticle, mf.mo_energy[kn][p], tol=1e-6, maxiter=100)
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

def get_sigma_diag(gw, orbs, kptlist, freqs, wts, iw_cutoff=None, max_memory=8000):
    '''
    Compute GW correlation self-energy (diagonal elements)
    in MO basis on imaginary axis
    '''
    mo_energy = np.array(gw._scf.mo_energy)
    mo_coeff = np.array(gw._scf.mo_coeff)
    nocc = gw.nocc
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

    # This code does not support metals
    homo = -99.
    lumo = 99.
    for k in range(nkpts):
        if homo < mo_energy[k][nocc-1]:
            homo = mo_energy[k][nocc-1]
        if lumo > mo_energy[k][nocc]:
            lumo = mo_energy[k][nocc]
    if (lumo-homo)<1e-3:
        logger.warn(gw, 'This GW-AC code is not supporting metals!')
    ef = (homo+lumo)/2.

    # Integration on numerical grids
    if iw_cutoff is not None:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    # Compute occ for -iw and vir for iw separately
    # to avoid branch cuts in analytic continuation
    omega_occ = np.zeros((nw_sigma), dtype=np.complex128)
    omega_vir = np.zeros((nw_sigma), dtype=np.complex128)
    omega_occ[1:] = -1j*freqs[:(nw_sigma-1)]
    omega_vir[1:] = 1j*freqs[:(nw_sigma-1)]
    orbs_occ = [i for i in orbs if i < nocc]
    norbs_occ = len(orbs_occ)

    emo_occ = np.zeros((nkpts,nmo,nw_sigma),dtype=np.complex128)
    emo_vir = np.zeros((nkpts,nmo,nw_sigma),dtype=np.complex128)
    for k in range(nkpts):
        emo_occ[k] = omega_occ[None,:] + ef - mo_energy[k][:,None]
        emo_vir[k] = omega_vir[None,:] + ef - mo_energy[k][:,None]

    sigma = np.zeros((nklist,norbs,nw_sigma),dtype=np.complex128)
    omega = np.zeros((norbs,nw_sigma),dtype=np.complex128)
    for p in range(norbs):
        orbp = orbs[p]
        if orbp < nocc:
            omega[p] = omega_occ.copy()
        else:
            omega[p] = omega_vir.copy()

    if gw.fc:
        # Set up q mesh for q->0 finite size correction
        q_pts = np.array([1e-3,0,0]).reshape(1,3)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = get_qij(gw, q_abs[0], mo_coeff)

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
                    # support uneqaul naux on different k points
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
                    Del_00 = 2./np.pi * (6.*np.pi**2/gw.mol.vol/nkpts)**(1./3.) * (eps_inv_00 - 1.)

                eps_inv_PQ = eps_body_inv
                g0_occ = wts[w] * emo_occ / (emo_occ**2+freqs[w]**2)
                g0_vir = wts[w] * emo_vir / (emo_vir**2+freqs[w]**2)

                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),eps_inv_PQ-np.eye(naux))
                    Wmn = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])
                    sigma[k][:norbs_occ] += -einsum('mn,mw->nw',Wmn[:,:norbs_occ],g0_occ[km])/np.pi
                    sigma[k][norbs_occ:] += -einsum('mn,mw->nw',Wmn[:,norbs_occ:],g0_vir[km])/np.pi

                    if gw.fc:
                        # apply head correction
                        assert (kn == km)
                        sigma[k][:norbs_occ] += -Del_00 * g0_occ[kn][orbs][:norbs_occ] /np.pi
                        sigma[k][norbs_occ:] += -Del_00 * g0_vir[kn][orbs][norbs_occ:] /np.pi

                        # apply wing correction
                        Wn_P0 = einsum('Pnm,P->nm',Lij[kn],eps_inv_P0).diagonal()
                        Wn_P0 = Wn_P0.real * 2.
                        Del_P0 = np.sqrt(gw.mol.vol/4./np.pi**3) * (6.*np.pi**2/gw.mol.vol/nkpts)**(2./3.) * Wn_P0[orbs]
                        sigma[k][:norbs_occ] += -einsum('n,nw->nw', Del_P0[:norbs_occ],
                                                        g0_occ[kn][orbs][:norbs_occ]) /np.pi
                        sigma[k][norbs_occ:] += -einsum('n,nw->nw', Del_P0[norbs_occ:],
                                                        g0_vir[kn][orbs][norbs_occ:]) /np.pi
        else:
            for w in range(nw):
                Pi = get_rho_response(gw, freqs[w], mo_energy, Lij, kL, kidx)
                Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
                g0_occ = wts[w] * emo_occ / (emo_occ**2+freqs[w]**2)
                g0_vir = wts[w] * emo_vir / (emo_vir**2+freqs[w]**2)
                for k in range(nklist):
                    kn = kptlist[k]
                    # Find km that conserves with kn and kL (-km+kn+kL=G)
                    km = kidx_r[kn]
                    Qmn = einsum('Pmn,PQ->Qmn',Lij[km][:,:,orbs].conj(),Pi_inv)
                    Wmn = 1./nkpts * einsum('Qmn,Qmn->mn',Qmn,Lij[km][:,:,orbs])
                    sigma[k][:norbs_occ] += -einsum('mn,mw->nw',Wmn[:,:norbs_occ],g0_occ[km])/np.pi
                    sigma[k][norbs_occ:] += -einsum('mn,mw->nw',Wmn[:,norbs_occ:],g0_vir[km])/np.pi

    return sigma, omega

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
    through kp perturbtation theory
    Ref: Phys. Rev. B 83, 245122 (2011)
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
                             gtol = 1e-10, max_nfev=1000, verbose=0, args=(omega[p], sigma[p]))
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

class KRGWAC(lib.StreamObject):

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
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('GW nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
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

if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc.lib import chkfile
    import os
    # This test takes a few minutes
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
               a = '''
               0.000000     1.783500     1.783500
               1.783500     0.000000     1.783500
               1.783500     1.783500     0.000000
               ''',
               atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
               dimension = 3,
               max_memory = 8000,
               verbose = 4,
               pseudo = 'gth-pade',
               basis='gth-szv',
               precision=1e-10)

    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_311.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf.xc = 'pbe'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    gw = KRGWAC(kmf)
    gw.linearized = False
    gw.ac = 'pade'
    # without finite size corrections
    gw.fc = False
    nocc = gw.nocc
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocc+3))
    print(gw.mo_energy)
    assert ((abs(gw.mo_energy[0][nocc-1]-0.62045797))<1e-5)
    assert ((abs(gw.mo_energy[0][nocc]-0.96574324))<1e-5)
    assert ((abs(gw.mo_energy[1][nocc-1]-0.52639137))<1e-5)
    assert ((abs(gw.mo_energy[1][nocc]-1.07513258))<1e-5)

    # with finite size corrections
    gw.fc = True
    gw.kernel(kptlist=[0,1,2],orbs=range(0,nocc+3))
    print(gw.mo_energy)
    assert ((abs(gw.mo_energy[0][nocc-1]-0.54277092))<1e-5)
    assert ((abs(gw.mo_energy[0][nocc]-0.80148537))<1e-5)
    assert ((abs(gw.mo_energy[1][nocc-1]-0.45073793))<1e-5)
    assert ((abs(gw.mo_energy[1][nocc]-0.92910108))<1e-5)
