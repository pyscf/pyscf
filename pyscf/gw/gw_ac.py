#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

'''
Spin-restricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N^5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
'''

import h5py
import numpy as np
from scipy.optimize import newton
import scipy.linalg as sla
import time

from pyscf import df, dft, lib, scf
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import einsum, logger, temporary_env
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.scf.addons import _fermi_smearing_occ, _gaussian_smearing_occ, _smearing_optimize

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.gw.utils import arraymath


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nocc = gw.nocc

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and (not gw.outcore):
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq

    # mean-field exchange-correlation
    with temporary_env(mf, verbose=0):
        v_mf = mo_coeff_frz.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff_frz
    gw.vxc = v_mf

    # exchange self-energy
    if gw.vhf_df and gw.outcore is False:
        # TODO: support smearing
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    # compute self-energy on imaginary axis
    if gw.outcore:
        sigmaI, omega = get_sigma_outcore(
            gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )
    else:
        sigmaI, omega = get_sigma(
            gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(orbs_frz, nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        if gw.ac_idx is not None:
            acobj.idx = gw.ac_idx
    acobj.ac_fit(sigmaI, omega, axis=-1)

    # get GW quasiparticle energy
    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=0, axis2=1)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for ip, (p_in_frz, p_in_all) in enumerate(zip(orbs_frz, orbs)):
        if gw.qpe_linearized is True:
            # linearized G0W0
            de = 1e-6
            sigmaR = diag_acobj[ip].ac_eval(gw._scf.mo_energy[p_in_all]).real
            dsigma = diag_acobj[ip].ac_eval(gw._scf.mo_energy[p_in_all] + de).real - sigmaR.real
            zn = 1.0 / (1.0 - dsigma / de)
            if gw.qpe_linearized_range is not None:
                zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
            mo_energy[p_in_all] = gw._scf.mo_energy[p_in_all] + \
                    zn * (sigmaR.real + vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz])
        else:
            # self-consistently solve QP equation
            def quasiparticle(omega):
                sigmaR = diag_acobj[ip].ac_eval(omega).real
                return omega - gw._scf.mo_energy[p_in_all] - (sigmaR + vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz])

            try:
                mo_energy[p_in_all] = newton(
                    quasiparticle, gw._scf.mo_energy[p_in_all], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                )
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', p_in_all)

    # save GW results
    gw.acobj = acobj
    gw.mo_energy = mo_energy
    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    if gw.writefile > 0:
        fn = 'vxc.h5'
        feri = h5py.File(fn, 'w')
        feri['vk'] = np.asarray(vk)
        feri['v_mf'] = np.asarray(v_mf)
        feri.close()

        fn = 'sigma_imag.h5'
        feri = h5py.File(fn, 'w')
        feri['sigmaI'] = np.asarray(sigmaI)
        feri['omega'] = np.asarray(omega)
        feri.close()

        acobj.save('ac_coeff.h5')

    return


def get_rho_response(omega, mo_energy, Lia, out=None):
    """Compute density-density response function in auxiliary basis at freq iw.
    See equation 58 in 10.1088/1367-2630/14/5/053020,
    and equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    Lia : double 3d ndarray
        occ-vir block of three-center density-fitting matrix
    out : double 2d array, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux, nocc, nvir = Lia.shape
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    # factor 4.0 comes from (1) spin-up and spin-down (2) conjugated term
    eia = 4.0 * eia / (omega**2 + eia**2)
    Pia = lib.broadcast_mul(Lia, eia)

    Pi = np.matmul(Pia.reshape(naux, nocc * nvir), Lia.reshape(naux, nocc * nvir).T, out=out)
    return Pi


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, out=None):
    """Get response function in auxiliary basis for metallic systems.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 1d array
        orbital energy
    mo_occ : double 1d array
        occupation number with a factor of 2
    Lpq : double 3d array
        three-center density-fitting matrix in MO
    out : double ndarray, optional
        a location into which the result is stored, by default None

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux = Lpq.shape[0]
    eia = mo_energy[:, None] - mo_energy[None, :]
    fia = mo_occ[:, None] - mo_occ[None, :]

    # factor 4.0 comes from (1) spin-up and spin-down (2) conjugated term
    # both ia and ai are included, this gives a factor of 2.0
    # restricted mo_occ gives another factor of 2.0
    eia = eia * fia / (omega**2 + eia**2)
    Pia = lib.broadcast_mul(Lpq, eia)

    Pi = np.matmul(Pia.reshape(naux, -1), Lpq.reshape(naux, -1).T, out=out)
    return Pi


def get_rho_response_head(omega, mo_energy, qij):
    """Compute head (G=0, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double ndarray
        orbital energy
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi_00 : complex
        head response function
    """
    nocc = qij.shape[0]

    Pi_00 = 0j
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia**2)
    Pi_00 += 4.0 * einsum('ia,ia->', eia, qij.conj() * qij)
    return Pi_00


def get_rho_response_wing(omega, mo_energy, Lia, qij):
    """Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw.
    equation 48 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        frequency point
    mo_energy : double 2d array
        orbital energy
    Lia : complex 4d array
        occupied-virtual block of three-center density fitting matrix in MO
    qij : complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi : complex ndarray
        wing response function
    """
    naux, nocc, nvir = Lia.shape

    Pi = np.zeros(shape=[naux], dtype=np.complex128)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    eia = eia / (omega**2 + eia**2)
    eia_q = eia * qij.conj()
    Pi += 4.0 * np.matmul(Lia.reshape(naux, nocc * nvir), eia_q.reshape(nocc * nvir))
    return Pi


def get_qij(gw, q, mo_energy, mo_coeff, uniform_grids=False):
    """Compute pair density matrix in the long-wavelength limit through kp perturbation theory
    k=0 only for gamma point
    qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2
    equation 51 in 10.1021/acs.jctc.0c00704
    Ref: Phys. Rev. B 83, 245122 (2011)

    Parameters
    ----------
    gw : GWAC
        gw object, provides attributes: nocc, nmo, kpts, mol
    q : double
        q grid
    mo_energy : double ndarray
        orbital energy
    mo_coeff : complex ndarray
        coefficient from AO to MO
    uniform_grids : bool, optional
        use uniform grids, by default False

    Returns
    -------
    qij : complex ndarray
        pair density matrix in the long-wavelength limit
    """
    from pyscf import pbc
    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo - nocc
    cell = gw.mol

    if uniform_grids:
        with temporary_env(cell, verbose=0):
            mydf = pbc.df.FFTDF(cell)
            coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        with temporary_env(cell, verbose=0):
            coords, weights = pbc.dft.gen_grid.get_becke_grids(cell, level=4)
    ngrid = len(coords)

    qij = np.zeros(shape=[nocc, nvir], dtype=np.complex128)
    ao_p = pbc.dft.numint.eval_ao(cell, coords, deriv=1)
    ao = ao_p[0]
    ao_grad = ao_p[1:4]
    if uniform_grids:
        ao_ao_grad = einsum('mg,xgn->xmn', ao.T.conj(), ao_grad) * cell.vol / ngrid
    else:
        ao_ao_grad = einsum('g,mg,xgn->xmn', weights, ao.T.conj(), ao_grad)
    q_ao_ao_grad = -1j * einsum('x,xmn->mn', q, ao_ao_grad)
    q_mo_mo_grad = mo_coeff[:, :nocc].T.conj() @ q_ao_ao_grad @ mo_coeff[:, nocc:]
    enm = 1.0 / (mo_energy[nocc:, None] - mo_energy[None, :nocc])
    dens = enm.T * q_mo_mo_grad
    qij = dens / np.sqrt(cell.vol)

    return qij


def get_sigma(
        gw, orbs, Lpq, quad_freqs, quad_wts, ef, mo_energy, mo_coeff=None, mo_occ=None, iw_cutoff=None, eval_freqs=None,
        mo_energy_w=None, fullsigma=False):
    """Compute GW correlation self-energy on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : GWAC
        GW object
    orbs : list
        list of orbital indexes
    Lpq : double 3d array
        three-center density-fitting matrix in MO space
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 1d array
        orbital energy in G
    mo_coeff : double 2d array
        coefficient from AO to MO
    mo_occ : double 1d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        imaginary frequency points at which to evaluate self energy.
        Self energy is evaluated at ef + 1j * eval_freqs.
    mo_energy_w : double 1d array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 2d or 3d ndarray
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nquadfreqs = len(quad_freqs)
    nevalfreqs = len(eval_freqs)
    naux, nmo, _ = Lpq.shape
    norbs = len(orbs)

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy

    if mo_coeff is None:
        mo_coeff = _mo_without_core(gw, gw.mo_coeff)

    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    # integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(eval_freqs < iw_cutoff)
    else:
        nw_sigma = nevalfreqs
    nw_cutoff = nw_sigma if iw_cutoff is None else sum(eval_freqs < iw_cutoff)

    omega = ef + 1j * eval_freqs[:nw_sigma]
    emo = omega[None] - mo_energy_g[:, None]

    if fullsigma is False:
        sigma_real = np.zeros(shape=[nw_sigma, norbs], dtype=np.double)
        sigma_imag = np.zeros(shape=[nw_sigma, norbs], dtype=np.double)
    else:
        sigma_real = np.zeros(shape=[nw_sigma, norbs, norbs], dtype=np.double)
        sigma_imag = np.zeros(shape=[nw_sigma, norbs, norbs], dtype=np.double)

    if gw.fc is True:
        if fullsigma is False:
            sigma_fc = np.zeros(shape=[norbs, nw_sigma], dtype=np.complex128)
        else:
            sigma_fc = np.zeros(shape=[norbs, norbs, nw_sigma], dtype=np.complex128)

    # make density-fitting matrix for contractions
    if hasattr(gw._scf, 'sigma') is False:
        Lia = np.ascontiguousarray(Lpq[:, :nocc, nocc:])
    # assume Lpq = Lpq, so we don't generate Lpq[:, mkslice(orbs), :]
    l_slice = Lpq[:, :, arraymath.mkslice(orbs)].reshape(naux, -1)

    # self-energy is calculated as equation 27 in doi.org/10.1021/acs.jctc.0c00704
    logger.info(gw, 'Starting get_sigma_diag main loop with %d frequency points.', nquadfreqs)
    Pi = None
    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = np.ones(shape=[1, naux], dtype=np.double)
    else:
        Qmn = np.zeros(shape=[naux, norbs], dtype=np.double)
        Wmn = np.zeros(shape=[nmo, norbs, norbs], dtype=np.double)

    if gw.fc is True:
        # Set up q mesh for q->0 finite size correction
        if not gw.fc_grid:
            q_pts = np.array([1e-3, 0, 0], dtype=np.double).reshape(1, 3)
        else:
            Nq = 3
            q_pts = np.zeros(shape=[Nq**3 - 1, 3], dtype=np.double)
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        else:
                            q_pts[i * Nq**2 + j * Nq + k - 1, 0] = k * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 1] = j * 5e-4
                            q_pts[i * Nq**2 + j * Nq + k - 1, 2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = gw.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nocc, nvir)
        qij = np.zeros(shape=[nq_pts, nocc, nmo - nocc], dtype=np.complex128)

        if not gw.fc_grid:
            for k in range(nq_pts):
                qij[k] = get_qij(gw, q_abs[k], mo_energy, mo_coeff)

    for w in range(nquadfreqs):
        if gw.verbose >= 4:
            gw.stdout.write(f'{w:4d} ')
            if w % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        # Pi_inv = (I - Pi)^-1 - I.
        if hasattr(gw._scf, 'sigma'):
            Pi = get_rho_response_metal(quad_freqs[w], mo_energy_w, mo_occ, Lpq, out=Pi)
        else:
            Pi = get_rho_response(quad_freqs[w], mo_energy_w, Lia)
        # Pi_inv contains (I - Pi)^-1.
        Pi_inv = arraymath.get_id_minus_pi_inv(Pi, overwrite_input=True)

        if gw.fc is True:
            eps_inv_00 = 0j
            eps_inv_P0 = np.zeros(shape=[naux], dtype=np.complex128)
            for iq in range(nq_pts):
                # head dielectric matrix eps_00, equation 47 in 10.1021/acs.jctc.0c00704
                Pi_00 = get_rho_response_head(quad_freqs[w], mo_energy, qij[iq])
                eps_00 = 1.0 - 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2.0 * Pi_00

                # wings dielectric matrix eps_P0, equation 48 in 10.1021/acs.jctc.0c00704
                Pi_P0 = get_rho_response_wing(quad_freqs[w], mo_energy, Lia, qij[iq])
                eps_P0 = -np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                # inverse dielectric matrix
                # equation 53 in 10.1021/acs.jctc.0c00704
                eps_inv_00 += 1.0 / nq_pts * 1.0 / (eps_00 - eps_P0.conj() @ Pi_inv @ eps_P0)
                # equation 54 in 10.1021/acs.jctc.0c00704
                eps_inv_P0 += 1.0 / nq_pts * (-eps_inv_00) * np.matmul(Pi_inv, eps_P0)

            # head correction, equation 43 in 10.1021/acs.jctc.0c00704
            Del_00 = 2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol) ** (1.0 / 3.0) * (eps_inv_00 - 1.0)

        # second line in equation 27
        # Pi_inv now contains (I - Pi)^-1 - I.
        arraymath.addto_diagonal(Pi_inv, -1.0)
        g0 = quad_wts[w] * emo / (emo**2 + quad_freqs[w] ** 2)

        # split g0 into real and imag parts to avoid costly type conversions
        g0r = np.ascontiguousarray(g0.real)
        g0i = np.ascontiguousarray(g0.imag)

        if fullsigma is False:
            # n is the index of orbitals in orbs, m is the index of orbitals of nmo
            # last line of equation 27, contraction from left to right
            # Qmn = \sum_P V^{nm}_P (Pi_inv)_{PQ} = \sum_P V^{mn}_P (Pi_inv)_{PQ}
            Qmn = np.matmul(Pi_inv, l_slice, out=Qmn)

            # Qmn = Qmn v^{mn}_Q
            Qmn *= l_slice

            # Wmn = \sum_Q Qmn
            Wmn = np.matmul(naux_ones, Qmn, out=Wmn)

            # sigma -= einsum('mn, mw -> nw', Wmn, g0) / np.pi
            sla.blas.dgemm(
                alpha=-1.0 / np.pi,
                a=Wmn.reshape(nmo, norbs).T,
                b=g0r.T,
                c=sigma_real.T,
                trans_a=0,
                trans_b=1,
                beta=1.0,
                overwrite_c=True,
            )
            sla.blas.dgemm(
                alpha=-1.0 / np.pi,
                a=Wmn.reshape(nmo, norbs).T,
                b=g0i.T,
                c=sigma_imag.T,
                trans_a=0,
                trans_b=1,
                beta=1.0,
                overwrite_c=True,
            )
        else:
            # n and n' are the index of orbitals in orbs, m is the index of orbitals of nmo
            # last line of equation 27, contraction from left to right
            # Qmn = \sum_P V^{nm}_P (PiV)_{PQ} = \sum_P V^{mn}_P (PiV)_{PQ}
            for orbm in range(nmo):
                np.matmul(Pi_inv, l_slice.reshape(naux, nmo, norbs)[:, orbm, :], out=Qmn)

                # Wmn is actually Wmnn'
                # Wmnn' = \sum_Q Qmn v^{mn'}_Q
                np.matmul(Qmn.T, l_slice.reshape(naux, nmo, norbs)[:, orbm, :], out=Wmn[orbm])

            # sigma -= einsum('mnl,mw->wnl', Wmn, g0)/np.pi
            sla.blas.dgemm(
                alpha=-1.0 / np.pi,
                a=Wmn.reshape(nmo, norbs * norbs).T,
                b=g0r.T,
                c=sigma_real.reshape(nw_sigma, norbs * norbs).T,
                trans_a=0,
                trans_b=1,
                beta=1.0,
                overwrite_c=True,
            )
            sla.blas.dgemm(
                alpha=-1.0 / np.pi,
                a=Wmn.reshape(nmo, norbs * norbs).T,
                b=g0i.T,
                c=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
                trans_a=0,
                trans_b=1,
                beta=1.0,
                overwrite_c=True,
            )

        if gw.fc is True:
            if fullsigma is False:
                # head correction
                sigma_fc += -Del_00 * g0[orbs] / np.pi

                # wing correction
                Wn_P0 = einsum('Pnn,P->n', Lpq, eps_inv_P0)
                Wn_P0 = Wn_P0[orbs].real * 2.0
                Del_P0 = np.sqrt(gw.mol.vol / 4 / np.pi**3) * (6 * np.pi**2 / gw.mol.vol) ** (2 / 3) * Wn_P0
                sigma_fc += -einsum('n,nw->nw', Del_P0, g0[orbs]) / np.pi
            else:
                # head correction
                tmp = -Del_00 * g0[orbs] / np.pi
                sigma_fc[np.arange(norbs), np.arange(norbs), :] += tmp

                # wing correction
                Wn_P0 = einsum('Pnn,P->n', Lpq, eps_inv_P0)
                Wn_P0 = Wn_P0[orbs].real * 2.0
                Del_P0 = np.sqrt(gw.mol.vol / 4 / np.pi**3) * (6 * np.pi**2 / gw.mol.vol) ** (2 / 3) * Wn_P0
                tmp = -einsum('n,nw->nw', Del_P0, g0[orbs]) / np.pi
                sigma_fc[np.arange(norbs), np.arange(norbs), :] += tmp

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = np.ascontiguousarray(sigma.transpose(1, 0))
    else:
        sigma = np.ascontiguousarray(sigma.transpose(1, 2, 0))

    if gw.fc is True:
        sigma += sigma_fc

    logger.info(gw, '\nFinished get_sigma_diag main loop.')

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_sigma_outcore(gw, orbs, quad_freqs, quad_wts, ef, mo_energy, mo_coeff, mo_occ=None,
                      iw_cutoff=None, eval_freqs=None, mo_energy_w=None, fullsigma=False):
    """Low-memory routine to compute GW correlation self-energy on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : GWAC
        GW object
    orbs : list
        list of orbital indexes
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 1d array
        orbital energy in G
    mo_coeff : double 2d array
        coefficient from AO to MO
    mo_occ : double 2d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        imaginary frequency points at which to evaluate self energy.
        Self energy is evaluated at ef + 1j * eval_freqs.
    mo_energy_w : double array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 2d or 3d array
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nmo = gw.nmo
    nw = len(quad_freqs)
    nw2 = len(eval_freqs)
    norbs = len(orbs)
    with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
        naux = gw.with_df.get_naoaux()

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy

    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    # integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(eval_freqs < iw_cutoff)
    else:
        nw_sigma = nw2
    nw_cutoff = nw_sigma if iw_cutoff is None else sum(eval_freqs < iw_cutoff)

    omega = ef + 1j * eval_freqs[:nw_sigma]

    Pi = np.zeros(shape=[nw, naux, naux], dtype=np.double)
    if hasattr(gw._scf, 'sigma'):
        nseg = nmo // gw.segsize + 1
    else:
        nseg = nocc // gw.segsize + 1
    for i in range(nseg):
        if hasattr(gw._scf, 'sigma'):
            orb_start = i * gw.segsize
            orb_end = min((i + 1) * gw.segsize, nmo)
            ijslice = (orb_start, orb_end, 0, nmo)
        else:
            orb_start = i * gw.segsize
            orb_end = min((i + 1) * gw.segsize, nocc)
            ijslice = (orb_start, orb_end, nocc, nmo)
        Lia = gw.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)

        for w in range(nw):
            if hasattr(gw._scf, 'sigma'):
                eia = mo_energy_w[orb_start:orb_end, None] - mo_energy_w[None, :]
                fia = mo_occ[orb_start:orb_end, None] - mo_occ[None, :]
                eia = eia * fia / (quad_freqs[w] ** 2 + eia**2)
                Pia = lib.broadcast_mul(Lia, eia)
            else:
                eia = mo_energy_w[orb_start:orb_end, None] - mo_energy_w[None, nocc:]
                eia = 4.0 * eia / (quad_freqs[w] ** 2 + eia**2)
                Pia = lib.broadcast_mul(Lia, eia)
            Pi[w] += np.matmul(Pia.reshape(naux, -1), Lia.reshape(naux, -1).T)
            del Pia
        del Lia

    for w in range(nw):
        Pi[w] = np.linalg.inv(np.eye(naux) - Pi[w]) - np.eye(naux)
    Pi_inv = Pi

    logger.info(gw, f'Starting get_sigma_diag_outcore main loop with {nseg} segments.')
    if fullsigma is False:
        sigma_real = np.zeros(shape=[nw_sigma, norbs], dtype=np.double)
        sigma_imag = np.zeros(shape=[nw_sigma, norbs], dtype=np.double)
    else:
        sigma_real = np.zeros(shape=[nw_sigma, norbs, norbs], dtype=np.double)
        sigma_imag = np.zeros(shape=[nw_sigma, norbs, norbs], dtype=np.double)

    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = np.ones((1, naux))

    emo = omega[None] - mo_energy_g[:, None]
    nseg = nmo // gw.segsize + 1
    for i in range(nseg):
        if gw.verbose >= 4:
            gw.stdout.write(f'{i:4d} ')
            if i % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        orb_start = i * gw.segsize
        orb_end = min((i + 1) * gw.segsize, nmo)
        ijslice = (orb_start, orb_end, 0, nmo)
        Lpq = gw.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice)
        l_slice = np.ascontiguousarray(Lpq[:, :, arraymath.mkslice(orbs)].reshape(naux, -1))
        del Lpq

        for w in range(nw):
            g0 = quad_wts[w] * emo[orb_start:orb_end] / (emo[orb_start:orb_end] ** 2 + quad_freqs[w] ** 2)
            # split g0 into real and imag parts to avoid costly type conversions
            g0r = np.ascontiguousarray(g0.real)
            g0i = np.ascontiguousarray(g0.imag)

            if fullsigma is False:
                Qmn = np.matmul(Pi_inv[w].T, l_slice)
                Qmn *= l_slice
                Wmn = np.matmul(naux_ones, Qmn)

                # sigma -= einsum('mn,mw->wn', Wmn, g0) / np.pi
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(orb_end - orb_start, norbs).T,
                    b=g0r.T,
                    c=sigma_real.T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(orb_end - orb_start, norbs).T,
                    b=g0i.T,
                    c=sigma_imag.T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )
            else:
                Wmn = np.zeros(shape=[orb_end - orb_start, norbs, norbs], dtype=np.double)
                for orbm in range(orb_end - orb_start):
                    Qmn = np.matmul(Pi_inv[w], l_slice.reshape(naux, orb_start - orb_end, norbs)[:, orbm, :])
                    np.matmul(Qmn.T, l_slice.reshape(naux, orb_start - orb_end, norbs)[:, orbm, :], out=Wmn[orbm])

                # sigma -= einsum('mnl,mw->wnl', Wmn, g0)/np.pi
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                    b=g0r.T,
                    c=sigma_real.reshape(nw_sigma, norbs * norbs).T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                    b=g0i.T,
                    c=sigma_imag.reshape(nw_sigma, norbs * norbs).T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = np.ascontiguousarray(sigma.transpose(1, 0))
    else:
        sigma = np.ascontiguousarray(sigma.transpose(1, 2, 0))

    logger.info(gw, '\nFinished get_sigma_diag_outcore main loop.')

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_g0(omega, mo_energy, eta):
    """Get non-interacting Green's function.

    Parameters
    ----------
    omega : double or complex array
        frequency grids
    mo_energy : double 1d array
        orbital energy
    eta : double
        broadening parameter

    Returns
    -------
    gf0 : complex 3d array
        non-interacting Green's function
    """
    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros(shape=[nmo, nmo, nw], dtype=np.complex128)
    gf0[np.arange(nmo), np.arange(nmo), :] = 1.0 / (omega[np.newaxis, :] + 1j * eta - mo_energy[:, np.newaxis])
    return gf0


def _mo_energy_without_core(gw, mo_energy):
    """Get non-frozen orbital energy.

    Parameters
    ----------
    gw : GWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo_energy : double 1d array
        full orbital energy

    Returns
    -------
    mo_energy : double 1d array
        non-frozen orbital energy
    """
    return mo_energy[get_frozen_mask(gw)]


def _mo_without_core(gw, mo):
    """Get non-frozen orbital coefficient.

    Parameters
    ----------
    gw : GWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo : double 3d array
        full orbital coefficient

    Returns
    -------
    mo : double 3d darray
        non-frozen orbital coefficient
    """
    return mo[:, get_frozen_mask(gw)]


def set_frozen_orbs(gw):
    """Set orbs and orbs_frz attributes from frozen mask.
    orbs: list of orbital index in all orbitals
    orbs_frz: list of orbital index in non-frozen orbitals

    Parameters
    ----------
    gw : GWAC
        restricted GW object
    """
    if gw.frozen is not None:
        if gw.orbs is not None:
            if isinstance(gw.frozen, (int, np.int64)):
                # frozen core
                gw.orbs_frz = [x - gw.frozen for x in gw.orbs]
            else:
                # frozen list
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs_frz = []
                for orbi in gw.orbs:
                    count = len([p for p in gw.frozen if p <= orbi])
                    gw.orbs_frz.append(orbi - count)
            if any(np.array(gw.orbs_frz) < 0):
                raise RuntimeError('GW orbs must be larger than frozen core!')
        else:
            gw.orbs_frz = range(gw.nmo)
            gw.orbs = range(len(gw._scf.mo_energy))
            if isinstance(gw.frozen, (int, np.int64)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0], (int, np.int64))
                gw.orbs = list(set(gw.orbs) - set(gw.frozen))
    else:
        if gw.orbs is None:
            gw.orbs = gw.orbs_frz = range(len(gw._scf.mo_energy))
        else:
            gw.orbs_frz = gw.orbs
    return

class GWAC(lib.StreamObject):
    def __init__(self, mf, frozen=None, auxbasis=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB
        self.auxbasis = auxbasis  # auxiliary basis set for density fitting

        # options
        self.frozen = frozen  # frozen orbital options
        self.orbs = None  # list of orbital index in full nmo
        self.fullsigma = False  # calculate off-diagonal self-energy
        self.rdm = False  # calculate GW density matrix
        self.vhf_df = False  # use density-fitting for exchange self-energy
        self.fc = False  # finite-size correction (for supercell)
        self.fc_grid = False  # finite-size correction grid (for supercell)
        self.outcore = False  # low-memory routine to calculate self-energy
        self.segsize = 100  # number of orbitals in one segment for outcore
        self.eta = 5.0e-3  # broadening parameter
        self.nw = 100  # number of quadrature points for integration
        self.nw2 = None  # number of points at which to evaluate self-energy
        self.ac = 'pade'  # analytical continuation method
        self.ac_iw_cutoff = 5.0  # imaginary frequency cutting for fitting self-energy
        self.ac_pade_npts = 18  # number of selected points for Pade approximation
        self.ac_pade_step_ratio = 2.0 / 3.0  # final/initial step size for Pade approximation
        self.ac_pes_mmax = 6
        self.ac_pes_maxiter = 200  # max iteration in PES fitting
        self.ac_pes_disp = False
        self.ac_idx = None # indices of iw to use for AC fitting
        self.qpe_max_iter = 100  # max iteration in iteratively solving quasiparticle equation
        self.qpe_tol = 1.0e-6  # tolerance in Newton method for iteratively quasiparticle equation
        self.qpe_linearized = False  # use linearized quasiparticle equation
        self.qpe_linearized_range = [0.5, 1.5]  # Z-shot factor range, if not in this range, z=1
        self.writefile = 0  # write file level

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self.orbs_frz = None  # list of orbital index in non-frozen orbitals
        self.mo_energy = np.array(mf.mo_energy, copy=True)  # quasiparticle energy
        self.mo_coeff = np.array(mf.mo_coeff, copy=True)  # quasiparticle orbtial coefficient
        self.mo_occ = np.array(mf.mo_occ, copy=True)  # quasiparticle orbital occupation
        self.Lpq = None  # three-center density-fitting matrix in MO

        # results
        self.vk = None  # exchange self-energy matrix
        self.vxc = None  # mean-field vxc matrix
        self.freqs = None  # frequency grids, size=nw2
        self.wts = None  # weights of frequency grids, size=nw2
        self.ef = None  # Fermi level
        self.sigmaI = None  # self-energy in the imaginary axis
        self.acobj = None  # analytical continuation object

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('frozen orbitals = %s', self.frozen)
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('broadening parameter = %.3e', self.eta)
        if self.nw2 is None:
            log.info('number of grids = %d', self.nw)
        else:
            log.info('grid size for W is %d', self.nw)
            log.info('grid size for self-energy is %d', self.nw2)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %.1f', self.ac_iw_cutoff)
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('use perturbative linearized QP eqn = %s', self.qpe_linearized)
        if self.qpe_linearized is True:
            log.info('linearized factor range = %s', self.qpe_linearized_range)
        else:
            log.info('QPE max iter = %d', self.qpe_max_iter)
            log.info('QPE tolerance = %.1e', self.qpe_tol)
        log.info('')
        return

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

    def kernel(self):
        """Do one-shot GW calculation using analytical continuation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting.

        Parameters
        ----------
        auxbasis : str, optional
            name of auxiliary basis set, by default None
        """
        if getattr(self._scf, 'with_df', None):
            self.with_df = self._scf.with_df
        else:
            self.with_df = df.DF(self._scf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = df.make_auxbasis(self._scf.mol, mp2fit=False)
        self._keys.update(['with_df'])
        return

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None

        Returns
        -------
        Lpq : double 3d array
            three-center density-fitting matrix in MO
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        naux = self.with_df.get_naoaux()
        nmo_pair = nmo * (nmo + 1) // 2
        nao_pair = nao * (nao + 1) // 2
        mem_incore = (max(nao_pair * naux, nmo**2 * naux) + nmo_pair * naux) * 8 / 1e6
        mem_incore = (2 * nmo**2 * naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < self.max_memory) or self.mol.incore_anyway:
            Lpq = nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux, nmo, nmo)
        else:
            raise MemoryError

    def loop_ao2mo(self, mo_coeff=None, ijslice=None):
        """Transform density-fitting integral from AO to MO by block.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            coefficient from AO to MO, by default None
        ijslice : tuple, optional
            tuples for (1st idx start, 1st idx end, 2nd idx start, 2nd idx end), by default None

        Returns
        -------
        eri_3d : double 3d array
            three-center density-fitting matrix in MO in a block
        """
        nmo = self.nmo
        naux = self.with_df.get_naoaux()

        mo = np.asarray(mo_coeff, order='F')
        if ijslice is None:
            ijslice = (0, nmo, 0, nmo)
        nislice = ijslice[1] - ijslice[0]
        njslice = ijslice[3] - ijslice[2]

        with_df = self.with_df
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now) * 0.3)
        blksize = int(min(naux, max(with_df.blockdim, (max_memory * 1e6 / 8) / (nmo * nmo))))
        eri_3d = []
        for eri1 in with_df.loop(blksize=blksize):
            Lpq = None
            Lpq = nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            eri_3d.append(Lpq)
        del eri1
        del Lpq

        eri_3d = np.vstack(eri_3d).reshape(naux, nislice, njslice)
        return eri_3d

    def get_ef(self, mo_energy=None):
        """Get Fermi level.
        For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
        For metallic systems, Fermi level is optmized according to mo_energy.

        Parameters
        ----------
        mo_energy : double 1d array, optional
            orbital energy, by default None

        Returns
        -------
        ef : double
            Fermi level
        """
        if mo_energy is None:
            mo_energy = self.mo_energy

        if hasattr(self._scf, 'sigma'):
            f_occ = _fermi_smearing_occ if self._scf.smearing_method.lower() == 'fermi' else _gaussian_smearing_occ
            ef = _smearing_optimize(f_occ, mo_energy, self._scf.mol.nelectron / 2, self._scf.sigma)[0]
        else:
            # working with full space mo_energy and nocc here
            nocc = self._scf.mol.nelectron // 2
            if (mo_energy[nocc] - mo_energy[nocc - 1]) < 1e-3:
                logger.warn(self, 'GW not well-defined for degeneracy!')
            ef = (mo_energy[nocc - 1] + mo_energy[nocc]) * 0.5
        return ef

    def energy_tot(self):
        """Compute GW total energy according to Galitskii-Migdal formula.
        See equation 6 in doi.org/10.1103/PhysRevB.86.081102,
        and equation 11 in doi.org/10.1021/acs.jctc.0c01264.
        NOTE: self-energy evaluted by numerical integration at large frequency is difficult to be accurate,
        so this function is numerically unstable.

        Returns
        -------
        E_tot : double
            GW total energy
        E_hf : double
            Hartree-Fock total energy
        Ec : double
            GW correlation energy
        """
        assert self.sigmaI is not None
        assert self.rdm and self.fullsigma
        sigmaI = self.sigmaI[:, :, 1:]
        freqs = 1j * self.freqs
        wts = self.wts
        nmo = self.nmo

        if len(self.orbs) != nmo:
            sigma = np.zeros((nmo, nmo, len(freqs)), dtype=sigmaI.dtype)
            for ia, a in enumerate(self.orbs):
                for ib, b in enumerate(self.orbs):
                    sigma[a, b, :] = sigmaI[ia, ib, :]
        else:
            sigma = sigmaI

        # Compute mean-field Green's function on imag freq
        gf0 = get_g0(freqs, np.array(self._scf.mo_energy) - self.ef, eta=0)

        # Compute GW correlation energy
        g_sigma = 1.0 / 2.0 / np.pi * einsum('ijw,ijw,w->ij', gf0, sigma, wts)
        # factor 2.0 from integration on negative imaginary axis
        Ec = 2.0 * np.trace(g_sigma).real

        # Compute HF energy using DFT density matrix
        # NOTE: this definitation can be wrong
        # TODO: update HF-energy with MRGW functions
        dm = self._scf.make_rdm1()
        rhf = scf.RHF(self.mol)
        E_hf = rhf.energy_elec(dm=dm)[0] + self._scf.energy_nuc()

        E_tot = E_hf + Ec

        return E_tot, E_hf, Ec

    def make_rdm1(self, ao_repr=False, mode='linear'):
        r"""Get GW density matrix from G(it=0).
        G(it=0) = \int G(iw) dw
        As shown in doi.org/10.1021/acs.jctc.0c01264, calculate G0W0 Green's function using Dyson equation is not
        particle number conserving.
        The linear mode G = G0 + G0 Sigma G0 is particle number conserving.

        Parameters
        ----------
        ao_repr : bool, optional
            return dm in AO space instead of MO space, by default False
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'linear'

        Returns
        -------
        rdm1 : double 2d array
            one-particle density matrix
        """

        assert self.sigmaI is not None
        assert self.rdm and self.fullsigma
        assert mode in ['dyson', 'linear']
        sigmaI = self.sigmaI[:, :, 1:]
        freqs = 1j * self.freqs
        wts = self.wts
        nmo = self.nmo
        if len(self.orbs) != nmo:
            sigma = np.zeros((nmo, nmo, len(freqs)), dtype=sigmaI.dtype)
            for ia, a in enumerate(self.orbs):
                for ib, b in enumerate(self.orbs):
                    sigma[a, b, :] = sigmaI[ia, ib, :]
        else:
            sigma = sigmaI

        # Compute GW Green's function on imag freq
        gf0 = get_g0(freqs, np.array(self._scf.mo_energy) - self.ef, eta=0)
        gf = np.zeros_like(gf0)
        if mode == 'linear':
            for iw in range(len(freqs)):
                gf[:, :, iw] = gf0[:, :, iw] + (gf0[:, :, iw] @ (sigma[:, :, iw] + self.vk - self.vxc) @ gf0[:, :, iw])
        elif mode == 'dyson':
            for iw in range(len(freqs)):
                gf[:, :, iw] = np.linalg.inv(np.linalg.inv(gf0[:, :, iw]) - sigma[:, :, iw] - self.vk + self.vxc)

        # GW density matrix
        rdm1 = 2.0 / np.pi * einsum('ijw,w->ij', gf, wts).real + np.eye(nmo)
        # Symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.T)
        logger.info(self, 'GW particle number = %s', np.trace(rdm1))

        if ao_repr is True:
            rdm1 = self._scf.mo_coeff @ rdm1 @ self._scf.mo_coeff.T

        return rdm1

    def make_gf(self, omega, eta=0.0, mode='dyson'):
        """Get G0W0 Green's function by AC fitting.

        Parameters
        ----------
        omega : complex 1d array
            frequency on which to evaluate the Green's function
        eta : double, optional
            broadening parameter. Defaults to 0.
        mode : str, optional
            mode for Dyson equation, 'linear' or 'dyson', by default 'dyson'

        Returns
        -------
        gf : complex 3d array
            GW Green's function
        gf0 : complex 3d array
            non-interacting Green's function
        sigma : complex 3d array
            self-energy
        """
        mo_energy = np.asarray(self._scf.mo_energy)
        gf0 = get_g0(omega, mo_energy, eta)

        gf = np.zeros_like(gf0)
        if self.fullsigma is True:
            sigma = self.acobj.ac_eval(omega + 1j * eta)
            sigma_diff = np.array(sigma, copy=True)
            for iw in range(len(omega)):
                sigma_diff[:, :, iw] += self.vk - self.vxc
        else:
            sigma = np.zeros_like(gf0)
            sigma_diff = np.zeros_like(gf0)
            for iw in range(len(omega)):
                for i in range(len(mo_energy)):
                    sigma[i, i, iw] = self.acobj[i].ac_eval(omega + 1j * eta)
                    sigma_diff[i, i, iw] = sigma[i, i, iw] + self.vk[i, i] - self.vxc[i, i]

        for iw in range(len(omega)):
            if mode == 'linear':
                gf[:, :, iw] = gf0[:, :, iw] + (gf0[:, :, iw] @ sigma_diff[:, :, iw] @ gf0[:, :, iw])
            elif mode == 'dyson':
                gf[:, :, iw] = np.linalg.inv(np.linalg.inv(gf0[:, :, iw]) - sigma_diff[:, :, iw])

        return gf, gf0, sigma

    def get_sigma_exchange(self, mo_coeff):
        """Get exchange self-energy (EXX).

        Parameters
        ----------
        mo_coeff : double 2d array
            orbital coefficient

        Returns
        -------
        vk : double 2d array
            exchange self-energy
        """
        dm = self._scf.make_rdm1()
        # keep this condition for embedding calculations
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = scf.RHF(self.mol)
            if hasattr(self._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=self._scf.sigma, method=self._scf.smearing_method)
        vk_ao = rhf.get_veff(dm=dm) - rhf.get_j(dm=dm)
        vk = mo_coeff.T @ vk_ao @ mo_coeff
        return vk

    def setup_evaluation_grid(self, fallback_freqs=None, fallback_wts=None):
        """Set up self-energy grid, aka freqs2, aka gw.freqs.

        Parameters
        ----------
        fallback_freqs : double 1d array
            These are used as last resort if neither gw.nw2 nor gw.freqs is set.
        fallback_wts : double 1d array
            weights corresponding to fallback_freqs.

        Returns
        -------
        eval_freqs_with_zero : double 1d array
            self.freqs prepended with 0 for convenience.
        """
        if self.freqs is not None:
            pass
        elif self.nw2 is not None:
            assert self.freqs is None and self.wts is None, 'freqs and wts must not be set if nw2 is specified'
            self.freqs, self.wts = _get_scaled_legendre_roots(self.nw2)
        else:
            assert fallback_freqs is not None and fallback_wts is not None, 'freqs and wts must be set'
            self.freqs = fallback_freqs.copy()
            self.wts = fallback_wts.copy()
        assert self.freqs.ndim == 1, 'freqs must be 1D array'
        assert self.wts.shape == self.freqs.shape, 'freqs and wts must have the same shape'

        eval_freqs_with_zero = np.concatenate(([0.0], self.freqs))
        return eval_freqs_with_zero


if __name__ == '__main__':
    from pyscf import gto, dft, scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    # diag self-energy, incore
    gw = GWAC(mf)
    gw.orbs=range(4, 6)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

    # full self-energy, incore
    gw = GWAC(mf)
    gw.fullsigma = True
    gw.orbs=range(4, 6)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

    # diag self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.outcore = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

    # full self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.fullsigma = True
    gw.outcore = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

    # frozen core
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.frozen = 1
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42667346) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16490656) < 1e-5

    # frozen list
    gw = GWAC(mf)
    gw.orbs = [4, 7]
    gw.frozen = [0, 5]
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.43309464) < 1e-5
    assert abs(gw.mo_energy[7] - 0.73675504) < 1e-5

    # get GW density matrix
    gw = GWAC(mf)
    gw.fullsigma = True
    gw.rdm = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5
    print("\nGW density matrix\n", gw.make_rdm1())

    # generate Green's function on real axis and print density of states
    gw = GWAC(mf)
    gw.fullsigma = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5
    omega = np.linspace(-0.5, 0.5, 101)
    gf, gf0, _ = gw.make_gf(omega=omega, eta=0.01, mode='dyson')
    print('\nDOS: KS, GW')
    for iw in range(len(omega)):
        print(omega[iw], -np.trace(gf0[:, :, iw].imag) / np.pi, -np.trace(gf[:, :, iw].imag) / np.pi)

    dos_plot = False
    if dos_plot:
        try:
            import matplotlib.pyplot as plt

            gfomega = np.linspace(-1, 1, 201)
            eta = 0.01
            # gw.vk_minus_vmf = None
            gf, gf0, sigma = gw.make_gf(gfomega, eta)
            dos = -np.trace(gf.imag, axis1=0, axis2=1) / np.pi
            plt.plot(gfomega, dos, label='gf')
            plt.axvline(-0.42667346, color='red', label='reference\nHOMO/LUMO')
            plt.axvline(0.16490657, color='red')
            plt.legend()
            plt.savefig('dos_gwac.png', bbox_inches='tight')
        except ModuleNotFoundError as e:
            print(str(e), 'cannot plot DOS.')

    print('passed tests!')
