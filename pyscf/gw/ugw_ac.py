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
Spin-unrestricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N^5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
'''

from functools import reduce
import h5py
import numpy as np
from scipy.optimize import newton
import scipy.linalg as sla
import time

from pyscf import dft, lib, scf
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.lib import einsum, logger, temporary_env
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.scf.addons import _fermi_smearing_occ, _gaussian_smearing_occ, _smearing_optimize

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.gw.utils.gw_np_helper import mkslice, get_id_minus_pi_inv_minus_id
from pyscf.gw.gw_ac import GWAC


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

    if gw.Lpq is None and gw.outcore is False:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq

    # mean-field exchange-correlation
    v_mf_ao = mf.get_veff()
    vj_ao = mf.get_j()
    v_mf_ao[0] = v_mf_ao[0] - (vj_ao[0] + vj_ao[1])
    v_mf_ao[1] = v_mf_ao[1] - (vj_ao[0] + vj_ao[1])
    v_mf = np.asarray([reduce(np.matmul, (mo_coeff_frz[s].T, v_mf_ao[s], mo_coeff_frz[s])) for s in range(2)])
    gw.vxc = v_mf

    # exchange self-energy
    if gw.vhf_df is True and gw.outcore is False:
        vk = np.asarray([einsum('Lpi,Liq->pq', Lpq[s, :, :, : nocc[s]], Lpq[s, :, : nocc[s], :]) for s in range(2)])
    else:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vk_ao = uhf.get_veff(dm=dm)
        vj_ao = uhf.get_j(dm=dm)
        vk_ao[0] = vk_ao[0] - (vj_ao[0] + vj_ao[1])
        vk_ao[1] = vk_ao[1] - (vj_ao[0] + vj_ao[1])
        vk = np.asarray([reduce(np.matmul, (mo_coeff_frz[s].T, vk_ao[s], mo_coeff_frz[s])) for s in range(2)])
    gw.vk = vk

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    # compute self-energy on imaginary axis
    if gw.outcore:
        sigmaI, omega = get_sigma_outcore(
            gw, orbs_frz, quad_freqs, quad_wts, ef,
            mo_energy=mo_energy_frz, mo_coeff=mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )
    else:
        sigmaI, omega = get_sigma(
            gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy=mo_energy_frz, iw_cutoff=gw.ac_iw_cutoff,
            eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma
        )

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(orbs, nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    acobj.ac_fit(sigmaI, omega, axis=-1)

    # get GW quasiparticle energy
    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=1, axis2=2)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(np.asarray(gw._scf.mo_energy))
    for s in range(2):
        for ip, p in enumerate(orbs_frz):
            if gw.qpe_linearized:
                # linearized G0W0
                de = 1e-6
                sigmaR = diag_acobj[s][ip].ac_eval(gw._scf.mo_energy[s][p]).real
                dsigma = diag_acobj[s][ip].ac_eval(gw._scf.mo_energy[s][p] + de).real - sigmaR.real
                zn = 1.0 / (1.0 - dsigma / de)
                if gw.qpe_linearized_range is not None:
                    zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
                mo_energy[s, orbs[ip]] = gw._scf.mo_energy[s][p] + zn * (sigmaR.real + vk[s, p, p] - v_mf[s, p, p])
            else:
                # self-consistently solve QP equation
                def quasiparticle(omega):
                    sigmaR = diag_acobj[s, ip].ac_eval(omega).real
                    return omega - gw._scf.mo_energy[s][orbs[ip]] - (sigmaR + vk[s, p, p] - v_mf[s, p, p])

                try:
                    mo_energy[s, orbs[ip]] = newton(
                        quasiparticle, gw._scf.mo_energy[s][orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    logger.warn(gw, 'QPE for spin=%d orbital=%d not converged!', s, orbs[ip])

    # save GW results
    gw.acobj = acobj
    gw.mo_energy = mo_energy
    with np.printoptions(threshold=len(mo_energy[0])):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
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


def get_rho_response(omega, mo_energy, Lpqa, Lpqb):
    """Compute density-density response function in auxiliary basis at freq iw.
    See equation 58 in 10.1088/1367-2630/14/5/053020,
    and equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 2d array
        orbital energy
    Lpqa : double 3d array
        occ-vir block of three-center density-fitting matrix of alpha spin
    Lpqb : double 3d array
        occ-vir block of three-center density-fitting matrix of beta spin

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux = Lpqa.shape[0]
    nocc = [Lpqa.shape[1], Lpqb.shape[1]]
    Pi = np.zeros(shape=[naux, naux], dtype=np.double)
    for s in range(2):
        eia = mo_energy[s, : nocc[s], None] - mo_energy[s, None, nocc[s] :]
        Lia = Lpqa if s == 0 else Lpqb

        # factor 2.0 comes from conjugated term
        eia = 2.0 * eia / (omega**2 + eia**2)
        Pia = lib.broadcast_mul(Lia, eia)
        Pi += np.matmul(Pia.reshape(naux, -1), Lia.reshape(naux, -1).T)
        del eia, Pia
    return Pi


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpqa, Lpqb):
    """Get response function in auxiliary basis for metallic systems.

    Parameters
    ----------
    omega : double
        imaginary part of a frequency point
    mo_energy : double 2d array
        orbital energy
    mo_occ : double 2d array
        occupation number
    Lpqa : double 3d array
        three-center density-fitting matrix of alpha spin
    Lpqb : double 3d array
        three-center density-fitting matrix of beta spin

    Returns
    -------
    Pi : double 2d array
        density-density response function in auxiliary basis at freq iw
    """
    naux = Lpqa.shape[0]
    Pi = np.zeros(shape=[naux, naux], dtype=np.double)
    for s in range(2):
        eia = mo_energy[s, :, None] - mo_energy[s, None, :]
        fia = mo_occ[s, :, None] - mo_occ[s, None, :]
        Lia = Lpqa if s == 0 else Lpqb

        # factor 2.0 comes from conjugated term
        # both ia and ai are included, this gives a factor of 2.0
        eia = eia * fia / (omega**2 + eia**2)
        Pia = lib.broadcast_mul(Lia, eia)
        Pi += np.matmul(Pia.reshape(naux, -1), Lia.reshape(naux, -1).T)
        del eia, Pia
    return Pi


def get_sigma(gw, orbs, Lpq, quad_freqs, quad_wts,ef, mo_energy,
              mo_occ=None, iw_cutoff=None, eval_freqs=None,
              mo_energy_w=None, fullsigma=False):
    """Compute GW correlation self-energy on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : UGWAC
        GW object
    orbs : list
        list of orbital indexes
    Lpq : double 4d array
        three-center density-fitting matrix in MO space
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 2d array
        orbital energy in G
    mo_occ : double 2d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        position of imaginary frequency grids to be integrated, by default None
    mo_energy_w : double 2d array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 3d or 4d array
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nquadfreqs = len(quad_freqs)
    nevalfreqs = len(eval_freqs)
    _, naux, nmo, _ = Lpq.shape
    norbs = len(orbs)

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy

    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    # integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(eval_freqs < iw_cutoff)
    else:
        nw_sigma = nevalfreqs
    nw_cutoff = nw_sigma if iw_cutoff is None else sum(eval_freqs < iw_cutoff)

    omega = ef + 1j * eval_freqs[:nw_sigma]
    emo = omega[None, None, :] - mo_energy_g[:, :, None]

    if fullsigma is False:
        sigma_imag = np.zeros(shape=[2, nw_sigma, norbs], dtype=np.double)
        sigma_real = np.zeros(shape=[2, nw_sigma, norbs], dtype=np.double)
    else:
        sigma_imag = np.zeros(shape=[2, nw_sigma, norbs, norbs], dtype=np.double)
        sigma_real = np.zeros(shape=[2, nw_sigma, norbs, norbs], dtype=np.double)

    # make density-fitting matrix for contractions
    if hasattr(gw._scf, 'sigma') is False:
        Lia_a = np.ascontiguousarray(Lpq[0, :, : nocc[0], nocc[0] :])
        Lia_b = np.ascontiguousarray(Lpq[1, :, : nocc[1], nocc[1] :])
    # assume Lpq = Lpq, so we don't generate Lpq[:, :, mkslice(orbs), :]
    l_slice = Lpq[:, :, :, mkslice(orbs)].reshape(2, naux, -1)

    # self-energy is calculated as equation 27 in doi.org/10.1021/acs.jctc.0c00704
    logger.info(gw, 'Starting get_sigma_diag main loop with %d frequency points.', nquadfreqs)
    Pi = None
    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = np.ones((1, naux))
    else:
        Qmn = np.zeros(shape=[naux, norbs], dtype=np.double)
        Wmn = np.zeros(shape=[nmo, norbs, norbs], dtype=np.double)
    for w in range(nquadfreqs):
        if gw.verbose >= 4:
            gw.stdout.write('%4d ' % (w + 1))
            if w % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        #  Pi_inv = (I - Pi)^-1 - I.
        if hasattr(gw._scf, 'sigma'):
            Pi = get_rho_response_metal(quad_freqs[w], mo_energy, mo_occ, Lpq[0], Lpq[1])
        else:
            Pi = get_rho_response(quad_freqs[w], mo_energy_w, Lia_a, Lia_b)
        Pi_inv = get_id_minus_pi_inv_minus_id(Pi, overwrite_input=True)

        for s in range(2):
            # second line in equation 27
            # second line in equation 27
            g0 = quad_wts[w] * emo[s] / (emo[s] ** 2 + quad_freqs[w] ** 2)

            # split g0 into real and imag parts to avoid costly type conversions
            g0r = np.ascontiguousarray(g0.real)
            g0i = np.ascontiguousarray(g0.imag)

            if fullsigma is False:
                # n is the index of orbitals in orbs, m is the index of orbitals of nmo
                # last line of equation 27, contraction from left to right
                # Qmn = \sum_P V^{nm}_P (Pi_inv)_{PQ} = \sum_P V^{mn}_P (Pi_inv)_{PQ}
                Qmn = np.matmul(Pi_inv.T, l_slice[s], out=Qmn)

                # Qmn = Qmn v^{mn}_Q
                Qmn *= l_slice[s]

                # Wmn = \sum_Q Qmn
                Wmn = np.matmul(naux_ones, Qmn, out=Wmn)

                # sigma -= einsum('mn,mw->wn', Wmn, g0) / np.pi
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(nmo, norbs).T,
                    b=g0r.T,
                    c=sigma_real[s].T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(nmo, norbs).T,
                    b=g0i.T,
                    c=sigma_imag[s].T,
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
                    np.matmul(Pi_inv, l_slice[s].reshape(naux, nmo, norbs)[:, orbm, :], out=Qmn)

                    # Wmn is actually Wmnn'
                    # Wmnn' = \sum_Q Qmn v^{mn'}_Q
                    np.matmul(Qmn.T, l_slice[s].reshape(naux, nmo, norbs)[:, orbm, :], out=Wmn[orbm])

                # sigma -= einsum('mnl,mw->wnl', Wmn, g0)/np.pi
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(nmo, norbs * norbs).T,
                    b=g0r.T,
                    c=sigma_real[s].reshape(nw_sigma, norbs * norbs).T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )
                sla.blas.dgemm(
                    alpha=-1.0 / np.pi,
                    a=Wmn.reshape(nmo, norbs * norbs).T,
                    b=g0i.T,
                    c=sigma_imag[s].reshape(nw_sigma, norbs * norbs).T,
                    trans_a=0,
                    trans_b=1,
                    beta=1.0,
                    overwrite_c=True,
                )

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = np.ascontiguousarray(sigma.transpose(0, 2, 1))
    else:
        sigma = np.ascontiguousarray(sigma.transpose(0, 2, 3, 1))

    logger.info(gw, '\nFinished get_sigma_diag main loop.')

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_sigma_outcore(gw, orbs, quad_freqs, quad_wts, ef, mo_energy, mo_coeff,
                      mo_occ=None, iw_cutoff=None, eval_freqs=None,
                      mo_energy_w=None, fullsigma=False):
    """Low-memory routine to compute GW correlation self-energy (diagonal elements) MO basis on imaginary axis.
    See equation 62 and 62 in 10.1088/1367-2630/14/5/053020,
    and equation 27 in doi.org/10.1021/acs.jctc.0c00704.

    Parameters
    ----------
    gw : UGWAC
        GW object
    orbs : list
        list of orbital indexes
    quad_freqs : double 1d array
        position of imaginary frequency grids used for integration
    quad_wts : double 1d array
        weight of imaginary frequency grids
    ef : double
        Fermi level
    mo_energy : double 2d array
        orbital energy in G
    mo_coeff : double 3d array
        coefficient from AO to MO
    mo_occ : double 2d array, optional
        occupation number, by default None
    iw_cutoff : double, optional
        imaginary grid cutoff for fitting, by default None
    eval_freqs : double 1d array, optional
        position of imaginary frequency grids to be integrated, by default None
    mo_energy_w : double 2d array, optional
        orbital energy in W, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma : complex 3d or 4d array
        self-energy on the imaginary axis
    omega : complex 1d array
        imaginary frequency grids of self-energy
    """
    if eval_freqs is None:
        eval_freqs = quad_freqs

    nocc = gw.nocc
    nmo = gw.nmo[0]
    nquadfreqs = len(quad_freqs)
    nevalfreqs = len(eval_freqs)
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
        nw_sigma = nevalfreqs
    nw_cutoff = nw_sigma if iw_cutoff is None else sum(eval_freqs < iw_cutoff)

    omega = ef + 1j * eval_freqs[:nw_sigma]

    Pi = np.zeros(shape=[nquadfreqs, naux, naux], dtype=np.double)
    for s in range(2):
        if hasattr(gw._scf, 'sigma'):
            nseg = nmo // gw.segsize + 1
        else:
            nseg = nocc[s] // gw.segsize + 1
        for i in range(nseg):
            if hasattr(gw._scf, 'sigma'):
                orb_start = i * gw.segsize
                orb_end = min((i + 1) * gw.segsize, nmo)
                ijslice = (orb_start, orb_end, 0, nmo)
            else:
                orb_start = i * gw.segsize
                orb_end = min((i + 1) * gw.segsize, nocc[s])
                ijslice = (orb_start, orb_end, nocc[s], nmo)
            if s == 0:
                Lia = gw.loop_ao2mo(mo_coeff=mo_coeff, spin='a', ijslicea=ijslice)
            else:
                Lia = gw.loop_ao2mo(mo_coeff=mo_coeff, spin='b', ijsliceb=ijslice)

            for w in range(nquadfreqs):
                if hasattr(gw._scf, 'sigma'):
                    eia = mo_energy_w[s, orb_start:orb_end, None] - mo_energy_w[s, None, :]
                    fia = mo_occ[s, orb_start:orb_end, None] - mo_occ[s, None, :]
                    eia = eia * fia / (quad_freqs[w] ** 2 + eia**2)
                    Pia = lib.broadcast_mul(Lia, eia)
                else:
                    eia = mo_energy_w[s, orb_start:orb_end, None] - mo_energy_w[s, None, nocc[s] :]
                    eia = 2.0 * eia / (quad_freqs[w] ** 2 + eia**2)
                    Pia = lib.broadcast_mul(Lia, eia)

                # This line computes Pi[:,:,w] += einsum('Pia, Qia -> PQ', Pia_a, Lpqa)
                Pi[w] += np.matmul(Pia.reshape(naux, -1), Lia.reshape(naux, -1).T)
                del eia, Pia
            del Lia

    for w in range(nquadfreqs):
        Pi[w] = np.linalg.inv(np.eye(naux) - Pi[w]) - np.eye(naux)
    Pi_inv = Pi

    if fullsigma is False:
        sigma_imag = np.zeros(shape=[2, nw_sigma, norbs], dtype=np.double)
        sigma_real = np.zeros(shape=[2, nw_sigma, norbs], dtype=np.double)
    else:
        sigma_imag = np.zeros(shape=[2, nw_sigma, norbs, norbs], dtype=np.double)
        sigma_real = np.zeros(shape=[2, nw_sigma, norbs, norbs], dtype=np.double)

    if fullsigma is False:
        Qmn = None
        Wmn = None
        naux_ones = np.ones((1, naux))

    nseg = nmo // gw.segsize + 1
    for s in range(2):
        logger.info(gw, 'Starting spin %d get_sigma_diag_outcore main loop with %d segments.', s, nseg)
        emo = omega[None, :] - mo_energy_g[s, :, None]
        for i in range(nseg):
            if gw.verbose >= 4:
                gw.stdout.write('%4d ' % (i + 1))
                if i % 12 == 11:
                    gw.stdout.write('\n')
                gw.stdout.flush()

            orb_start = i * gw.segsize
            orb_end = min((i + 1) * gw.segsize, nmo)
            ijslice = (orb_start, orb_end, 0, nmo)
            if s == 0:
                Lpq = gw.loop_ao2mo(mo_coeff=mo_coeff, spin='a', ijslicea=ijslice)
            else:
                Lpq = gw.loop_ao2mo(mo_coeff=mo_coeff, spin='b', ijsliceb=ijslice)
            l_slice = np.ascontiguousarray(Lpq[:, :, mkslice(orbs)].reshape(naux, -1))
            del Lpq

            for w in range(nquadfreqs):
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
                        c=sigma_real[s].T,
                        trans_a=0,
                        trans_b=1,
                        beta=1.0,
                        overwrite_c=True,
                    )
                    sla.blas.dgemm(
                        alpha=-1.0 / np.pi,
                        a=Wmn.reshape(orb_end - orb_start, norbs).T,
                        b=g0i.T,
                        c=sigma_imag[s].T,
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
                        c=sigma_real[s].reshape(nw_sigma, norbs * norbs).T,
                        trans_a=0,
                        trans_b=1,
                        beta=1.0,
                        overwrite_c=True,
                    )
                    sla.blas.dgemm(
                        alpha=-1.0 / np.pi,
                        a=Wmn.reshape(orb_end - orb_start, norbs * norbs).T,
                        b=g0i.T,
                        c=sigma_imag[s].reshape(nw_sigma, norbs * norbs).T,
                        trans_a=0,
                        trans_b=1,
                        beta=1.0,
                        overwrite_c=True,
                    )
        logger.info(gw, '\nFinished spin %d get_sigma_diag_outcore main loop.', s)

    sigma = sigma_real + 1.0j * sigma_imag
    if fullsigma is False:
        sigma = np.ascontiguousarray(sigma.transpose(0, 2, 1))
    else:
        sigma = np.ascontiguousarray(sigma.transpose(0, 2, 3, 1))

    if gw.rdm is True:
        gw.sigmaI = sigma

    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_g0(omega, mo_energy, eta):
    """Get non-interacting Green's function.

    Parameters
    ----------
    omega : double or complex array
        frequency grids
    mo_energy : double 2d array
        orbital energy
    eta : double
        broadening parameter

    Returns
    -------
    gf0 : complex 4d array
        non-interacting Green's function
    """
    nmo = len(mo_energy[0])
    nw = len(omega)
    gf0 = np.zeros(shape=[2, nmo, nmo, nw], dtype=np.complex128)
    for s in range(2):
        for iw in range(nw):
            gf0[s, :, :, iw] = np.diag(1.0 / (omega[iw] + 1j * eta - mo_energy[s]))
    return gf0


def _mo_energy_without_core(gw, mo_energy):
    """Get non-frozen orbital energy.

    Parameters
    ----------
    gw : UGWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo_energy : double 2d array
        full orbital energy

    Returns
    -------
    mo_energy : double 2d array
        non-frozen orbital energy
    """
    moidx = get_frozen_mask(gw)
    mo_energy = (mo_energy[0][moidx[0]], mo_energy[1][moidx[1]])
    return np.asarray(mo_energy)


def _mo_without_core(gw, mo):
    """Get non-frozen orbital coefficient.

    Parameters
    ----------
    gw : UGWAC
        GW object, provides attributes: frozen, mo_occ, _nmo
    mo : double 3d array
        full orbital coefficient

    Returns
    -------
    mo : double 3d array
        non-frozen orbital coefficient
    """
    moidx = get_frozen_mask(gw)
    mo = (mo[0][:, moidx[0]], mo[1][:, moidx[1]])
    return np.asarray(mo)


def set_frozen_orbs(gw):
    """Set .frozen attribute from frozen mask.
    orbs: list of orbital index in all orbitals
    orbs_frz: list of orbital index in non-frozen orbitals

    Parameters
    ----------
    gw : UGWAC
        unrestricted GW object
    """
    assert gw.nmo[0] == gw.nmo[1], 'current implementation requires nmoa = nmob.'

    if gw.frozen is not None:
        if gw.orbs is not None:
            if isinstance(gw.frozen, (int, np.int64)):
                # frozen core
                gw.orbs_frz = [x - gw.frozen for x in gw.orbs]
            else:
                # frozen list
                assert isinstance(gw.frozen[0][0], (int, np.int64))
                assert gw.frozen[0] == gw.frozen[1]
                gw.orbs_frz = []
                for orbi in gw.orbs:
                    count = len([p for p in gw.frozen[0] if p <= orbi])
                    gw.orbs_frz.append(orbi - count)
            if any(np.array(gw.orbs_frz) < 0):
                raise RuntimeError('GW orbs must be larger than frozen core!')
        else:
            gw.orbs_frz = range(gw.nmo[0])
            gw.orbs = range(len(gw._scf.mo_energy[0]))
            if isinstance(gw.frozen, (int, np.int64)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0][0], (int, np.int64))
                assert gw.frozen[0] == gw.frozen[1]
                gw.orbs = list(set(gw.orbs) - set(gw.frozen[0]))
    else:
        if gw.orbs is None:
            gw.orbs = range(len(gw._scf.mo_energy[0]))
        gw.orbs_frz = gw.orbs
    return


class UGWAC(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
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
            log.info('number of grids for integration= %d', self.nw)
            log.info('number of grids to be integrated = %d', self.nw2)
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

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self):
        """Do one-shot spin-unrestricted GW calculation using analytical continuation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        if isinstance(self.frozen, list) and (not isinstance(self.frozen[0], list)):
            # make sure self.frozen is a list of lists if not frozen core
            self.frozen = [self.frozen, self.frozen]
        else:
            assert self.frozen is None or isinstance(self.frozen, (int, np.int64))

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integral from AO to MO.

        Parameters
        ----------
        mo_coeff : double 3d array, optional
            coefficient from AO to MO, by default None

        Returns
        -------
        Lpq : double 4d array
            three-center density-fitting matrix in MO
        """
        nmoa, nmob = self.nmo
        nao = self.mo_coeff[0].shape[0]
        naux = self.with_df.get_naoaux()
        mem_incore = (nmoa**2 * naux + nmob**2 * naux + nao**2 * naux) * 8 / 1e6
        mem_now = lib.current_memory()[0]

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        ijslicea = (0, nmoa, 0, nmoa)
        ijsliceb = (0, nmob, 0, nmob)
        if (mem_incore + mem_now < 0.99 * self.max_memory) or self.mol.incore_anyway:
            Lpqa = nr_e2(self.with_df._cderi, moa, ijslicea, aosym='s2')
            Lpqb = nr_e2(self.with_df._cderi, mob, ijsliceb, aosym='s2')
            return np.asarray((Lpqa.reshape(naux, nmoa, nmoa), Lpqb.reshape(naux, nmob, nmob)))
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

    def loop_ao2mo(self, mo_coeff=None, spin=None, ijslicea=None, ijsliceb=None):
        """Transform density-fitting integral from AO to MO by block.

        Parameters
        ----------
        mo_coeff : double 3d array, optional
            coefficient from AO to MO, by default None
        spin : str, optional
            spin channel, by default None
        ijslicea : tuple, optional
            tuples for (1st idx start, 1st idx end, 2nd idx start, 2nd idx end) of alpha spin, by default None
        ijsliceb : tuple, optional
            tuples for (1st idx start, 1st idx end, 2nd idx start, 2nd idx end) of beta spin, by default None

        Returns
        -------
        eri_3d : double 3d or 4d array
            three-center density-fitting matrix in MO in a block
        """
        nmoa, nmob = self.nmo
        naux = self.with_df.get_naoaux()

        moa = np.asarray(mo_coeff[0], order='F')
        mob = np.asarray(mo_coeff[1], order='F')
        if ijslicea is None:
            ijslicea = (0, nmoa, 0, nmoa)
        if ijsliceb is None:
            ijsliceb = (0, nmob, 0, nmob)
        nislicea = ijslicea[1] - ijslicea[0]
        njslicea = ijslicea[3] - ijslicea[2]
        nisliceb = ijsliceb[1] - ijsliceb[0]
        njsliceb = ijsliceb[3] - ijsliceb[2]

        with_df = self.with_df
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, (self.max_memory - mem_now) * 0.3)
        blksize = int(min(naux, max(with_df.blockdim, (max_memory * 1e6 / 8) / (nmoa * nmoa))))

        if spin is None or spin == 'a':
            eri_3d_a = []
            for eri1 in with_df.loop(blksize=blksize):
                Lpqa = None
                Lpqa = nr_e2(eri1, moa, ijslicea, aosym='s2', out=Lpqa)
                eri_3d_a.append(Lpqa)
            del eri1
            del Lpqa
            eri_3d_a = np.vstack(eri_3d_a).reshape(naux, nislicea, njslicea)

        if spin is None or spin == 'b':
            eri_3d_b = []
            for eri1 in with_df.loop(blksize=blksize):
                Lpqb = None
                Lpqb = nr_e2(eri1, mob, ijsliceb, aosym='s2', out=Lpqb)
                eri_3d_b.append(Lpqb)
            del eri1
            del Lpqb
            eri_3d_b = np.vstack(eri_3d_b).reshape(naux, nisliceb, njsliceb)

        if spin is None:
            return (eri_3d_a, eri_3d_b)
        elif spin == 'a':
            return eri_3d_a
        elif spin == 'b':
            return eri_3d_b
        else:
            raise ValueError('Wrong spin keyword!')

    def get_ef(self, mo_energy=None):
        """Get Fermi level.
        For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
        For metallic systems, Fermi level is optmized according to mo_energy.

        Parameters
        ----------
        mo_energy : double 2d array, optional
            orbital energy, by default None

        Returns
        -------
        ef : double
            Fermi level
        """
        if mo_energy is None:
            mo_energy = np.asarray(self.mo_energy)

        if hasattr(self._scf, 'sigma'):
            f_occ = _fermi_smearing_occ if self._scf.smearing_method.lower() == 'fermi' else _gaussian_smearing_occ
            ef = _smearing_optimize(f_occ, np.hstack(mo_energy), self._scf.mol.nelectron, self._scf.sigma)[0][0]

        else:
            # working with full space mo_energy and nocc here
            nocca = int(np.sum(self._scf.mo_occ[0]))
            noccb = int(np.sum(self._scf.mo_occ[1]))
            homo = max(mo_energy[0][nocca - 1], mo_energy[1][noccb - 1])
            lumo = min(mo_energy[0][nocca], mo_energy[1][noccb])
            if (lumo - homo) < 1e-3:
                logger.warn(self, 'GW not well-defined for degeneracy!')
            ef = (homo + lumo) * 0.5
        return ef

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
        rdm1 : double 3d array
            one-particle density matrix
        """
        assert self.sigmaI is not None
        assert self.rdm and self.fullsigma
        assert mode in ['dyson', 'linear']
        sigmaI = self.sigmaI[..., 1:]
        freqs = 1j * self.freqs
        wts = self.wts
        nmo = self.nmo[0]

        if len(self.orbs) != nmo:
            sigma = np.zeros(shape=[2, nmo, nmo, len(freqs)], dtype=sigmaI.dtype)
            for s in range(2):
                for ia, a in enumerate(self.orbs):
                    for ib, b in enumerate(self.orbs):
                        sigma[a, b, :] = sigmaI[ia, ib, :]
        else:
            sigma = sigmaI

        # compute GW Green's function on imag freq
        gf0 = get_g0(omega=freqs + self.ef, mo_energy=self._scf.mo_energy, eta=0)
        gf = np.zeros_like(gf0)
        for s in range(2):
            for iw in range(len(freqs)):
                if mode == 'linear':
                    gf[s, :, :, iw] = gf0[s, :, :, iw] + reduce(
                        np.matmul, (gf0[s, :, :, iw], self.vk[s] + sigma[s, :, :, iw] - self.vxc[s], gf0[s, :, :, iw])
                    )
                elif mode == 'dyson':
                    gf[s, :, :, iw] = np.linalg.inv(
                        np.linalg.inv(gf0[s, :, :, iw]) - (self.vk[s] + sigma[s, :, :, iw] - self.vxc[s])
                    )

        # GW density matrix
        rdm1 = 1.0 / np.pi * einsum('sijw,w->sij', gf, wts).real + np.eye(nmo) * 0.5
        # symmetrize density matrix
        rdm1 = 0.5 * (rdm1 + rdm1.transpose((0, 2, 1)))
        nelec = np.trace(rdm1, axis1=1, axis2=2)
        logger.info(self, 'GW particle number up = %s, dn = %s, total = %s', nelec[0], nelec[1], nelec[0] + nelec[1])

        if ao_repr is True:
            for s in range(2):
                rdm1[s] = reduce(np.matmul, (self._scf.mo_coeff[s], rdm1[s], self._scf.mo_coeff[s].T))

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
        gf : complex 4d array
            GW Green's function
        gf0 : complex 4d array
            non-interacting Green's function
        sigma : complex 4d array
            self-energy
        """
        mo_energy = np.asarray(self._scf.mo_energy)
        gf0 = get_g0(omega, mo_energy, eta)

        gf = np.zeros_like(gf0)
        if self.fullsigma is True:
            sigma = self.acobj.ac_eval(omega + 1j * eta)
            sigma_diff = np.array(sigma, copy=True)
            for iw in range(len(omega)):
                sigma_diff[..., iw] += self.vk - self.vxc
        else:
            sigma = np.zeros_like(gf0)
            sigma_diff = np.zeros_like(gf0)
            for s in range(2):
                for iw in range(len(omega)):
                    for i in range(len(mo_energy)):
                        sigma[s, i, i, iw] = self.acobj[s, i].ac_eval(omega + 1j * eta)
                        sigma_diff[s, i, i, iw] = sigma[s, i, i, iw] + self.vk[s, i, i] - self.vxc[s, i, i]

        for s in range(2):
            for iw in range(len(omega)):
                if mode == 'linear':
                    gf[s, :, :, iw] = gf0[s, :, :, iw] + reduce(
                        np.matmul, (gf0[s, :, :, iw], sigma_diff[s, :, :, iw], gf0[s, :, :, iw])
                    )
                elif mode == 'dyson':
                    gf[s, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[s, :, :, iw]) - sigma_diff[s, :, :, iw])

        return gf, gf0, sigma
