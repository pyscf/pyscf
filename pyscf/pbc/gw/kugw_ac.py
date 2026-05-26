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
Periodic spin-unrestricted G0W0 method based on the analytic continuation scheme.
This implementation has N^4 scaling,
and is faster than GW-CD (N^4~N^5) and fully analytic GW (N^6) methods.
GW-AC is recommended for valence states only, and is inaccurate for core states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14 053020 (2012)
'''

from functools import reduce
import h5py
import time
import numpy as np
import scipy

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.lib import einsum, logger, temporary_env
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kump2 import get_frozen_mask

from pyscf.pbc.gw.krgw_ac import KRGWAC
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.gw.utils.gw_np_helper import mkslice, array_scale


def kernel(gw):
    mf = gw._scf
    nmo = gw.nmo[0]
    nkpts = gw.nkpts

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)
    mo_coeff_frz = _mo_frozen(gw, gw.mo_coeff)

    # v_xc
    with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
        dm = mf.make_rdm1()
        v_mf_ao = mf.get_veff()
        vj_ao = mf.get_j(dm_kpts=dm)
    v_mf_ao[0] = v_mf_ao[0] - (vj_ao[0] + vj_ao[1])
    v_mf_ao[1] = v_mf_ao[1] - (vj_ao[0] + vj_ao[1])
    v_mf = np.zeros(shape=[2, nkpts, nmo, nmo], dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            v_mf[s, k] = reduce(np.matmul, (mo_coeff_frz[s, k].T.conj(), v_mf_ao[s, k], mo_coeff_frz[s, k]))
    gw.vxc = v_mf

    # v_hf from DFT/HF density
    if isinstance(mf.with_df, df.GDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).density_fit()
    elif isinstance(mf.with_df, df.RSDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).rs_density_fit()
    if hasattr(mf, 'sigma'):
        uhf = scf.addons.smearing_(uhf, sigma=mf.sigma, method=mf.smearing_method)
    uhf.with_df = gw.with_df
    uhf.verbose = uhf.mol.verbose = 0
    with temporary_env(uhf, verbose=0), temporary_env(uhf.with_df, verbose=0):
        vk_ao = uhf.get_veff(dm_kpts=dm)
        vj_ao = uhf.get_j(dm_kpts=dm)
    vk_ao[0] = vk_ao[0] - (vj_ao[0] + vj_ao[1])
    vk_ao[1] = vk_ao[1] - (vj_ao[0] + vj_ao[1])
    vk = np.zeros(shape=[2, nkpts, nmo, nmo], dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            vk[s, k] = reduce(np.matmul, (mo_coeff_frz[s, k].T.conj(), vk_ao[s, k], mo_coeff_frz[s, k]))

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
        for s in range(2):
            for k in range(nkpts):
                # NOTE: here was a bug in commits before 2024/12
                for i in range(gw.nocc[s]):
                    vk[s][k][i, i] = vk[s][k][i, i] + vk_corr
    gw.vk = vk

    # set up Fermi level
    ef = gw.ef = get_ef(kmf=mf, mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # calculate self-energy on imaginary axis
    sigmaI, omega = get_sigma(
        gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
        fullsigma=gw.fullsigma)

    # analytic continuation
    if gw.ac == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), gw.nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

    acobj.ac_fit(sigmaI, omega, axis=-1)

    if gw.fullsigma:
        diag_acobj = acobj.diagonal(axis1=2, axis2=3)
    else:
        diag_acobj = acobj

    mo_energy = np.zeros_like(mf.mo_energy)
    for s in range(2):
        for ik, k in enumerate(kptlist):
            for ip, p in enumerate(orbs_frz):
                if gw.qpe_linearized:
                    # linearized G0W0
                    de = 1e-6
                    ep = mf.mo_energy[s][k][orbs[ip]]
                    sigmaR = diag_acobj[s, ik, ip].ac_eval(ep).real
                    dsigma = diag_acobj[s, ik, ip].ac_eval(ep + de).real - sigmaR.real
                    zn = 1.0 / (1.0 - dsigma / de)
                    if gw.qpe_linearized_range is not None:
                        zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
                    mo_energy[s, k, orbs[ip]] = ep + zn * (sigmaR + vk[s, k, p, p] - v_mf[s, k, p, p]).real
                else:
                    # self-consistently solve QP equation
                    def quasiparticle(omega):
                        sigmaR = diag_acobj[s, ik, ip].ac_eval(omega)
                        return omega - mf.mo_energy[s][k][orbs[ip]] - (sigmaR + vk[s, k, p, p] - v_mf[s, k, p, p]).real

                    try:
                        mo_energy[s, k, orbs[ip]] = scipy.optimize.newton(
                            quasiparticle, mf.mo_energy[s][k][orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                        )
                    except RuntimeError:
                        logger.warn(gw, 'QPE for spin=%d k=%d orbital=%d not converged!', s, k, orbs[ip])

    # save GW results
    gw.mo_energy = mo_energy
    gw.acobj = acobj

    if gw.verbose >= logger.DEBUG:
        with np.printoptions(threshold=len(mf.mo_energy[0][0])):
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy spin-up @ k%d =\n%s', k, mo_energy[0, k])
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy spin-down @ k%d =\n%s', k, mo_energy[1, k])

    if gw.writefile > 0:
        with h5py.File('vxc.h5', 'w') as feri:
            feri['vk'] = np.asarray(vk)
            feri['v_mf'] = np.asarray(v_mf)

        with h5py.File('sigma_imag.h5', 'w') as feri:
            feri['sigmaI'] = np.asarray(sigmaI)
            feri['omega'] = np.asarray(omega)
            if gw.sigmaI is not None:
                feri['sigmaI_full'] = np.asarray(gw.sigmaI)

        acobj.save('ac_coeff.h5')

    return


def get_rho_response(omega, nocc, mo_energy, Lia, kidx):
    """Get Pi=PV.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    nocc : list of int
        number of occupied orbitals for two spins
    mo_energy : double ndarray
        orbital energy
    Lia : list of complex 4d array
        occupied-virtual block three-center density-fitting matrix in MO
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    nkpts, naux = Lia[0].shape[:2]
    nocc = [Lia[0].shape[2], Lia[1].shape[2]]
    nvir = [Lia[0].shape[3], Lia[1].shape[3]]

    # Compute Pi for kL
    Pi = np.zeros(shape=[naux, naux], dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]

        for s in range(2):
            eia = mo_energy[s, i, :nocc[s], None] - mo_energy[s, a, None, nocc[s]:]
            Lia_i_s = Lia[s][i]
            eia = eia / (omega**2 + eia**2)
            Pia = Lia_i_s * eia

            # Pi += einsum('Pia,Qia->PQ', Pia, Lia.conj())
            scipy.linalg.blas.zgemm(
                alpha=2.0 / nkpts,
                a=Lia_i_s.reshape(naux, nocc[s] * nvir[s]).T,
                b=Pia.reshape(naux, nocc[s] * nvir[s]).T,
                c=Pi.T,
                trans_a=2,
                trans_b=0,
                beta=1.0,
                overwrite_c=True,
            )
            Pia = Lia_i_s = None
    return Pi


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, kidx):
    """Get Pi=PV for metallic systems.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double ndarray
        orbital energy
    mo_occ : double ndarray
        occupation number
    Lpq : complex ndarray
        three-center density-fitting matrix in MO
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    nkpts, nspin, naux, nmo, nmo = Lpq.shape

    # Compute Pi for kL
    Pi = np.zeros(shape=[naux, naux], dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]

        for s in range(nspin):
            eia = mo_energy[s, i, :, None] - mo_energy[s, a, None, :]
            fia = mo_occ[s][i][:, None] - mo_occ[s][a][None, :]
            Lia = np.ascontiguousarray(Lpq[i, s])
            eia = eia * fia / (omega**2 + eia**2)
            Pia = Lia * eia

            # Pi += einsum('Pia, Qia -> PQ', Pia, Lia.conj()) / nkpts
            scipy.linalg.blas.zgemm(
                alpha=1.0 / nkpts,
                a=Lia.reshape(naux, nmo * nmo).T,
                b=Pia.reshape(naux, nmo * nmo).T,
                c=Pi.T,
                trans_a=2,
                trans_b=0,
                beta=1.0,
                overwrite_c=True,
            )
            Pia = Lia = None
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
    qij : list of complex ndarray
        pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

    Returns
    -------
    Pi_00 : complex
        head response function
    """
    nkpts = qij[0].shape[0]
    nocc = [qij[0].shape[1], qij[1].shape[1]]

    Pi_00 = 0j
    for k in range(nkpts):
        for s in range(2):
            eia = mo_energy[s, k, : nocc[s], None] - mo_energy[s, k, None, nocc[s] :]
            eia = eia / (omega**2 + eia**2)
            Pi_00 += 2.0 / nkpts * einsum('ia,ia->', eia, qij[s][k].conj() * qij[s][k])

    return Pi_00


def get_rho_response_wing(omega, mo_energy, Lia, qij):
    """Compute wing (G=P, G'=0) density response function in auxiliary basis at freq iw.
     equation 48 in 10.1021/acs.jctc.0c00704

     Parameters
     ----------
     omega : double
         frequency point
     mo_energy : double ndarray
         orbital energy
     Lia : complex ndarray
         occupied-virtual block three-center density fitting matrix in MO
     qij : list of complex ndarray
         pair density matrix defined as equation 51 in 10.1021/acs.jctc.0c00704

     Returns
     -------
    Pi : complex ndarray
         wing response function
    """
    nkpts, naux = Lia[0].shape[:2]
    nocc = [Lia[0].shape[2], Lia[1].shape[2]]
    nvir = [Lia[0].shape[3], Lia[1].shape[3]]

    Pi = np.zeros(shape=[naux], dtype=np.complex128)
    for k in range(nkpts):
        for s in range(2):
            eia = mo_energy[s, k, :nocc[s], None] - mo_energy[s, k, None, nocc[s]:]
            eia = eia / (omega**2 + eia**2)
            eia_q = eia * qij[s][k].conj()

            Pi += 2.0 / nkpts * np.matmul(Lia[s][k].reshape(naux, nocc[s] * nvir[s]), eia_q.reshape(-1))

    return Pi


def get_qij(gw, q, mo_energy, mo_coeff, uniform_grids=False):
    """Compute pair density matrix in the long-wavelength limit through kp perturbation theory
    qij = 1/Omega * |< psi_{ik} | e^{iqr} | psi_{ak-q} >|^2
    equation 51 in 10.1021/acs.jctc.0c00704
    Ref: Phys. Rev. B 83, 245122 (2011)

    Parameters
    ----------
    gw : KUGWAC
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
    list
        pair density matrix of two spins in the long-wavelength limit
    """
    nocca, noccb = gw.nocc
    nmoa, nmob = gw.nmo
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    kpts = gw.kpts
    nkpts = len(kpts)
    cell = gw.mol

    if uniform_grids:
        with temporary_env(cell, verbose=0):
            mydf = df.FFTDF(cell, kpts=kpts)
            coords = cell.gen_uniform_grids(mydf.mesh)
    else:
        with temporary_env(cell, verbose=0):
            coords, weights = dft.gen_grid.get_becke_grids(cell, level=4)
    ngrid = len(coords)

    qij_a = np.zeros(shape=[nkpts, nocca, nvira], dtype=np.complex128)
    qij_b = np.zeros(shape=[nkpts, noccb, nvirb], dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        ao_p = dft.numint.eval_ao(cell, coords, kpt=kpti, deriv=1)
        ao = ao_p[0]
        ao_grad = ao_p[1:4]
        if uniform_grids:
            ao_ao_grad = einsum('mg,xgn->xmn', ao.T.conj(), ao_grad) * cell.vol / ngrid
        else:
            ao_ao_grad = einsum('g,mg,xgn->xmn', weights, ao.T.conj(), ao_grad)
        q_ao_ao_grad = -1j * einsum('x,xmn->mn', q, ao_ao_grad)
        q_mo_mo_grad_a = reduce(
            np.matmul, (mo_coeff[0, i][:, :nocca].T.conj(), q_ao_ao_grad, mo_coeff[0, i][:, nocca:])
        )
        q_mo_mo_grad_b = reduce(
            np.matmul, (mo_coeff[1, i][:, :noccb].T.conj(), q_ao_ao_grad, mo_coeff[1, i][:, noccb:])
        )
        enm_a = 1.0 / (mo_energy[0, i][nocca:, None] - mo_energy[0, i][None, :nocca])
        enm_b = 1.0 / (mo_energy[1, i][noccb:, None] - mo_energy[1, i][None, :noccb])
        dens_a = enm_a.T * q_mo_mo_grad_a
        dens_b = enm_b.T * q_mo_mo_grad_b
        qij_a[i] = dens_a / np.sqrt(cell.vol)
        qij_b[i] = dens_b / np.sqrt(cell.vol)

    return (qij_a, qij_b)


def get_sigma(
    gw, freqs, wts, ef, mo_energy, orbs=None, kptlist=None, mo_coeff=None, mo_occ=None, iw_cutoff=None, fullsigma=False
):
    """Get GW self-energy.
    See equation 27 in 10.1021/acs.jctc.0c00704

    Parameters
    ----------
    gw : KUGWAC
        GW objects,
        provides attributes: _scf, mol, frozen, nmo, nocc, kpts, nkpts, mo_coeff, mo_occ, fc, fc_grid, with_df
    freqs : double array
        position of imaginary frequency
    wts : double array
        weight of frequency points
    ef : double
        Fermi level
    mo_energy : double ndarray
        non-frozen orbital energy
    orbs : list, optional
        orbital index in non-frozen nmo to calculate self-energy, by default None
    kptlist : list, optional
        k-point index to calculate self-energy, by default None
    mo_coeff : complex ndarray, optional
        coefficient from AO to non-frozen MO, by default None
    mo_occ : double ndarray, optional
        non-frozen occupation number, by default None
    iw_cutoff : complex, optional
        imaginary grid cutoff for fitting, by default None
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False

    Returns
    -------
    sigma: complex ndarray
        self-energy on the imaginary axis
    omega: complex ndarray
        imaginary frequency grids of self-energy
    """
    nocca, noccb = nocc = gw.nocc
    nmoa, nmob = nmo = gw.nmo
    nkpts = gw.nkpts
    kpts = gw.kpts

    assert nmoa == nmob
    if orbs is None:
        orbs = list(range(nmoa))
    if kptlist is None:
        kptlist = list(range(nkpts))
    norbs = len(orbs)
    nklist = len(kptlist)
    nw = len(freqs)

    if mo_coeff is None:
        mo_coeff = _mo_frozen(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_occ_frozen(gw, gw.mo_occ)
    nao = mo_coeff.shape[2]

    # possible kpts shift
    kscaled = gw.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    # Integration on numerical grids
    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(iw < iw_cutoff for iw in freqs) + 1
    else:
        nw_sigma = nw + 1

    omega = np.zeros(shape=[nw_sigma], dtype=np.complex128)
    omega[1:] = 1j * freqs[: (nw_sigma - 1)] + ef
    emo_a = omega[None, None, :] - mo_energy[0][:, :, None]
    emo_b = omega[None, None, :] - mo_energy[1][:, :, None]

    if fullsigma is False:
        sigma = np.zeros(shape=[2, nklist, norbs, nw_sigma], dtype=np.complex128)
    else:
        sigma = np.zeros(shape=[2, nklist, norbs, norbs, nw_sigma], dtype=np.complex128)
    if gw.fc:
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

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij_a = np.zeros(shape=[nq_pts, nkpts, nocca, nmoa - nocca], dtype=np.complex128)
        qij_b = np.zeros(shape=[nq_pts, nkpts, noccb, nmob - noccb], dtype=np.complex128)

        if not gw.fc_grid:
            for k in range(nq_pts):
                qij_tmp = get_qij(gw, q_abs[k], mo_energy, mo_coeff)
                qij_a[k] = qij_tmp[0]
                qij_b[k] = qij_tmp[1]
        else:
            for k in range(nq_pts):
                qij_tmp = get_qij(gw, q_abs[k], mo_energy, mo_coeff)
                qij_a[k] = qij_tmp[0]
                qij_b[k] = qij_tmp[1]

    cderiarr = gw.with_df.cderi_array()
    for kL in range(nkpts):
        # Lij: (ki, 2, L, i, j) for looping every kL
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)
        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(gw, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, i, j))

                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = cderiarr.load(kpti, kptj)
                    if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
                        Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
                    else:
                        Lpq = Lpq.reshape(-1, nao**2)
                    Lpq = Lpq.astype(np.complex128)

                    moija, ijslicea = _conc_mos(mo_coeff[0, i], mo_coeff[0, j])[2:]
                    moijb, ijsliceb = _conc_mos(mo_coeff[1, i], mo_coeff[1, j])[2:]
                    Lij_out_a = None
                    Lij_out_b = None
                    Lij_out_a = _ao2mo.r_e2(Lpq, moija, ijslicea, tao=[], ao_loc=None, out=Lij_out_a)
                    Lij_out_b = _ao2mo.r_e2(Lpq, moijb, ijsliceb, tao=[], ao_loc=None, out=Lij_out_b)
                    Lij.append(np.asarray((Lij_out_a.reshape(-1, nmoa, nmoa), Lij_out_b.reshape(-1, nmob, nmob))))

        Lij = np.ascontiguousarray(Lij)
        naux = Lij.shape[2]
        if hasattr(gw._scf, 'sigma') is False:
            Lia = [
                np.ascontiguousarray(Lij[:, 0, :, : nocc[0], nocc[0] :]),
                np.ascontiguousarray(Lij[:, 1, :, : nocc[1], nocc[1] :]),
            ]

        naux_ones = np.ones(shape=[1, naux], dtype=np.complex128)
        for w in range(nw):
            # body dielectric matrix eps_body
            if hasattr(gw._scf, 'sigma'):
                Pi = get_rho_response_metal(freqs[w], mo_energy, mo_occ, Lij, kidx)
            else:
                Pi = get_rho_response(freqs[w], nocc, mo_energy, Lia, kidx)
            Pi_inv = np.linalg.inv(np.eye(naux) - Pi)

            if gw.fc and kL == 0:
                eps_inv_00 = 0j
                eps_inv_P0 = np.zeros(shape=[naux], dtype=np.complex128)
                for iq in range(nq_pts):
                    # head dielectric matrix eps_00, equation 47 in 10.1021/acs.jctc.0c00704
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, (qij_a[iq], qij_b[iq]))
                    eps_00 = 1.0 - 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00

                    # wings dielectric matrix eps_P0, equation 48 in 10.1021/acs.jctc.0c00704
                    Pi_P0 = get_rho_response_wing(freqs[w], mo_energy, Lia, (qij_a[iq], qij_b[iq]))
                    eps_P0 = -np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                    # inverse dielectric matrix
                    # equation 53 in 10.1021/acs.jctc.0c00704
                    eps_inv_00 += 1.0 / nq_pts * 1.0 / (eps_00 - reduce(np.matmul, (eps_P0.conj(), Pi_inv, eps_P0)))
                    # equation 54 in 10.1021/acs.jctc.0c00704
                    eps_inv_P0 += 1.0 / nq_pts * (-eps_inv_00) * np.matmul(Pi_inv, eps_P0)

                # head correction
                Del_00 = 2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0) * (eps_inv_00 - 1.0)

            Pi_inv -= np.eye(naux)
            g0_a = wts[w] * emo_a / (emo_a**2 + freqs[w] ** 2)
            g0_b = wts[w] * emo_b / (emo_b**2 + freqs[w] ** 2)
            g0 = [g0_a, g0_b]
            for k, kn in enumerate(kptlist):
                # Find km that conserves with kn and kL (-km+kn+kL=G)
                km = kidx_r[kn]

                for s in range(2):
                    # Qmn_a = einsum('Pmn,PQ->Qmn', Lij[km, s][:, :, orbs].conj(), Pi_inv)
                    if len(orbs) == nmo[s]:
                        l_slice = Lij[km, s].reshape(naux, -1)
                    else:
                        l_slice = np.ascontiguousarray(Lij[km, s, :, :, mkslice(orbs)].reshape(naux, -1))
                    Qmn = np.zeros(shape=[nmo[s] * norbs, naux], dtype=np.complex128)
                    scipy.linalg.blas.zgemm(alpha=1.0, a=Pi_inv.T, b=l_slice.T, c=Qmn.T, overwrite_c=1, trans_b=2)
                    Qmn = Qmn.T

                    if fullsigma is False:
                        # Wmn = 1.0 / nkpts * einsum('Qmn,Qmn->mn', Qmn, Lij[km, s][:, :, orbs])
                        Qmn = Qmn * l_slice
                        Wmn = np.matmul(naux_ones, Qmn)
                        array_scale(Wmn, 1.0 / nkpts)

                        # sigma[s, k] += -einsum('mn,mw->nw', Wmn, g0[s][km]) / np.pi
                        sigma[s, k] -= np.matmul(Wmn.reshape(nmo[s], norbs).T, g0[s][km]) / np.pi
                    else:
                        # for orbm in range(nmo):
                        #     Wmn[orbm] = 1./nkpts * np.dot(Qmn[:,orbm,:].transpose(),Lij[km][:,orbm,orbs])
                        Qmn = Qmn.reshape(naux, nmo[s], norbs)
                        Wmn = np.zeros(shape=[nmo[s], norbs, norbs], dtype=np.complex128)
                        for m in range(nmo[s]):
                            np.matmul(Qmn[:, m, :].T, np.ascontiguousarray(Lij[km, s, :, m, mkslice(orbs)]), out=Wmn[m])
                        array_scale(Wmn, 1.0 / nkpts)
                        Wmn = Wmn.reshape(nmo[s], norbs * norbs).T

                        # sigma[s, k] += -einsum('mnl,mw->nlw',Wmn,g0[km])/np.pi
                        sigma[s, k] -= np.matmul(Wmn, g0[s][km]).reshape(norbs, norbs, nw_sigma) / np.pi

                if gw.fc and kL == 0:
                    assert kn == km
                    for s in range(2):
                        if fullsigma is False:
                            # apply head correction
                            sigma[s, k] += -Del_00 * g0[s][kn][orbs] / np.pi

                            # apply wing correction
                            Wn_P0 = einsum('Pnn,P->n', Lij[kn, s], eps_inv_P0)
                            Wn_P0 = Wn_P0[orbs].real * 2.0
                            Del_P0 = (
                                np.sqrt(gw.mol.vol / 4.0 / np.pi**3)
                                * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (2 / 3)
                                * Wn_P0
                            )

                            sigma[s, k] += -einsum('n,nw->nw', Del_P0, g0[s][kn][orbs]) / np.pi
                        else:
                            # head correction
                            tmp = -Del_00 * g0[s][kn][orbs] / np.pi
                            for p in range(norbs):
                                sigma[s, k, p, p, :] += tmp[p, :]
                            # sigma[s, k, np.arange(norbs), np.arange(norbs), :] += tmp

                            # wing correction
                            Wn_P0 = einsum('Pnn,P->n', Lij[kn, s], eps_inv_P0)
                            Wn_P0 = Wn_P0[orbs].real * 2.0
                            Del_P0 = (
                                np.sqrt(gw.mol.vol / 4.0 / np.pi**3)
                                * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (2 / 3)
                                * Wn_P0
                            )
                            tmp = -einsum('n,nw->nw', Del_P0, g0[s][kn][orbs]) / np.pi
                            for p in range(norbs):
                                sigma[s, k, p, p, :] += tmp[p, :]
                            #sigma[s, k, np.arange(norbs), np.arange(norbs), :] += tmp

    if gw.rdm:
        gw.sigmaI = sigma

    return sigma, omega


def get_ef(kmf, mo_energy):
    """Get Fermi level.
    For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
    For metallic systems, Fermi level is optmized according to mo_energy.

    Parameters
    ----------
    kmf : pyscf.pbc.scf.uhf.UHF/pyscf.pbc.dft.uks.UKS
        mean-field object, provides attributes: kpts, sigma, smearing_method
    mo_energy : double array
        orbital energy

    Returns
    -------
    ef : double
        Fermi level
    """
    if hasattr(kmf, "sigma"):
        from pyscf.scf import addons as mol_addons

        if kmf.smearing_method.lower() == "fermi":
            f_occ = mol_addons._fermi_smearing_occ
        else:
            f_occ = mol_addons._gaussian_smearing_occ
        mo_energy_stack_a = np.hstack(mo_energy[0])
        mo_energy_stack_b = np.hstack(mo_energy[1])
        mo_energy_stack = np.append(mo_energy_stack_a, mo_energy_stack_b)
        nelectron = kmf.mol.tot_electrons(len(kmf.kpts))
        ef = mol_addons._smearing_optimize(f_occ, mo_energy_stack, nelectron, kmf.sigma)[0]
    else:
        nkpts = len(kmf.kpts)
        neleca = 0.0
        nelecb = 0.0
        for k in range(nkpts):
            neleca += np.sum(kmf.mo_occ[0][k])
            nelecb += np.sum(kmf.mo_occ[1][k])
        nocca = int(neleca / nkpts)
        noccb = int(nelecb / nkpts)

        homo = -99.0
        lumo = 99.0
        for k in range(len(kmf.kpts)):
            if homo < max(mo_energy[0][k][nocca - 1], mo_energy[1][k][noccb - 1]):
                homo = max(mo_energy[0][k][nocca - 1], mo_energy[1][k][noccb - 1])
            if lumo > min(mo_energy[0][k][nocca], mo_energy[1][k][noccb]):
                lumo = min(mo_energy[0][k][nocca], mo_energy[1][k][noccb])
        ef = (homo + lumo) / 2.0
    return ef


def get_g0_k(omega, mo_energy, eta):
    """Get non-interacting Green's function.

    Parameters
    ----------
    omega : double or complex ndarray
        frequency grids
    mo_energy : double ndarray
        orbital energy
    eta : double
        broadening parameter

    Returns
    -------
    gf0 : complex ndarray
        non-interacting Green's function
    """
    nkpts = len(mo_energy[0])
    nmo = mo_energy[0][0].shape[0]
    nw = len(omega)
    gf0 = np.zeros(shape=[2, nkpts, nmo, nmo, nw], dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            for iw in range(nw):
                gf0[s, k, :, :, iw] = np.diag(1.0 / (omega[iw] + 1j * eta - mo_energy[s][k]))
    return gf0


def make_gf(gw, omega, eta):
    """Get dynamical Green's function and self-energy.

    Parameters
    ----------
    gw : KUGWAC
        GW object, provides attributes: orbs, kptlist, ef, ac_coeff, omega_fit, vk, vxc, _scf.mo_energy
    omega : double or complex array
        frequency grids
    eta : double
        broadening parameter

    Returns
    -------
    gf : complex ndarray
        GW Green's function
    gf0 : complex ndarray
        mean-field Green's function
    sigma : complex ndarray
        GW correlation self-energy
    """
    nmo = gw.nmo[0]

    nomega = len(omega)
    sigma = np.zeros(shape=[2, gw.nkpts, nmo, nmo, nomega], dtype=np.complex128)
    if gw.fullsigma:
        for s in range(2):
            for ik, k in enumerate(gw.kptlist):
                for ip, p in enumerate(gw.orbs_frz):
                    for iq, q in enumerate(gw.orbs_frz):
                        sigma[s, k, p, q] = gw.acobj[s, ik, ip, iq].ac_eval(omega + 1j * eta)
                        sigma[s, k, p, q] += gw.vk[s, k, p, q] - gw.vxc[s, k, p, q]
    else:
        for s in range(2):
            for k, kn in enumerate(gw.kptlist):
                for ip, p in enumerate(gw.orbs_frz):
                    sigma[s, k, p, p] = gw.acobj[s, ik, ip].ac_eval(omega + 1j * eta)
                    sigma[s, kn, p, p] += gw.vk[s, kn, p, p] - gw.vxc[s, kn, p, p]

    gf0 = get_g0_k(omega, gw._scf.mo_energy, eta)
    gf = np.zeros_like(gf0)
    for s in range(2):
        for k in range(gw.nkpts):
            for iw in range(nomega):
                gf[s, k, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[s, k, :, :, iw]) - sigma[s, k, :, :, iw])

    return gf, gf0, sigma


def make_rdm1_linear(gw, ao_repr=False):
    """Get GW density matrix from Green's function G(it=0).
    G is from linear Dyson equation, which conserves particle number
    G = G0 + G0 Sigma G0
    See equation 16 in 10.1021/acs.jctc.0c01264

    Parameters
    ----------
    gw : KUGWAC
        GW object, provides attributes: sigmaI, mol, _scf, freqs, wts, frozen, orbs, fc
    ao_repr : bool, optional
        return density matrix in AO, by default False

    Returns
    -------
    rdm1 : double ndarray
        density matrix
    """
    assert gw.sigmaI is not None
    assert gw.rdm is True and gw.fullsigma is True
    assert gw.frozen is None or gw.frozen == 0
    sigmaI = gw.sigmaI[:, :, :, 1:]
    freqs = 1j * gw.freqs
    wts = gw.wts
    nmo = gw.nmo
    nkpts = gw.nkpts
    if len(gw.orbs) != nmo:
        sigma = np.zeros(shape=[nkpts, nmo, nmo, len(freqs)], dtype=sigmaI.dtype)
        for s in range(2):
            for k in range(nkpts):
                for ia, a in enumerate(gw.orbs):
                    for ib, b in enumerate(gw.orbs):
                        sigma[s, k, a, b, :] = sigmaI[s, k, ia, ib, :]
    else:
        sigma = sigmaI

    for iw in range(len(freqs)):
        sigma[:, :, :, iw] += gw.vk - gw.vxc
    gf0 = get_g0_k(freqs, np.array(gw._scf.mo_energy) - gw.ef, eta=0)
    gf = np.array(gf0, copy=True)
    for s in range(2):
        for k in range(nkpts):
            for iw in range(len(freqs)):
                gf[s, k, :, :, iw] = gf0[s, k, :, :, iw] @ sigma[s, k, :, :, iw] @ gf0[s, k, :, :, iw]

    # GW density matrix
    rdm1 = np.zeros(shape=[2, nkpts, nmo, nmo], dtype=np.double)
    for s in range(2):
        for k in range(nkpts):
            rdm1[s, k] = (1.0 / np.pi) * einsum('ijw, w -> ij', gf[s, k], wts).real + np.eye(nmo) * 0.5
            channel = "spin-up" if s == 0 else "spin-down"
            logger.info(gw, 'GW particle number %s @ k%d = %s', channel, k, np.trace(rdm1[s, k]))

    # Symmetrize density matrix
    for s in range(2):
        for k in range(nkpts):
            rdm1[s, k] = 0.5 * (rdm1[s, k] + rdm1[s, k].T)

    if ao_repr is True:
        ovlp = gw._scf.get_ovlp()
        for s in range(2):
            for k in range(nkpts):
                CS = np.matmul(ovlp, gw._scf.mo_coeff[s, k])
                rdm1[s, k] = reduce(np.matmul, (CS, rdm1[s, k], CS.conj().T))

    return rdm1


def _mo_energy_frozen(gw, mo_energy):
    """Get non-frozen orbital energy.
    Assume nmoa = nmob.

    Parameters
    ----------
    gw : KUGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_energy : double ndarray
        full orbital energy

    Returns
    -------
    mo_energy_frozen : double ndarray
        non-frozen orbital energy
    """
    frozen_mask = get_frozen_mask(gw)
    nmoa, _ = gw.nmo
    nkpts = gw.nkpts
    mo_energy_frozen = np.zeros(shape=[2, nkpts, nmoa], dtype=np.double)
    for s in range(2):
        for k in range(nkpts):
            mo_energy_frozen[s, k] = mo_energy[s][k][frozen_mask[s][k]]
    return mo_energy_frozen


def _mo_frozen(gw, mo):
    """Get non-frozen orbital coefficient.
    Assume nmoa = nmob.

    Parameters
    ----------
    gw : KUGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo : complex ndarray
        full orbital coefficient

    Returns
    -------
    mo_frozen : complex ndarray
        non-frozen orbital coefficient
    """
    frozen_mask = get_frozen_mask(gw)
    nmoa, _ = gw.nmo
    nkpts = gw.nkpts
    nao = mo[0][0].shape[0]
    mo_frozen = np.zeros(shape=[2, nkpts, nao, nmoa], dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            mo_frozen[s, k] = mo[s][k][:, frozen_mask[s][k]]
    return mo_frozen


def _mo_occ_frozen(gw, mo_occ):
    """Get non-frozen occupation number.
    Assume nmoa = nmob.

    Parameters
    ----------
    gw : KUGWAC
        GW object, provides attributes: frozen, nmo, nkpt
    mo_occ : complex ndarray
        full occupation number

    Returns
    -------
    mo_occ_frozen : double ndarray
        non-frozen occupation number
    """
    frozen_mask = get_frozen_mask(gw)
    nmoa, _ = gw.nmo
    nkpts = gw.nkpts
    mo_occ_frozen = np.zeros(shape=[2, nkpts, nmoa], dtype=np.complex128)
    for s in range(2):
        for k in range(nkpts):
            mo_occ_frozen[s, k] = mo_occ[s][k][frozen_mask[s][k]]
    return mo_occ_frozen


def set_frozen_orbs(gw):
    """Set .frozen attribute from frozen mask.

    Parameters
    ----------
    gw : KUGWAC
        unrestricted GW object
    """
    assert gw.nmo[0] == gw.nmo[1], "current implementation requires nmoa = nmob."

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
            gw.orbs = range(len(gw._scf.mo_energy[0][0]))
            if isinstance(gw.frozen, (int, np.int64)):
                gw.orbs = list(set(gw.orbs) - set(range(gw.frozen)))
            else:
                assert isinstance(gw.frozen[0][0], (int, np.int64))
                assert gw.frozen[0] == gw.frozen[1]
                gw.orbs = list(set(gw.orbs) - set(gw.frozen[0]))
    else:
        if gw.orbs is None:
            gw.orbs = range(len(gw._scf.mo_energy[0][0]))
        gw.orbs_frz = gw.orbs
    return


class KUGWAC(KRGWAC):
    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)', nocca, noccb, nvira, nvirb)
        log.info('nkpt = %d', self.nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        if self.kptlist is not None:
            log.info('k-point list = %s', str(self.kptlist))
        if self.orbs is not None:
            log.info('orbital list = %s', str(self.orbs))
        log.info('off-diagonal self-energy = %s', self.fullsigma)
        log.info('GW density matrix = %s', self.rdm)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('finite size corrections = %s', self.fc)
        if self.fc_grid is not None:
            log.info('grids for finite size corrections = %s', self.fc_grid)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('number of grids = %d', self.nw)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
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
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy[0])
        neleca = 0.0
        nelecb = 0.0
        for k in range(nkpts):
            neleca += np.sum(self._scf.mo_occ[0][k][frozen_mask[0][k]])
            nelecb += np.sum(self._scf.mo_occ[1][k][frozen_mask[1][k]])
        neleca = int(neleca / nkpts)
        nelecb = int(nelecb / nkpts)
        return (neleca, nelecb)

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        frozen_mask = get_frozen_mask(self)
        nmoa = len(self._scf.mo_energy[0][0][frozen_mask[0][0]])
        nmob = len(self._scf.mo_energy[1][0][frozen_mask[1][0]])
        return (nmoa, nmob)

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self, orbs=None, kptlist=None):
        """Run a G0W0 calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

        if isinstance(self.frozen, list) and (not isinstance(self.frozen[0], list)):
            # make sure self.frozen is a list of lists if not frozen core
            self.frozen = [self.frozen, self.frozen]
        else:
            assert self.frozen is None or isinstance(self.frozen, (int, np.int64))

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, "sigma"):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        nmoa, _ = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (3 * nkpts * nmoa**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.warn(self, 'GW QP energies may not be sorted from min to max')
        logger.timer(self, 'GW', *cput0)
        return

    set_frozen_orbs = set_frozen_orbs
    make_rdm1 = make_rdm1_linear
    make_gf = make_gf
