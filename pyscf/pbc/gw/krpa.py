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
# Author: Chaoqun Zhang <cq_zhang@outlook.com>
# Author: Jincheng Yu <pimetamon@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Periodic spin-restricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import numpy as np
import scipy.linalg.blas as blas
import time

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.ao2mo._ao2mo import r_e2
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import scf, tools
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.pbc.gw.krgw_ac import get_rho_response, get_rho_response_head, get_rho_response_wing, get_qij, \
    _mo_occ_frozen, _mo_energy_frozen, _mo_frozen


einsum = lib.einsum


def kernel(rpa, mo_energy, mo_coeff, nw=None, with_e_hf=None):
    """RPA correlation and total energy

    Parameters
    ----------
    rpa : KRPA
        rpa object
    mo_energy : double array
        molecular orbital energies
    mo_coeff : double ndarray
        molecular orbital coefficients
    Lpq : double array, optional
        density fitting 3-center integral in MO basis, by default None
    nw : int, optional
        number of frequency point on imaginary axis, by default None
    with_e_hf : float, optional
        extra input HF energy, by default None

    Returns
    -------
    e_tot : float
        RPA total energy
    e_hf : float
        HF energy (exact exchange for given mo_coeff)
    e_corr : float
        RPA correlation energy
    """
    # Compute HF exchange energy (EXX)
    if with_e_hf is None:
        rhf = scf.KRHF(rpa.mol, rpa.kpts, exxdiv=rpa._scf.exxdiv)
        rhf.verbose = 0
        if hasattr(rpa._scf, 'sigma'):
            rhf = scf.addons.smearing_(rhf, sigma=rpa._scf.sigma, method=rpa._scf.smearing_method)
        rhf.with_df = rpa._scf.with_df
        with temporary_env(rpa.with_df, verbose=0), temporary_env(rhf.mol, verbose=0):
            dm = rpa._scf.make_rdm1()
            e_1e = 1.0 / len(rpa.kpts) * lib.einsum('kij,kji', dm, rhf.get_hcore()).real
            e_j = 0.5 / len(rpa.kpts) * lib.einsum('kij,kji', dm, rhf.get_j(rhf.cell, dm)).real
            e_x = get_rpa_exx(rpa, acfd=rpa.acfd_exx, correction_only=False)
            e_nuc = rpa._scf.energy_nuc()
            e_hf = e_1e + e_j + e_x + e_nuc
    else:
        e_hf = with_e_hf
        logger.debug(rpa, f'  Setting EXX energy explicitly to {e_hf}')

    is_metal = hasattr(rpa._scf, 'sigma')

    # Turn off FC for metals
    if is_metal and rpa.fc:
        logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
        rpa.fc = False

    # Grids for integration on imaginary axis
    freqs, wts = rpa.get_grids(nw=nw, mo_energy=mo_energy)

    # Compute RPA correlation energy
    if rpa.outcore:
        if is_metal:
            e_corr = get_rpa_ecorr_outcore_metal(rpa, freqs, wts)
        else:
            e_corr = get_rpa_ecorr_outcore(rpa, freqs, wts)
    else:
        e_corr = get_rpa_ecorr(rpa, freqs, wts)

    # Compute total energy
    e_tot = e_hf + e_corr

    logger.debug(rpa, f'  RPA total energy = {e_tot}')
    logger.debug(rpa, f'  EXX energy = {e_hf}, RPA corr energy = {e_corr}')

    return e_tot, e_hf, e_corr


def get_idx_metal(mo_occ, threshold=1.0e-6):
    """Get index of occupied/virtual/fractional orbitals of metals.

    Parameters
    ----------
    mo_occ : double 1d array
        occupation number
    threshold : double, optional
        threshold to determine fractionally occupied orbitals, by default 1.0e-6

    Returns
    -------
    idx_occ : list
        list of occupied orbital indexes
    idx_frac : list
        list of fractionally occupied orbital indexes
    idx_vir : list
        list of virtual orbital indexes
    """
    idx_occ = np.where(mo_occ > 2.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    idx_frac = list(range(idx_occ[-1] + 1, idx_vir[0]))

    return idx_occ, idx_frac, idx_vir


def get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, kidx):
    """Get Pi=PV for metallic systems.
    P is density-density response function.
    V is two-electron integral.
    See equation 24 in doi.org/10.1021/acs.jctc.0c00704.

    NOTE: this function is different from the one in krgw_ac.py.
    They should be merged in the future. The metal version here
    is more efficient both in memory and computational time.

    Parameters
    ----------
    omega : double
        real position of imaginary frequency
    mo_energy : double ndarray
        orbital energy
    mo_occ : double ndarray
        occupation number
    Lpq : list of complex ndarray
        three-center density-fitting matrix in MO.
        Lpq[ki] contains the naux x (nocc_i + nfrac_i) x (nfrac_i + nvir_i) sub-block.
    kidx : list
        momentum-conserved k-point list kj=kidx[ki]

    Returns
    -------
    Pi : complex ndarray
        Pi in auxiliary basis at freq iw
    """
    nkpts = len(Lpq)
    naux = Lpq[0].shape[0]

    # Compute Pi for kL
    Pi = np.zeros(shape=[naux, naux], dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        idx_occ_i, _, idx_vir_i = get_idx_metal(mo_occ[i])
        idx_occ_a, idx_frac_a, idx_vir_a = get_idx_metal(mo_occ[a])

        # merge index
        idx_i = slice(idx_occ_i[0], idx_vir_i[0])
        idx_a = slice(idx_occ_a[-1] + 1, idx_vir_a[-1] + 1)
        nocc_i = len(idx_occ_i)
        nfrac_a = len(idx_frac_a)

        eia = mo_energy[i, idx_i, None] - mo_energy[a, None, idx_a]
        fia = (mo_occ[i][idx_i, None] - mo_occ[a][None, idx_a]) / 2.0

        # factor of 0.5 is for double counting
        fia[nocc_i:, :nfrac_a] *= 0.5
        # Response from both spin-up and spin-down density
        rho_accum_inner(Pi, eia, omega, Lpq[i], alpha=4.0 / nkpts, fia=fia)

    return Pi


def rho_accum_inner(Pi, eia, omega, Lov, alpha=0.0, fia=None):
    """Get contribution to response function from current occupied-virtual block.

    Parameters
    ----------
    Pi : complex 2d array
        density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        real position of imaginary frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    alpha : float, optional
        prefactor, by default 0.0
    fia : double 2d array, optional
        occupied-virtual occupation number difference, by default None
    """
    naux, nocc, nvir = Lov.shape

    if fia is None:
        eia = eia / (omega**2 + eia**2)
    else:
        eia = eia * fia / (omega**2 + eia**2)
    Pia = (Lov * eia).reshape(naux, nocc * nvir)

    # The following call to blas.zgemm may be replaced with
    # Pi += alpha * np.einsum('Pia, Qia -> PQ', Pia, Lov.conj(), optimize=True)
    # with a moderate performance hit.

    # zgemm is complex matrix multiplication. A wrapper is included in SciPy.
    # C <- alpha * op(A) @ op(B) + beta * C
    blas.zgemm(
        alpha=alpha,
        a=Lov.reshape(naux, nocc * nvir).T,
        b=Pia.T,
        trans_a=2,  # take conjugate transpose of A (this gives Lov.conj())
        trans_b=0,  # B is Pia.T
        beta=1.0,
        c=Pi.T,  # Pi.T += alpha * Lov.conj() @ Pia.T
        overwrite_c=True,
    )

    return


def rho_wing_accum_inner(Pi_P0, eia, omega, Lov, qov, alpha=0.0):
    """Accumulate the finite-size-correction wing response for one OV slice.

    Parameters
    ----------
    Pi_P0 : complex 1d array
        finite-size correction to density-density response function, will be overwritten
    eia : double 2d array
        occupied-virtual orbital energy difference
    omega : double
        frequency
    Lov : complex 3d array
        occupied-virtual block of three-center density-fitting matrix in MO
    qov : complex 2d array
        virtual-occupied correction
    alpha : float, optional
        prefactor, by default 0.0
    """
    naux, nocc, nvir = Lov.shape
    eia_q = eia * qov.conj() / (omega**2 + eia**2)
    Pi_P0 += alpha * np.matmul(Lov.reshape(naux, nocc * nvir), eia_q.reshape(nocc * nvir))

    return


def get_rpa_ecorr(rpa, freqs, wts):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo - nocc
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = hasattr(rpa._scf, 'sigma')

    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts)
    cderiarr = mydf.cderi_array()

    for kL in range(nkpts):
        # Lij: (ki, L, i, j) for looping every kL
        if is_metal:
            Lij = []
        else:
            Lij = None
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        # kidx = np.zeros((nkpts),dtype=np.int64)
        # kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            j = kconserv_table[kL, i]
            kptj = kpts[j]
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12  # kidx[i] = j
            # kidx_r[j] = i
            logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, ki: {i}, kj: {j})')
            # Read (L|pq) and ao2mo transform to (L|ij)
            # support unequal naux on different k points
            Lpq = cderiarr.load(kpti, kptj)
            if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
            else:
                Lpq = Lpq.reshape(-1, nao**2)
            Lpq = Lpq.astype(np.complex128)
            tao = []
            ao_loc = None
            moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

            naux = Lpq.shape[0]
            if not is_metal:
                if Lij is None:
                    Lij = np.zeros((nkpts, naux, nocc, nvir), dtype=np.complex128)
                ijslice = (0, nocc, nmo + nocc, 2 * nmo)
                r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij[i])
            else:
                # Only (nocc+nfrac, nfrac+nvir) block of Lpq is needed
                # This is consistent with the new get_rho_response_metal implementation
                idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i])
                idx_occ_j, idx_frac_j, idx_vir_j = get_idx_metal(mo_occ[j])

                nocc_i = len(idx_occ_i)
                nfrac_i = len(idx_frac_i)
                nocc_j = len(idx_occ_j)
                nfrac_j = len(idx_frac_j)
                nvir_j = len(idx_vir_j)
                ijslice = (0, nocc_i + nfrac_i, nmo + nocc_j, 2 * nmo)

                Lij.append(r_e2(Lpq, moij, ijslice, tao, ao_loc).reshape(naux, nocc_i + nfrac_i, nfrac_j + nvir_j))

        for w in range(nw):
            if is_metal:
                Pi = get_rho_response_metal(freqs[w], mo_energy, mo_occ, Lij, kconserv_table[kL])
            else:
                Pi = get_rho_response(freqs[w], mo_energy, Lij, kconserv_table[kL])
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    # head Pi_00
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    Pi_00 = 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00
                    # wings Pi_P0
                    Pi_P0 = get_rho_response_wing(freqs[w], mo_energy, Lij, qij[iq])
                    Pi_P0 = np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                    # assemble Pi
                    Pi_fc = np.zeros((naux + 1, naux + 1), dtype=Pi.dtype)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0.conj()
                    Pi_fc[1:, 0] = Pi_P0
                    Pi_fc[1:, 1:] = Pi

                    e_corr += get_rpa_ecorr_w(Pi_fc, wts[w])
            else:
                e_corr += get_rpa_ecorr_w(Pi, wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_outcore(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))

    nocc = rpa.nocc
    nmo = rpa.nmo
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    if rpa.fc:
        qij, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts)
    cderiarr = mydf.cderi_array()

    for kL in range(nkpts):
        Pi = None
        Pi_P0 = None
        nseg = nocc // rpa.segsize + 1
        for iseg in range(nseg):
            orb_start = iseg * rpa.segsize
            orb_end = min((iseg + 1) * rpa.segsize, nocc)
            if orb_end == orb_start:
                continue
            norb_this_iter = orb_end - orb_start

            # Lij: (ki, L, i, j) for looping every kL
            # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
            # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
            # kidx = np.zeros((nkpts),dtype=np.int64)
            # kidx_r = np.zeros((nkpts),dtype=np.int64)
            for i, kpti in enumerate(kpts):
                j = kconserv_table[kL, i]
                kptj = kpts[j]
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, ki: {i}, kj: {j})')
                # Read (L|pq) and ao2mo transform to (L|ij)
                # support uneqaul naux on different k points
                Lpq = cderiarr.load(kpti, kptj)
                if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
                    Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
                else:
                    Lpq = Lpq.reshape(-1, nao**2)
                Lpq = Lpq.astype(np.complex128)
                naux = Lpq.shape[0]

                tao = []
                ao_loc = None
                moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                ijslice = (orb_start, orb_end, nmo + nocc, 2 * nmo)
                Lij_slice = r_e2(Lpq, moij, ijslice, tao, ao_loc)
                Lij_slice = Lij_slice.reshape(naux, norb_this_iter, nmo - nocc)
                if Pi is None:
                    Pi = np.zeros((nw, naux, naux), dtype=np.complex128)
                    if kL == 0 and rpa.fc:
                        Pi_P0 = np.zeros((nq_pts, nw, naux), dtype=np.complex128)

                # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                a_inner = kconserv_table[kL, i]
                eia = mo_energy[i][orb_start:orb_end, None] - mo_energy[a_inner][None, nocc:]
                for w in range(nw):
                    rho_accum_inner(Pi[w], eia, freqs[w], Lij_slice, alpha=4.0 / nkpts)
                    if kL == 0 and rpa.fc:
                        for iq in range(nq_pts):
                            rho_wing_accum_inner(
                                Pi_P0[iq, w],
                                eia,
                                freqs[w],
                                Lij_slice,
                                qij[iq, i, orb_start:orb_end],
                                alpha=4.0 / nkpts,
                            )

        for w in range(nw):
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, qij[iq])
                    Pi_00 = 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00
                    Pi_P0_iq = np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0[iq, w]

                    Pi_fc = np.zeros((naux + 1, naux + 1), dtype=Pi.dtype)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0_iq.conj()
                    Pi_fc[1:, 0] = Pi_P0_iq
                    Pi_fc[1:, 1:] = Pi[w]

                    e_corr += get_rpa_ecorr_w(Pi_fc, wts[w])
            else:
                e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_outcore_metal(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy for metals.

    Parameters
    ----------
    rpa : KRPA
        rpa object
    freqs : double 1d array
        frequency grid
    wts : double 1d array
        weight of grids

    Returns
    -------
    e_corr : double
        correlation energy
    """
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nmo = rpa.nmo
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    e_corr = 0j

    # Precompute k-conservation table
    # Given k-point indices (kL, i), kconserv_table[kshift,i] contains
    # the index j that satisfies momentum conservation,
    # (k(i) - k(j) - k(kL)) \dot a = 2n\pi
    # i.e.
    # - ki + kj + kL = G
    kconserv_table = get_kconserv_ria_efficient(rpa.mol, kpts)
    cderiarr = mydf.cderi_array()

    for kL in range(nkpts):
        Pi = None
        # Lij: (ki, L, i, j) for looping every kL
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        # kidx = np.zeros((nkpts),dtype=np.int64)
        # kidx_r = np.zeros((nkpts),dtype=np.int64)
        for i, kpti in enumerate(kpts):
            j = kconserv_table[kL, i]
            kptj = kpts[j]
            # Find (ki,kj) that satisfies momentum conservation with kL
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
            logger.debug(rpa, f'Read Lpq (kL: {kL+1} / {nkpts}, ki: {i}, kj: {j})')
            # Read (L|pq) and ao2mo transform to (L|ij)
            # support uneqaul naux on different k points
            Lpq = cderiarr.load(kpti, kptj)
            if Lpq.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq = lib.unpack_tril(Lpq).reshape(-1, nao**2)
            else:
                Lpq = Lpq.reshape(-1, nao**2)
            Lpq = Lpq.astype(np.complex128)
            naux = Lpq.shape[0]

            idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[i])
            idx_occ_j, idx_frac_j, idx_vir_j = get_idx_metal(mo_occ[j])

            nocc_i = len(idx_occ_i)
            nfrac_i = len(idx_frac_i)
            nocc_j = len(idx_occ_j)
            nfrac_j = len(idx_frac_j)
            nseg = (nocc_i + nfrac_i) // rpa.segsize + 1
            for iseg in range(nseg):
                orb_start = iseg * rpa.segsize
                orb_end = min((iseg + 1) * rpa.segsize, nocc_i + nfrac_i)
                if orb_end == orb_start:
                    break
                norb_this_iter = orb_end - orb_start

                tao = []
                ao_loc = None
                moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]

                ijslice = (orb_start, orb_end, nmo + nocc_j, 2 * nmo)
                Lij_slice = r_e2(Lpq, moij, ijslice, tao, ao_loc)
                Lij_slice = Lij_slice.reshape(naux, norb_this_iter, nmo - nocc_j)
                if Pi is None:
                    Pi = np.zeros((nw, naux, naux), dtype=np.complex128)

                # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                eia = mo_energy[i][orb_start:orb_end, None] - mo_energy[j][None, nocc_j:]
                fia = (mo_occ[i][orb_start:orb_end, None] - mo_occ[j][None, nocc_j:]) / 2.0
                # The overall fia[nocc_i:, :nfrac_j] *= 0.5 for double counting
                if orb_start >= nocc_i:
                    fia[:, :nfrac_j] *= 0.5
                elif orb_end > nocc_i:
                    offset = nocc_i - orb_start
                    fia[offset:, :nfrac_j] *= 0.5
                for w in range(nw):
                    rho_accum_inner(Pi[w], eia, freqs[w], Lij_slice, alpha=4.0 / nkpts, fia=fia)

        for w in range(nw):
            e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_w(Pi_w, wts_w):
    """Get contribution to RPA correlation energy from a single frequency.

    Parameters
    ----------
    Pi_w : complex 2d array
        density-density response function at a single frequency
    wts_w : double
        weights of the frequency

    Returns
    -------
    e_corr : double
        correlation energy
    """
    # First, compute ec_w = Tr(Pi_w) + |log(det(I-Pi_w))|
    ec_w = np.trace(Pi_w)
    # The following two lines are equivalent to
    # Pi_w = np.eye(naux) - Pi_w
    blas.zdscal(-1.0, Pi_w.ravel(), overwrite_x=1)
    np.fill_diagonal(Pi_w, np.diagonal(Pi_w) + 1.0)

    ec_w += np.linalg.slogdet(Pi_w)[1]
    #e_corr = 1.0 / (2.0 * np.pi) / nkpts * ec_w * wts_w
    e_corr = ec_w * wts_w

    return e_corr


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : KRPA
        rpa object
    acfd : bool, optional
        calculate ACFD exchange energy, by default False
    correction_only : bool, optional
        only calculate the correction term, by default False

    Returns
    -------
    ex : double
        exchange energy
    """
    mo_energy = np.array(_mo_energy_frozen(rpa, rpa._scf.mo_energy))
    mo_coeff = np.array(_mo_frozen(rpa, rpa._scf.mo_coeff))
    mo_occ = np.array(_mo_occ_frozen(rpa, rpa._scf.mo_occ))

    nocc = rpa.nocc
    nao = rpa._scf.mo_coeff[0].shape[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    mydf = rpa.with_df

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    ex = 0j
    cderiarr = mydf.cderi_array()
    for kL in range(nkpts):
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros(shape=[nkpts], dtype=np.int64)
        kidx_r = np.zeros(shape=[nkpts], dtype=np.int64)
        for i in range(nkpts):
            for j in range(nkpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i

        for kn in range(nkpts):
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            km = kidx_r[kn]

            # logger.debug(gw, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s @ Rank %d)' % (kL + 1, nkpts, i, j, rank))
            # Read (L|pq) and ao2mo transform to (L|ij)
            # support unequal naux on different k points
            Lpq_ao = cderiarr.load(kpts[km], kpts[kn])
            if Lpq_ao.shape[-1] == (nao * (nao + 1)) // 2:
                Lpq_ao = lib.unpack_tril(Lpq_ao).reshape(-1, nao**2)
            else:
                Lpq_ao = Lpq_ao.reshape(-1, nao**2)
            Lpq_ao = Lpq_ao.astype(np.complex128)

            Lij = None
            if hasattr(rpa._scf, 'sigma'):
                idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[km])
                idx_occ_j, idx_frac_j, _ = get_idx_metal(mo_occ[kn])
                nocc_i = len(idx_occ_i) + len(idx_frac_i)
                nocc_j = len(idx_occ_j) + len(idx_frac_j)
                moij, ijslice = _conc_mos(mo_coeff[km][:, :nocc_i], mo_coeff[kn][:, :nocc_j])[2:]
                Lij = r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                Lij = Lij.reshape(-1, nocc_i, nocc_j)

                if acfd is True:
                    if correction_only is True:
                        mo_occ_ij = np.minimum(mo_occ[km][:nocc_i, None], mo_occ[kn][None, :nocc_j]) / 2.0
                        mo_occ_ij -= mo_occ[km][:nocc_i, None] * mo_occ[kn][None, :nocc_j] / 4.0
                    else:
                        # numerical integration for equation 12 in doi.org/10.1103/PhysRevB.81.115126
                        # NOTE: this integration is not stable!!!
                        # w, wts = _get_scaled_legendre_roots(200)
                        #eij = mo_energy[km][:nocc_i, None] - mo_energy[kn][None, :nocc_j]
                        ##integrad = eij[:, :, None] / lib.direct_sum("ij+w->ijw", eij**2, w**2) * wts[None, None]
                        #integrand = eij[:, :, None] / (eij[:, :, None]**2 + w**2) * wts[None, None]
                        #integrand = np.sum(integrand, axis=2) * 2.0 / np.pi

                        # The following line is equivalent to the frequency integration in equation 12 in
                        # doi.org/10.1103/PhysRevB.81.115126
                        # TODO: add a detailed note
                        eij = mo_energy[km][:nocc_i, None] - mo_energy[kn][None, :nocc_j]
                        integrand = np.zeros((nocc_i, nocc_j), dtype=np.complex128)
                        integrand[eij > 1e-6] = 1
                        integrand[eij < -1e-6] = -1
                        mo_occ_ij = 1.0 - integrand
                        # spin-restricted mo_occ should be divided by 2
                        mo_occ_ij = mo_occ_ij * mo_occ[km][:nocc_i, None] / 2.0
                else:
                    mo_occ_ij = mo_occ[km][:nocc_i, None] * mo_occ[kn][None, :nocc_j] / 4.0
                Lij_occ = Lij * mo_occ_ij[None]
                # ex -= np.einsum('Lij,Lij->', Lij_occ.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                ex -= blas.zdotc(Lij_occ.ravel(), Lij.ravel())
            else:
                moij, ijslice = _conc_mos(mo_coeff[km][:, :nocc], mo_coeff[kn][:, :nocc])[2:]
                Lij = r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                # ex -= np.einsum('Lij,Lij->', Lij.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                ex -= blas.zdotc(Lij.ravel(), Lij.ravel())

    ex = ex.real
    ex /= nkpts**2

    if rpa._scf.exxdiv == 'ewald' and rpa._scf.cell.dimension != 0:
        madelung = tools.pbc.madelung(rpa._scf.cell, kpts)
        exxdiv_shift = madelung * np.sum(mo_occ**2) / (4.0 * nkpts)
        ex -= exxdiv_shift
        if acfd is True:
            for k in range(nkpts):
                idx_occ, idx_frac, _ = get_idx_metal(mo_occ[k])
                f_i = mo_occ[k][:(len(idx_occ) + len(idx_frac))] / 2.0
                ex -= madelung * np.sum(f_i - f_i * f_i) / nkpts

    return ex


def get_kconserv_ria_efficient(cell, kpts, tol=1e-12):
    r"""Get the momentum conservation array for single excitation amplitudes
    for a set of k-points with appropriate k-shift.


    Given k-point indices (kshift, m) the array kconserv[kshift,m] returns
    the index n that satisfies momentum conservation,

        (k(m) - k(n) - k(kshift)) \dot a = 2n\pi
    """
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2 * np.pi)

    kconserv = np.zeros((nkpts, nkpts), dtype=int)
    kvKM = -kpts[:, None, :] + kpts[:, :]
    for N, kvN in enumerate(kpts):
        kvKMN = np.einsum('wx,kmx->wkm', a, kvKM - kvN, optimize=True)
        # check whether (1/(2pi) k_{KLN} dot a) is an integer
        kvKMN_int = np.rint(kvKMN)
        mask = np.einsum('wkm->km', abs(kvKMN - kvKMN_int), optimize=True) < tol
        kconserv[mask] = N
    return kconserv


class KRPA(lib.StreamObject):
    def __init__(self, mf, frozen=None):
        self.mol = mf.mol  # mol object
        self._scf = mf  # mean-field object
        self.verbose = self.mol.verbose  # verbose level
        self.stdout = self.mol.stdout  # standard output
        self.max_memory = mf.max_memory  # max memory in MB

        # options
        self.frozen = frozen  # frozen orbital options
        self.grids_alg = 'legendre'  # algorithm to generate grids
        self.outcore = False  # low-memory routine
        self.segsize = 50  # number of orbitals in one segment for outcore
        self.fc = False  # finite-size correction
        self.fc_grid = False  # grids for finite-size correction
        self.acfd_exx = False  # calculate ACFD exchange energy

        # don't modify the following attributes, they are not input options
        self._nocc = None  # number of occupied orbitals
        self._nmo = None  # number of orbitals (exclude frozen orbitals)
        self.kpts = mf.kpts  # k-points
        self.nkpts = len(self.kpts)  # number of k-points
        self.mo_energy = np.array(mf.mo_energy, copy=True)  # orbital energy
        self.mo_coeff = np.array(mf.mo_coeff, copy=True)  # orbital coefficient
        self.mo_occ = np.array(mf.mo_occ, copy=True)  # occupation number
        self.e_corr = None  # correlation energy
        self.e_hf = None  # Hartree-Fock energy
        self.e_tot = None  # total energy

        # KRPA must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info(f'RPA nocc = {nocc}, nvir = {nvir}, nkpts = {nkpts}')
        if self.frozen is not None:
            log.info(f'frozen orbitals = {str(self.frozen)}')
        log.info('grid type = %s', self.grids_alg)
        log.info('outcore mode = %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('RPA finite size corrections = %s', self.fc)
        log.info('ACFD exchange energy = %s', self.acfd_exx)
        log.info('')
        return

    @property
    def nocc(self):
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy)
        nelec = 0.0
        for k in range(nkpts):
            nelec += np.sum(self._scf.mo_occ[k][frozen_mask[k]])
        nelec = int(nelec / nkpts)
        return nelec // 2

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        frozen_mask = get_frozen_mask(self)
        return len(self._scf.mo_energy[0][frozen_mask[0]])

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, nw=None, with_e_hf=None):
        """RPA correlation and total energy

        Calculated total energy, HF energy and RPA correlation energy
        are stored in self.e_tot, self.e_hf, self.e_corr

        Parameters
        ----------
        mo_energy : double array
            molecular orbital energies
        mo_coeff : double ndarray
            molecular orbital coefficients
        nw : int, optional
            number of frequency point on imaginary axis, by default None
        with_e_hf : float, optional
            If given, overrides the HF energy computation.

        Returns
        -------
        e_tot : float
            RPA total energy
        e_hf : float
            HF energy (exact exchange for given mo_coeff)
        e_corr : float
            RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = _mo_frozen(self, self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, mo_coeff, nw=nw, with_e_hf=with_e_hf)
        logger.timer(self, 'RPA', *cput0)
        return self.e_tot, self.e_hf, self.e_corr

    def get_grids(self, alg=None, nw=None, mo_energy=None):
        """Generate grids for integration.

        Parameters
        ----------
        alg : str, optional
            algorithm for generating grids, by default None
        nw : int, optional
            number of grids, by default None
        mo_energy : double 2d array, optional
            orbital energy, used for minimax grids, by default None

        Returns
        -------
        freqs : double 1d array
            frequency grid
        wts : double 1d array
            weight of grids
        """
        if alg is None:
            alg = self.grids_alg
        if mo_energy is None:
            mo_energy = _mo_energy_frozen(self, self._scf.mo_energy)
        if alg == 'legendre':
            nw = 40 if nw is None else nw
            freqs, wts = _get_scaled_legendre_roots(nw)
        else:
            raise NotImplementedError('Grids algorithm not implemented!')

        return freqs, wts

    def get_q_mesh(self, mo_energy, mo_coeff):
        """Get q-mesh for finite size correction.
        Equation 39-42 in doi.org/10.1021/acs.jctc.0c00704

        Parameters
        ----------
        mo_energy : double 2d array
            orbital energy
        mo_coeff : double 3d array
            coefficient from AO to MO

        Returns
        -------
        qij : double 1d array
            q-mesh grids
        q_abs : double 1d array
            absolute positions of q-mesh grids
        nq_pts : init
            number of q-mesh grids
        """
        nocc = self.nocc
        nmo = self.nmo
        nkpts = self.nkpts
        # Set up q mesh for q->0 finite size correction
        if not self.fc_grid:
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
        q_abs = self.mol.get_abs_kpts(q_pts)

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = np.zeros(shape=[nq_pts, nkpts, nocc, nmo - nocc], dtype=np.complex128)

        if not self.fc_grid:
            for k in range(nq_pts):
                qij[k] = get_qij(self, q_abs[k], mo_energy, mo_coeff)
        else:
            for k in range(nq_pts):
                qij[k] = get_qij(self, q_abs[k], mo_energy, mo_coeff)

        return qij, q_abs, nq_pts

    def get_acfd_exx(self, correction_only=False):
        """Calculate ACFD exchange energy.

        Parameters
        ----------
        correction_only : bool
            only return the correction term

        Returns
        -------
        ex_acfd : double
            ACFD exchange energy
        """
        ex_acfd = get_rpa_exx(self, acfd=True, correction_only=correction_only)
        return ex_acfd
