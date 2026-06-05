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
Periodic spin-unrestricted random phase approximation (direct RPA) with N^4 scaling.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    New J. Phys. 14, 053020 (2012)
"""

import time
import numpy as np
import scipy.linalg.blas as blas

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import scf, tools
from pyscf.pbc.cc.kccsd_uhf import get_nocc, get_nmo, get_frozen_mask

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.pbc.gw.kugw_ac import get_rho_response, get_rho_response_metal, get_rho_response_head, \
    get_rho_response_wing, get_qij
from pyscf.pbc.gw.krpa import KRPA, rho_accum_inner, get_rpa_ecorr_w, get_kconserv_ria_efficient


def kernel(rpa, mo_energy, mo_coeff, nw=None, with_e_hf=None):
    """RPA correlation and total energy

    Parameters
    ----------
    rpa : KURPA
        rpa object
    mo_energy : double array
        molecular orbital energies
    mo_coeff : double ndarray
        molecular orbital coefficients
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
    assert rpa.frozen == 0 or rpa.frozen is None

    # Compute HF exchange energy (EXX)
    if with_e_hf is None:
        uhf = scf.KUHF(rpa.mol, rpa.kpts, exxdiv=rpa._scf.exxdiv)
        uhf.verbose = 0
        if hasattr(rpa._scf, 'sigma'):
            uhf = scf.addons.smearing_(uhf, sigma=rpa._scf.sigma, method=rpa._scf.smearing_method)
        uhf.with_df = rpa._scf.with_df
        with temporary_env(rpa.with_df, verbose=0), temporary_env(rpa.mol, verbose=0):
            dm = rpa._scf.make_rdm1()
            vj = uhf.get_j(uhf.cell, dm)
            vj_tot = vj[0] + vj[1]
            e_1e = 1.0 / len(rpa.kpts) * lib.einsum('kij,kji', dm[0] + dm[1], uhf.get_hcore()).real
            e_j = 0.5 / len(rpa.kpts) * lib.einsum('kij,kji', dm[0] + dm[1], vj_tot).real
            e_x = get_rpa_exx(rpa, acfd=rpa.acfd_exx, correction_only=False)
            e_nuc = rpa._scf.energy_nuc()
            e_hf = e_1e + e_j + e_x + e_nuc
    else:
        e_hf = with_e_hf
        logger.debug(rpa, f'  Setting EXX energy explicitly to {e_hf}')

    is_metal = hasattr(rpa._scf, 'sigma')

    # Turn off FC for metals and outcore
    if is_metal and rpa.fc:
        logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
        rpa.fc = False
    if rpa.fc and rpa.outcore:
        logger.warn(rpa, 'FC not available for outcore - setting rpa.fc to False')
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

    logger.debug(rpa, '  RPA total energy = %s', e_tot)
    logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

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
    idx_occ = np.where(mo_occ > 1.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    idx_frac = list(range(idx_occ[-1] + 1, idx_vir[0]))

    return idx_occ, idx_frac, idx_vir


def get_rpa_ecorr(rpa, freqs, wts):
    """Compute RPA correlation energy.

    Parameters
    ----------
    rpa : KURPA
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
    mo_energy = np.array(rpa._scf.mo_energy)
    mo_coeff = np.array(rpa._scf.mo_coeff)
    nmoa, nmob = rpa.nmo
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df
    mo_occ = rpa.mo_occ

    # possible kpts shift
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = hasattr(rpa._scf, 'sigma')

    if rpa.fc:
        qij_a, qij_b, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

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
        # Lij: (2, ki, L, i, j) for looping every kL
        # Lij = np.zeros((2,nkpts,naux,nmoa,nmoa),dtype=np.complex128)
        Lij = []
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for i, kpti in enumerate(kpts):
            j = kconserv_table[kL, i]
            kptj = kpts[j]
            # Find (ki,kj) that satisfies momentum conservation with kL
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12  # kidx[i] = j
            kidx[i] = j
            kidx_r[j] = i
            logger.debug(rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, i, j))
            Lij_out_a = None
            Lij_out_b = None
            # Read (L|pq) and ao2mo transform to (L|ij)
            Lpq = cderiarr.load(kpti, kptj)
            if Lpq.shape[-1] == (nmoa * (nmoa + 1)) // 2:
                Lpq = lib.unpack_tril(Lpq).reshape(-1, nmoa**2)
            else:
                Lpq = Lpq.reshape(-1, nmoa**2)
            Lpq = Lpq.astype(np.complex128)
            moija, ijslicea = _conc_mos(mo_coeff[0, i], mo_coeff[0, j])[2:]
            moijb, ijsliceb = _conc_mos(mo_coeff[1, i], mo_coeff[1, j])[2:]
            Lij_out_a = _ao2mo.r_e2(Lpq, moija, ijslicea, tao=[], ao_loc=None, out=Lij_out_a)
            Lij_out_b = _ao2mo.r_e2(Lpq, moijb, ijsliceb, tao=[], ao_loc=None, out=Lij_out_b)
            Lij.append(np.asarray((Lij_out_a.reshape(-1, nmoa, nmoa), Lij_out_b.reshape(-1, nmob, nmob))))

        Lij = np.asarray(Lij)
        naux = Lij.shape[2]
        if is_metal is False:
            Lia = [
                np.ascontiguousarray(Lij[:, 0, :, : rpa.nocc[0], rpa.nocc[0] :]),
                np.ascontiguousarray(Lij[:, 1, :, : rpa.nocc[1], rpa.nocc[1] :]),
            ]

        for w in range(nw):
            # body polarizability
            if is_metal:
                Pi = get_rho_response_metal(freqs[w], mo_energy, mo_occ, Lij, kidx)
            else:
                Pi = get_rho_response(freqs[w], rpa.nocc, mo_energy, Lia, kidx)
            if kL == 0 and rpa.fc:
                for iq in range(nq_pts):
                    # head Pi_00
                    Pi_00 = get_rho_response_head(freqs[w], mo_energy, (qij_a[iq], qij_b[iq]))
                    Pi_00 = 4.0 * np.pi / np.linalg.norm(q_abs[iq]) ** 2 * Pi_00
                    # wings Pi_P0
                    Pi_P0 = get_rho_response_wing(freqs[w], mo_energy, Lia, (qij_a[iq], qij_b[iq]))
                    Pi_P0 = np.sqrt(4.0 * np.pi) / np.linalg.norm(q_abs[iq]) * Pi_P0

                    # assemble Pi
                    Pi_fc = np.zeros((naux + 1, naux + 1), dtype=Pi.dtype)
                    Pi_fc[0, 0] = Pi_00
                    Pi_fc[0, 1:] = Pi_P0.conj()
                    Pi_fc[1:, 0] = Pi_P0
                    Pi_fc[1:, 1:] = Pi

                    # First, compute ec_w = Tr(Pi) + |log(det(I-Pi))|
                    ec_w = np.trace(Pi_fc)
                    # The following two lines are equivalent to
                    # Pi = np.eye(naux) - Pi
                    blas.zdscal(-1.0, Pi_fc.ravel(), overwrite_x=1)
                    np.fill_diagonal(Pi_fc, np.diagonal(Pi_fc) + 1.0)
                    ec_w += np.linalg.slogdet((Pi_fc))[1]
                    e_corr += 1.0 / (2.0 * np.pi) * 1.0 / nkpts * 1.0 / nq_pts * ec_w * wts[w]
            else:
                # First, compute ec_w = Tr(Pi) + |log(det(I-Pi))|
                ec_w = np.trace(Pi)
                # The following two lines are equivalent to
                # Pi = np.eye(naux) - Pi
                blas.zdscal(-1.0, Pi.ravel(), overwrite_x=1)
                np.fill_diagonal(Pi, np.diagonal(Pi) + 1.0)
                ec_w += np.linalg.slogdet((Pi))[1]
                e_corr += 1.0 / (2.0 * np.pi) * 1.0 / nkpts * ec_w * wts[w]

    return e_corr.real


def get_rpa_ecorr_outcore(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy.

    Parameters
    ----------
    rpa : KURPA
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
    mo_energy = np.array(rpa._scf.mo_energy)
    mo_coeff = np.array(rpa._scf.mo_coeff)
    nmoa = rpa.nmo[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df

    # possible kpts shift
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    if rpa.fc:
        qij_a, qij_b, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

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
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for s in range(2):
            nseg = rpa.nocc[s] // rpa.segsize + 1
            for iseg in range(nseg):
                orb_start = iseg * rpa.segsize
                orb_end = min((iseg + 1) * rpa.segsize, rpa.nocc[s])
                if orb_end == orb_start:
                    continue
                norb_this_iter = orb_end - orb_start

                for i, kpti in enumerate(kpts):
                    j = kconserv_table[kL, i]
                    kptj = kpts[j]
                    # Find (ki,kj) that satisfies momentum conservation with kL
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12  # kidx[i] = j
                    kidx[i] = j
                    kidx_r[j] = i
                    logger.debug(rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, i, j))
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = cderiarr.load(kpti, kptj)
                    if Lpq.shape[-1] == (nmoa * (nmoa + 1)) // 2:
                        Lpq = lib.unpack_tril(Lpq).reshape(-1, nmoa**2)
                    else:
                        Lpq = Lpq.reshape(-1, nmoa**2)
                    Lpq = Lpq.astype(np.complex128)
                    naux = Lpq.shape[0]
                    moij, ijslice = _conc_mos(mo_coeff[s, i], mo_coeff[s, j])[2:]
                    ijslice = (orb_start, orb_end, rpa.nmo[s] + rpa.nocc[s], 2 * rpa.nmo[s])
                    Lij_slice = _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None)
                    Lij_slice = Lij_slice.reshape(naux, norb_this_iter, rpa.nmo[s] - rpa.nocc[s])
                    if Pi is None:
                        Pi = np.zeros((nw, naux, naux), dtype=np.complex128)
                    eia = mo_energy[s, i][orb_start:orb_end, None] - mo_energy[s, j][None, rpa.nocc[s] :]
                    for w in range(nw):
                        rho_accum_inner(Pi[w], eia, freqs[w], Lij_slice, alpha=2.0 / nkpts)

        for w in range(nw):
            e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts
    return e_corr


def get_rpa_ecorr_outcore_metal(rpa, freqs, wts):
    """Low-memory routine to compute RPA correlation energy for metals.

    Parameters
    ----------
    rpa : KURPA
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
    mo_energy = np.array(rpa._scf.mo_energy)
    mo_coeff = np.array(rpa._scf.mo_coeff)
    nmoa = rpa.nmo[0]
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df
    mo_occ = np.array(rpa.mo_occ)

    # possible kpts shift
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    if rpa.fc:
        qij_a, qij_b, q_abs, nq_pts = rpa.get_q_mesh(mo_energy, mo_coeff)

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
        kidx = np.zeros((nkpts), dtype=np.int64)
        kidx_r = np.zeros((nkpts), dtype=np.int64)
        for s in range(2):
            for i, kpti in enumerate(kpts):
                j = kconserv_table[kL, i]
                kptj = kpts[j]
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                assert np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12  # kidx[i] = j
                kidx[i] = j
                kidx_r[j] = i
                logger.debug(rpa, 'Read Lpq (kL: %s / %s, ki: %s, kj: %s)' % (kL + 1, nkpts, i, j))
                # Read (L|pq) and ao2mo transform to (L|ij)
                Lpq = cderiarr.load(kpti, kptj)
                if Lpq.shape[-1] == (nmoa * (nmoa + 1)) // 2:
                    Lpq = lib.unpack_tril(Lpq).reshape(-1, nmoa**2)
                else:
                    Lpq = Lpq.reshape(-1, nmoa**2)
                Lpq = Lpq.astype(np.complex128)
                naux = Lpq.shape[0]

                idx_occ_i, idx_frac_i, idx_vir_i = get_idx_metal(mo_occ[s, i])
                idx_occ_j, idx_frac_j, idx_vir_j = get_idx_metal(mo_occ[s, j])

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

                    moij, ijslice = _conc_mos(mo_coeff[s, i], mo_coeff[s, j])[2:]

                    ijslice = (orb_start, orb_end, rpa.nmo[s] + nocc_j, 2 * rpa.nmo[s])
                    Lij_slice = _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None)
                    Lij_slice = Lij_slice.reshape(naux, norb_this_iter, rpa.nmo[s] - nocc_j)
                    if Pi is None:
                        Pi = np.zeros((nw, naux, naux), dtype=np.complex128)

                    # Find ka that conserves with ki and kL (-ki+ka+kL=G)
                    eia = mo_energy[s, i][orb_start:orb_end, None] - mo_energy[s, j][None, nocc_j:]
                    fia = mo_occ[s, i][orb_start:orb_end, None] - mo_occ[s, j][None, nocc_j:]
                    # The overall fia[nocc_i:, :nfrac_j] *= 0.5 for double counting
                    if orb_start >= nocc_i:
                        fia[:, :nfrac_j] *= 0.5
                    elif orb_end > nocc_i:
                        offset = nocc_i - orb_start
                        fia[offset:, :nfrac_j] *= 0.5
                    for w in range(nw):
                        rho_accum_inner(Pi[w], eia, freqs[w], Lij_slice, alpha=2.0 / nkpts, fia=fia)

        for w in range(nw):
            e_corr += get_rpa_ecorr_w(Pi[w], wts[w])

    e_corr = e_corr.real
    e_corr *= 1.0 / (2.0 * np.pi) / nkpts

    return e_corr


def get_rpa_exx(rpa, acfd=False, correction_only=False):
    """Calculate RPA exchange energy.
    For gapped systems, Hartree-Fock and adiabatic connection fluctuation dissipation exchange energies are the same.
    For metallic systems, they are different.
    The ACFD exchange energy is given by equation 12 in doi.org/10.1103/PhysRevB.81.115126

    Parameters
    ----------
    rpa : KURPA
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
    mo_energy = np.asarray(rpa._scf.mo_energy)
    mo_coeff = np.asarray(rpa._scf.mo_coeff)
    mo_occ = np.asarray(rpa._scf.mo_occ)

    nocc = rpa.nocc
    nspin, _, nao, _ = mo_coeff.shape
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

            for s in range(nspin):
                Lij = None
                if hasattr(rpa._scf, 'sigma'):
                    idx_occ_i, idx_frac_i, _ = get_idx_metal(mo_occ[s][km])
                    idx_occ_j, idx_frac_j, _ = get_idx_metal(mo_occ[s][kn])
                    nocc_i = len(idx_occ_i) + len(idx_frac_i)
                    nocc_j = len(idx_occ_j) + len(idx_frac_j)
                    moij, ijslice = _conc_mos(mo_coeff[s][km][:, :nocc_i], mo_coeff[s][kn][:, :nocc_j])[2:]
                    Lij = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                    Lij = Lij.reshape(-1, nocc_i, nocc_j)

                    if acfd is True:
                        if correction_only is True:
                            mo_occ_ij = np.minimum(mo_occ[s][km][:nocc_i, None], mo_occ[s][kn][None, :nocc_j])
                            mo_occ_ij -= mo_occ[s][km][:nocc_i, None] * mo_occ[s][kn][None, :nocc_j]
                        else:
                            # The following line is equivalent to the frequency integration in equation 12 in
                            # doi.org/10.1103/PhysRevB.81.115126
                            # TODO: add a detailed note
                            eij = mo_energy[s][km][:nocc_i, None] - mo_energy[s][kn][None, :nocc_j]
                            integrand = np.zeros((nocc_i, nocc_j), dtype=np.complex128)
                            integrand[eij > 1e-6] = 1
                            integrand[eij < -1e-6] = -1
                            mo_occ_ij = 1.0 - integrand
                            mo_occ_ij = mo_occ_ij * mo_occ[s][km][:nocc_i, None]
                    else:
                        mo_occ_ij = mo_occ[s][km][:nocc_i, None] * mo_occ[s][kn][None, :nocc_j]
                    Lij_occ = Lij * mo_occ_ij[None]
                    # ex -= np.einsum('Lij,Lij->', Lij_occ.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                    ex -= blas.zdotc(Lij_occ.ravel(), Lij.ravel())
                else:
                    moij, ijslice = _conc_mos(mo_coeff[s][km][:, :nocc[s]], mo_coeff[s][kn][:, :nocc[s]])[2:]
                    Lij = _ao2mo.r_e2(Lpq_ao, moij, ijslice, tao=[], ao_loc=None, out=Lij)
                    # ex -= np.einsum('Lij,Lij->', Lij.reshape(-1, nocc, nocc), Lij.reshape(-1, nocc, nocc).conj())
                    ex -= blas.zdotc(Lij.ravel(), Lij.ravel())

    ex = ex.real
    ex *= 0.5 / nkpts**2

    if rpa._scf.exxdiv == 'ewald' and rpa._scf.cell.dimension != 0:
        madelung = tools.pbc.madelung(rpa._scf.cell, kpts)
        for s in range(nspin):
            exxdiv_shift = 0.5 * madelung * np.sum(mo_occ[s]**2) / (nkpts)
            ex -= exxdiv_shift
            if acfd is True:
                for k in range(nkpts):
                    idx_occ, idx_frac, _ = get_idx_metal(mo_occ[s][k])
                    f_i = mo_occ[s][k][:(len(idx_occ) + len(idx_frac))]
                    ex -= 0.5 * madelung * np.sum(f_i - f_i * f_i) / nkpts

    return ex


class KURPA(KRPA):
    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        nkpts = self.nkpts
        log.info(
            'RPA (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d), nkpts = %d', nocca, noccb, nvira, nvirb, nkpts
        )
        if self.frozen is not None and self.frozen > 0:
            log.info('frozen orbitals %s', str(self.frozen))
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
        mo_occ = self._scf.mo_occ
        return (int(np.sum(mo_occ[0][0])), int(np.sum(mo_occ[1][0])))

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return (len(self._scf.mo_energy[0][0]), len(self._scf.mo_energy[1][0]))

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, nw=None):
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
            mo_coeff = np.array(self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = np.array(self._scf.mo_energy)

        nmoa = self.nmo[0]
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (3 * nkpts * nmoa**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = kernel(self, mo_energy, mo_coeff, nw=nw)
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
        mo_energy : double 3d array, optional
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
            mo_energy = np.array(self._scf.mo_energy)

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
        mo_energy : double 3d array
            orbital energy
        mo_coeff : double 4d array
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
        # Set up q mesh for q->0 finite size correction
        nmoa, nmob = self.nmo
        nocca, noccb = self.nocc
        nkpts = self.nkpts
        if not self.fc_grid:
            q_pts = np.array([1e-3, 0, 0]).reshape(1, 3)
        else:
            Nq = 4
            q_pts = np.zeros((Nq**3 - 1, 3))
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
        qij_a = np.zeros((nq_pts, nkpts, nocca, nmoa - nocca), dtype=np.complex128)
        qij_b = np.zeros((nq_pts, nkpts, noccb, nmob - noccb), dtype=np.complex128)
        for k in range(nq_pts):
            qij_tmp = get_qij(self, q_abs[k], mo_energy, mo_coeff)
            qij_a[k] = qij_tmp[0]
            qij_b[k] = qij_tmp[1]

        return qij_a, qij_b, q_abs, nq_pts

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
