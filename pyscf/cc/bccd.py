#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>
#

"""
Brueckner coupled-cluster doubles (BCCD).
"""

from functools import reduce

import numpy as np
from scipy import linalg as la

from pyscf import lib

def get_umat_from_t1(t1):
    """
    Get rotation matrix from t1.
    """
    if isinstance(t1, np.ndarray) and t1.ndim == 2:
        nocc, nvir = t1.shape
        amat = np.zeros((nocc+nvir, nocc+nvir), dtype=t1.dtype)
        amat[:nocc, -nvir:] = -t1
        amat[-nvir:, :nocc] = t1.conj().T
        umat = la.expm(amat)
    else: # UHF
        spin = len(t1)
        nmo = np.sum(t1[0].shape)
        umat = np.zeros((spin, nmo, nmo), dtype=np.result_type(*t1))
        for s in range(spin):
            nocc, nvir = t1[s].shape
            amat = np.zeros((nmo, nmo), dtype=t1[s].dtype)
            amat[:nocc, -nvir:] = -t1[s]
            amat[-nvir:, :nocc] = t1[s].conj().T
            umat[s] = la.expm(amat)
    return umat

def transform_t1_to_bo(t1, umat):
    """
    Transform t1 to brueckner orbital basis.
    """
    if isinstance(t1, np.ndarray) and t1.ndim == 2:
        nocc, nvir = t1.shape
        umat_occ = umat[:nocc, :nocc]
        umat_vir = umat[nocc:, nocc:]
        return reduce(np.dot, (umat_occ.conj().T, t1, umat_vir))
    else: # UHF
        spin = len(t1)
        return [transform_t1_to_bo(t1[s], umat[s]) for s in range(spin)]

def transform_t2_to_bo(t2, umat, umat_b=None):
    """
    Transform t2 to brueckner orbital basis.
    """
    if isinstance(t2, np.ndarray) and t2.ndim == 4:
        umat_a = umat
        if umat_b is None:
            umat_b = umat_a

        nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
        umat_occ_a = umat_a[:nocc_a, :nocc_a]
        umat_occ_b = umat_b[:nocc_b, :nocc_b]
        umat_vir_a = umat_a[nocc_a:, nocc_a:]
        umat_vir_b = umat_b[nocc_b:, nocc_b:]
        t2_bo = np.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2, umat_occ_a,
                          umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)
        # (T) need a continuous array
        t2_bo = np.asarray(t2_bo, order='C')
    else: # UHF
        t2_bo = [None, None, None]
        t2_bo[0] = transform_t2_to_bo(t2[0], umat[0])
        t2_bo[1] = transform_t2_to_bo(t2[1], umat[0], umat_b=umat[1])
        t2_bo[2] = transform_t2_to_bo(t2[2], umat[1])
    return t2_bo

transform_l1_to_bo = transform_t1_to_bo
transform_l2_to_bo = transform_t2_to_bo

def get_mo_ovlp(mo1, mo2, ovlp):
    """
    Get MO overlap, C_1.conj().T ovlp C_2.
    """
    ovlp = np.asarray(ovlp)
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    if mo1.ndim == 2:
        res = reduce(np.dot, (mo1.conj().T, ovlp, mo2))
    else:
        assert mo1.shape[0] == mo2.shape[0]
        spin, nao, nmo1 = mo1.shape
        nmo2 = mo2.shape[-1]
        res = np.zeros((spin, nmo1, nmo2), dtype=np.result_type(mo1, mo2))
        for s in range(spin):
            res[s] = reduce(np.dot, (mo1[s].conj().T, ovlp, mo2[s]))
    return res

def logm(mrot):
    if np.iscomplexobj(mrot):
        return la.logm(mrot)
    else:
        rs = mrot + mrot.T
        rl, rv = la.eigh(rs)
        rd = rv.T @ mrot @ rv
        ra, rdet = 1, rd[0, 0]
        for i in range(1, len(rd)):
            ra, rdet = rdet, rd[i, i] * rdet - \
                rd[i - 1, i] * rd[i, i - 1] * ra
        assert rdet > 0
        ld = np.zeros_like(rd)
        for i in range(0, len(rd) // 2 * 2, 2):
            xcos = (rd[i, i] + rd[i + 1, i + 1]) * 0.5
            xsin = (rd[i, i + 1] - rd[i + 1, i]) * 0.5
            theta = np.arctan2(xsin, xcos)
            ld[i, i + 1] = theta
            ld[i + 1, i] = -theta
        return rv @ ld @ rv.T

def bccd_kernel_(mycc, u=None, conv_tol_normu=1e-5, max_cycle=20, diis=True,
                 canonicalization=True, verbose=4):
    """
    Brueckner coupled-cluster wrapper, using an outer-loop algorithm.

    Args:
        mycc: a converged CCSD object.
        u: initial transformation matrix.
        conv_tol_normu: convergence tolerance for u matrix.
        max_cycle: Maximum number of BCC cycles.
        diis: whether perform DIIS.
        canonicalization: whether to semi-canonicalize the Brueckner orbitals.
        verbose: verbose for CCSD inner iterations.

    Returns:
        mycc: a modified CC object with t1 vanished.
              mycc._scf and mycc will be modified.
    """
    log = lib.logger.new_logger(mycc, verbose)
    log.info("BCCD loop starts.")

    def trans_mo(mo_coeff, u):
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim == 2:
            res = np.dot(mo_coeff, u)
        else:
            spin, nao, nmo = mo_coeff.shape
            res = np.zeros((spin, nao, nmo), dtype=mo_coeff.dtype)
            for s in range(spin):
                res[s] = np.dot(mo_coeff[s], u[s])
        return res

    def u2A(u):
        if u.ndim == 2:
            if la.det(u) < 0:
                u[:, 0] *= -1
            A = logm(u)
        else:
            if la.det(u[0]) < 0:
                u[0][:, 0] *= -1
            A_a = logm(u[0])
            if la.det(u[1]) < 0:
                u[1][:, 0] *= -1
            A_b = logm(u[1])
            A = np.asarray((A_a, A_b))
        return A

    def A2u(A):
        if A.ndim == 2:
            u = la.expm(A)
        else:
            u_a = la.expm(A[0])
            u_b = la.expm(A[1])
            u = np.asarray((u_a, u_b))
        return u

    mf = mycc._scf
    ovlp = mf.get_ovlp()
    adiis = lib.diis.DIIS()
    frozen = mycc.frozen
    level_shift = mycc.level_shift
    frozen_mask = mycc.get_frozen_mask()

    if u is None:
        u = get_umat_from_t1(mycc.t1)

    mo_coeff_new = np.array(mycc.mo_coeff, copy=True)

    if u.ndim == 2:
        mo_coeff_ref = np.array(mycc.mo_coeff[:, frozen_mask], copy=True)
        u_tot = np.eye(u.shape[-1], dtype=mo_coeff_ref.dtype)
    else:
        mo_coeff_ref = np.array((mycc.mo_coeff[0][:, frozen_mask[0]],
                                 mycc.mo_coeff[1][:, frozen_mask[1]]),
                                 copy=True)
        u_tot = np.asarray((np.eye(u.shape[-1]), np.eye(u.shape[-1])),
                           dtype=mo_coeff_ref.dtype)

    with lib.temporary_env(mf, verbose=verbose):
        e_tot_last = mycc.e_tot
        for i in range(max_cycle):
            u_tot = trans_mo(u_tot, u)
            if diis:
                A = u2A(u_tot)
                if u_tot.ndim == 2:
                    A = adiis.update(A, xerr=mycc.t1)
                    u_tot = A2u(A)
                    mo_xcore = trans_mo(mo_coeff_ref, u_tot)
                    u = get_mo_ovlp(mf.mo_coeff[:, frozen_mask], mo_xcore, ovlp)
                    mo_coeff_new[:, frozen_mask] = mo_xcore
                else:
                    t1_ravel = np.hstack((mycc.t1[0].ravel(), mycc.t1[1].ravel()))
                    A = adiis.update(A, xerr=t1_ravel)
                    u_tot = A2u(A)
                    u = []
                    for s in range(2):
                        mo_xcore = trans_mo(mo_coeff_ref[s], u_tot[s])
                        u.append(get_mo_ovlp(mf.mo_coeff[s][:, frozen_mask[s]], mo_xcore, ovlp))
                        mo_coeff_new[s][:, frozen_mask[s]] = mo_xcore
                    u = np.asarray(u)
            else:
                mo_coeff_new = trans_mo(mo_coeff_ref, u_tot)
            mf.mo_coeff = mo_coeff_new

            mf.e_tot = mf.energy_tot()
            t1 = transform_t1_to_bo(mycc.t1, u)
            t2 = transform_t2_to_bo(mycc.t2, u)

            mycc.__init__(mf)
            mycc.frozen = frozen
            mycc.level_shift = level_shift
            mycc.verbose = verbose

            mycc.kernel(t1=t1, t2=t2)
            dE = mycc.e_tot - e_tot_last
            e_tot_last = mycc.e_tot
            if not mycc.converged:
                log.warn("CC not converged")
            if u_tot.ndim == 2:
                t1_norm = la.norm(mycc.t1)
            else:
                t1_ravel = np.hstack((mycc.t1[0].ravel(), mycc.t1[1].ravel()))
                t1_norm = la.norm(t1_ravel)

            log.info("BCC iter: %4d  E: %20.12f  dE: %12.3e  |t1|: %12.3e",
                     i, mycc.e_tot, dE, t1_norm)
            if t1_norm < conv_tol_normu:
                break
            u = get_umat_from_t1(mycc.t1)
        else:
            log.warn("BCC: not converged, max_cycle reached.")

    # semi-canonicalization
    if canonicalization:
        dm = mf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mf.get_veff(mycc.mol, dm)
        fockao = mf.get_fock(vhf=vhf, dm=dm)
        e_corr = mycc.e_corr

        if u.ndim == 2:
            fock = mycc.mo_coeff.conj().T @ fockao @ mycc.mo_coeff
            fock_xcore = fock[np.ix_(frozen_mask, frozen_mask)]
            foo = fock_xcore[:mycc.nocc, :mycc.nocc]
            fvv = fock_xcore[mycc.nocc:, mycc.nocc:]
            ew_o, ev_o = la.eigh(foo)
            ew_v, ev_v = la.eigh(fvv)
            umat_xcore = la.block_diag(ev_o, ev_v)
            umat = np.eye(mycc.mo_coeff.shape[-1])
            umat[np.ix_(frozen_mask, frozen_mask)] = umat_xcore
            mf.mo_coeff = mf.mo_coeff @ umat
            mycc.mo_coeff = mf.mo_coeff
        else:
            umat = []
            umat_xcore = []
            for s in range(2):
                fock = mycc.mo_coeff[s].conj().T @ fockao[s] @ mycc.mo_coeff[s]
                fock_xcore = fock[np.ix_(frozen_mask[s], frozen_mask[s])]
                foo = fock_xcore[:mycc.nocc[s], :mycc.nocc[s]]
                fvv = fock_xcore[mycc.nocc[s]:, mycc.nocc[s]:]
                ew_o, ev_o = la.eigh(foo)
                ew_v, ev_v = la.eigh(fvv)
                umat_xcore.append(la.block_diag(ev_o, ev_v))
                umat_s = np.eye(mycc.mo_coeff[s].shape[-1])
                umat_s[np.ix_(frozen_mask[s], frozen_mask[s])] = umat_xcore[-1]
                umat.append(umat_s)

            umat = np.asarray(umat)
            mf.mo_coeff = np.einsum('spm, smn -> spn', mf.mo_coeff, umat)
            mycc.mo_coeff = mf.mo_coeff

        t1 = transform_t1_to_bo(mycc.t1, umat_xcore)
        t2 = transform_t2_to_bo(mycc.t2, umat_xcore)

        mf.e_tot = mf.energy_tot()
        mycc.__init__(mf)
        mycc.e_hf = mycc.get_e_hf()
        mycc.e_corr = e_corr
        mycc.frozen = frozen
        mycc.level_shift = level_shift
        mycc.verbose = verbose
        mycc.t1 = t1
        mycc.t2 = t2

    return mycc
