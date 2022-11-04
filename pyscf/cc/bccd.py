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

import numpy
from scipy import linalg as la

from pyscf import lib

def get_umat_from_t1(t1):
    """
    Get rotation matrix from t1.
    """
    if isinstance(t1, numpy.ndarray) and t1.ndim == 2:
        nocc, nvir = t1.shape
        amat = numpy.zeros((nocc+nvir, nocc+nvir), dtype=t1.dtype)
        amat[:nocc, -nvir:] = -t1
        amat[-nvir:, :nocc] = t1.conj().T
        umat = la.expm(amat)
    else: # UHF
        spin = len(t1)
        nmo = numpy.sum(t1[0].shape)
        umat = numpy.zeros((spin, nmo, nmo), dtype=numpy.result_type(*t1))
        for s in range(spin):
            nocc, nvir = t1[s].shape
            amat = numpy.zeros((nmo, nmo), dtype=t1[s].dtype)
            amat[:nocc, -nvir:] = -t1[s]
            amat[-nvir:, :nocc] = t1[s].conj().T
            umat[s] = la.expm(amat)
    return umat

def transform_t1_to_bo(t1, umat):
    """
    Transform t1 to brueckner orbital basis.
    """
    if isinstance(t1, numpy.ndarray) and t1.ndim == 2:
        nocc, nvir = t1.shape
        umat_occ = umat[:nocc, :nocc]
        umat_vir = umat[nocc:, nocc:] 
        return reduce(numpy.dot, (umat_occ.conj().T, t1, umat_vir))
    else: # UHF
        spin = len(t1)
        return [transform_t1_to_bo(t1[s], umat[s]) for s in range(spin)]

def transform_t2_to_bo(t2, umat, umat_b=None):
    """
    Transform t2 to brueckner orbital basis.
    """
    if isinstance(t2, numpy.ndarray) and t2.ndim == 4:
        umat_a = umat
        if umat_b is None:
            umat_b = umat_a

        nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
        umat_occ_a = umat_a[:nocc_a, :nocc_a]
        umat_occ_b = umat_b[:nocc_b, :nocc_b]
        umat_vir_a = umat_a[nocc_a:, nocc_a:]
        umat_vir_b = umat_b[nocc_b:, nocc_b:]
        t2_bo = numpy.einsum("ijab, iI, jJ, aA, bB -> IJAB", t2, umat_occ_a,
                             umat_occ_b, umat_vir_a, umat_vir_b, optimize=True)
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
    ovlp = numpy.asarray(ovlp)
    mo1 = numpy.asarray(mo1)
    mo2 = numpy.asarray(mo2)
    if mo1.ndim == 2:
        res = reduce(numpy.dot, (mo1.conj().T, ovlp, mo2))
    else:
        assert mo1.shape[0] == mo2.shape[0]
        spin, nao, nmo1 = mo1.shape
        nmo2 = mo2.shape[-1]
        res = numpy.zeros((spin, nmo1, nmo2), dtype=numpy.result_type(mo1, mo2))
        for s in range(spin):
            res[s] = reduce(numpy.dot, (mo1[s].conj().T, ovlp, mo2[s]))
    return res

def logm(mrot):
    rs = mrot + mrot.T
    rl, rv = la.eigh(rs)
    rd = rv.T @ mrot @ rv
    ra, rdet = 1, rd[0, 0]
    for i in range(1, len(rd)):
        ra, rdet = rdet, rd[i, i] * rdet - \
            rd[i - 1, i] * rd[i, i - 1] * ra
    assert rdet > 0
    ld = numpy.zeros_like(rd)
    for i in range(0, len(rd) // 2 * 2, 2):
        xcos = (rd[i, i] + rd[i + 1, i + 1]) * 0.5
        xsin = (rd[i, i + 1] - rd[i + 1, i]) * 0.5
        theta = numpy.arctan2(xsin, xcos)
        ld[i, i + 1] = theta
        ld[i + 1, i] = -theta
    return rv @ ld @ rv.T

def bccd_kernel_(mycc, u=None, conv_tol_normu=1e-5, max_cycle=20, diis=True,
                 verbose=4):
    """
    Brueckner coupled-cluster wrapper, using an outer-loop algorithm.
    
    Args:
        mycc: a converged CCSD object.
        u: initial transformation matrix.
        conv_tol_normu: convergence tolerance for u matrix.
        max_cycle: Maximum number of BCC cycles.
        diis: whether perform DIIS.
        verbose: verbose for CCSD inner iterations.

    Returns:
        mycc: a modified CC object with t1 vanished.
    """ 
    log = lib.logger.new_logger(mycc, verbose)
    log.info("BCCD loop starts.")
    
    def trans_mo(mo_coeff, u):
        mo_coeff = numpy.asarray(mo_coeff)
        if mo_coeff.ndim == 2:
            res = numpy.dot(mo_coeff, u)
        else:
            spin, nao, nmo = mo_coeff.shape
            res = numpy.zeros((spin, nao, nmo), dtype=mo_coeff.dtype)
            for s in range(spin):
                res[s] = numpy.dot(mo_coeff[s], u[s])
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
            A = numpy.asarray((A_a, A_b))
        return A
    
    def A2u(A):
        if A.ndim == 2:
            u = la.expm(A)
        else:
            u_a = la.expm(A[0])
            u_b = la.expm(A[1])
            u = numpy.asarray((u_a, u_b))
        return u
    
    mf = mycc._scf
    ovlp = mf.get_ovlp()
    adiis = lib.diis.DIIS()
    frozen = mycc.frozen
    level_shift = mycc.level_shift
    
    if u is None:
        u = get_umat_from_t1(mycc.t1)
    mo_coeff_ref = numpy.array(mf.mo_coeff, copy=True)
    if u.ndim == 2:
        u_tot = numpy.eye(u.shape[-1])
    else:
        u_tot = numpy.asarray((numpy.eye(u.shape[-1]), numpy.eye(u.shape[-1])))

    with lib.temporary_env(mf, verbose=verbose):
        e_tot_last = mycc.e_tot
        for i in range(max_cycle):
            u_tot = trans_mo(u_tot, u)
            if diis:
                A = u2A(u_tot)
                if u_tot.ndim == 2:
                    A_new = adiis.update(A, xerr=mycc.t1)
                else:
                    t1_ravel = numpy.hstack((mycc.t1[0].ravel(), mycc.t1[1].ravel()))
                    A_new = adiis.update(A, xerr=t1_ravel)
                u_tot = A2u(A_new)
                mo_coeff_new = trans_mo(mo_coeff_ref, u_tot)
                u = get_mo_ovlp(mf.mo_coeff, mo_coeff_new, ovlp)
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
                t1_ravel = numpy.hstack((mycc.t1[0].ravel(), mycc.t1[1].ravel()))
                t1_norm = la.norm(t1_ravel)

            log.info("BCC iter: %4d  E: %20.12f  dE: %12.3e  |t1|: %12.3e", 
                     i, mycc.e_tot, dE, t1_norm)
            if t1_norm < conv_tol_normu:
                break
            u = get_umat_from_t1(mycc.t1)
        else:
            log.warn("BCC: not converged, max_cycle reached.")
    return mycc

if __name__ == "__main__":
    import pyscf
    from pyscf import cc
    
    numpy.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = 'ccpvdz',
        verbose = 4,
        spin = 0,
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    
    mycc = cc.CCSD(myhf)
    mycc.kernel()
    
    mycc = bccd_kernel_(mycc, diis=True, verbose=4)

    print (la.norm(mycc.t1))
    assert la.norm(mycc.t1) < 1e-5
