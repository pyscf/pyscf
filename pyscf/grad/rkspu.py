#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Analytical derivatives for DFT+U
'''

import numpy as np
import scipy.linalg as la
from pyscf import lib
from pyscf import gto
from pyscf.grad import rks as rks_grad
from pyscf.dft.rkspu import _set_U, _make_minao_lo, reference_mol

def generate_first_order_local_orbitals(mol, minao_ref='MINAO'):
    if isinstance(minao_ref, str):
        pmol = reference_mol(mol, minao_ref)
    else:
        pmol = minao_ref
    sAA = mol.intor('int1e_ovlp', hermi=1)
    sAB = gto.intor_cross('int1e_ovlp', mol, pmol)
    sAA_cd = la.cho_factor(sAA)
    C0_minao = la.cho_solve(sAA_cd, sAB)

    # Lowdin orthogonalization coefficients = S^{-1/2}
    S0 = sAB.conj().T.dot(C0_minao)
    w2, v = la.eigh(S0)
    w = np.sqrt(w2)
    S0_lowdin = (v/w).dot(v.conj().T)

    sAA_ip1 = mol.intor('int1e_ipovlp')
    sAB_ip1 = gto.intor_cross('int1e_ipovlp', mol, pmol)
    sBA_ip1 = gto.intor_cross('int1e_ipovlp', pmol, mol)

    nao, n_minao = C0_minao.shape
    aoslice = mol.aoslice_by_atom()
    minao_slice = pmol.aoslice_by_atom()

    def make_coeff(atm_id):
        p0, p1 = aoslice[atm_id,2:]
        q0, q1 = minao_slice[atm_id,2:]
        C1 = np.empty((3, nao, n_minao))
        for n in range(3):
            sAA1 = np.zeros((nao, nao))
            sAA1[p0:p1,:] -= sAA_ip1[n,p0:p1]
            sAA1[:,p0:p1] -= sAA_ip1[n,p0:p1].conj().T
            sAB1 = np.zeros((nao, n_minao))
            sAB1[p0:p1,:] -= sAB_ip1[n,p0:p1]
            sAB1[:,q0:q1] -= sBA_ip1[n,q0:q1].conj().T

            # The first order of A = S^{-1/2}
            # A S A = 1
            # A1 S0 A0 + A0 S1 A0 + A0 S0 A1 = 0
            # inv(A0) A1 S0 + S1 + S0 A1 inv(A0) = 0
            # A0 = (U/w) U^T = U (U/w)^T
            # U (U w)^T A1 U w^2 U^T + U w^2 U^T A1 U w U^T = -S1
            # (Uw)^T A1 Uw = - U^T S1 U / (w[:,None] + w)
            S1 = sAB1.conj().T.dot(C0_minao)
            S1 = S1 + S1.conj().T
            S1 -= C0_minao.conj().T.dot(sAA1).dot(C0_minao)
            S1 = v.conj().T.dot(-S1).dot(v)
            S1 /= (w[:,None] + w)
            vw = v / w
            S1_lowdin = vw.dot(S1).dot(vw.conj().T)

            C1_minao = la.cho_solve(sAA_cd, sAB1 - sAA1.dot(C0_minao))
            C1[n] = C1_minao.dot(S0_lowdin)
            C1[n] += C0_minao.dot(S1_lowdin)
        return C1
    return make_coeff

def _hubbard_U_deriv1(mf, dm=None):
    assert mf.alpha is None
    assert mf.C_ao_lo is None
    assert mf.minao_ref is not None
    if dm is None:
        dm = mf.make_rdm1()

    mol = mf.mol
    # Construct orthogonal minao local orbitals.
    pmol = reference_mol(mol, mf.minao_ref)
    C_ao_lo = _make_minao_lo(mol, pmol)
    U_idx, U_val = _set_U(mol, pmol, mf.U_idx, mf.U_val)[:2]
    U_idx_stack = np.hstack(U_idx)
    C0 = C_ao_lo[:,U_idx_stack]

    ovlp0 = mf.get_ovlp()
    C_inv = np.dot(C0.conj().T, ovlp0)
    dm_deriv0 = C_inv.dot(dm).dot(C_inv.conj().T)
    ovlp1 = mol.intor('int1e_ipovlp')
    f_local_ao = generate_first_order_local_orbitals(mol, pmol)

    ao_slices = mol.aoslice_by_atom()
    natm = mol.natm
    dE_U = np.zeros((natm, 3))
    for atm_id, (p0, p1) in enumerate(ao_slices[:,2:]):
        C1 = f_local_ao(atm_id)[:,:,U_idx_stack]
        SC1 = lib.einsum('pq,xqi->xpi', ovlp0, C1)
        SC1 -= lib.einsum('xqp,qi->xpi', ovlp1[:,p0:p1], C0[p0:p1])
        SC1[:,p0:p1] -= lib.einsum('xpq,qi->xpi', ovlp1[:,p0:p1], C0)
        dm_deriv1 = lib.einsum('pj,xjq->xpq', C_inv.dot(dm), SC1)
        i0 = i1 = 0
        for idx, val in zip(U_idx, U_val):
            i0, i1 = i1, i1 + len(idx)
            P0 = dm_deriv0[i0:i1,i0:i1]
            P1 = dm_deriv1[:,i0:i1,i0:i1]
            dE_U[atm_id] += (val * 0.5) * (
                np.einsum('xii->x', P1).real * 2 # *2 for P1+P1.T
                - np.einsum('xij,ji->x', P1, P0).real * 2)
    return dE_U

class Gradients(rks_grad.Gradients):
    def get_veff(self, mol=None, dm=None):
        self._dE_U = _hubbard_U_deriv1(self.base, dm)
        return rks_grad.get_veff(self, mol, dm)

    def extra_force(self, atom_id, envs):
        val = super().extra_force(atom_id, envs)
        return self._dE_U[atom_id] + val
