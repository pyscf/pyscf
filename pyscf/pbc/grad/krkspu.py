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
Analytical derivatives for DFT+U with kpoints sampling
'''

import numpy as np
import scipy.linalg as la
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.grad import krks as krks_grad
from pyscf.pbc.dft.krkspu import _set_U, _make_minao_lo, reference_mol

def generate_first_order_local_orbitals(cell, minao_ref='MINAO', kpts=None):
    kpts = kpts.reshape(-1, 3)
    nkpts = len(kpts)
    if isinstance(minao_ref, str):
        pcell = reference_mol(cell, minao_ref)
    else:
        pcell = minao_ref
    sAA = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    sAB = gto.intor_cross('int1e_ovlp', cell, pcell, kpts=kpts)

    C0_minao = []
    sAA_cd = []
    wv_ks = []
    S0_lowdin = []
    for k in range(nkpts):
        sAA_cd.append(la.cho_factor(sAA[k]))
        C0_minao.append(la.cho_solve(sAA_cd[k], sAB[k]))

        # Lowdin orthogonalization coefficients = S^{-1/2}
        S0 = sAB[k].conj().T.dot(C0_minao[k])
        w2, v = la.eigh(S0)
        w = np.sqrt(w2)
        wv_ks.append((w, v))
        S0_lowdin.append((v/w).dot(v.conj().T))

    sAA_ip1 = cell.pbc_intor('int1e_ipovlp', kpts=kpts)
    sAB_ip1 = gto.intor_cross('int1e_ipovlp', cell, pcell, kpts=kpts)
    sBA_ip1 = gto.intor_cross('int1e_ipovlp', pcell, cell, kpts=kpts)

    nao, n_minao = C0_minao[0].shape
    ao_slice = cell.aoslice_by_atom()
    minao_slice = pcell.aoslice_by_atom()
    dtype = np.result_type(*C0_minao)

    def make_coeff(atm_id):
        p0, p1 = ao_slice[atm_id,2:]
        q0, q1 = minao_slice[atm_id,2:]
        C1 = np.empty((nkpts, 3, nao, n_minao), dtype=dtype)
        for k in range(nkpts):
            w, v = wv_ks[k]
            for n in range(3):
                sAA1 = np.zeros((nao, nao), dtype=dtype)
                sAA1[p0:p1,:] -= sAA_ip1[k][n,p0:p1]
                sAA1[:,p0:p1] -= sAA_ip1[k][n,p0:p1].conj().T
                sAB1 = np.zeros((nao, n_minao), dtype=dtype)
                sAB1[p0:p1,:] -= sAB_ip1[k][n,p0:p1]
                sAB1[:,q0:q1] -= sBA_ip1[k][n,q0:q1].conj().T

                S1 = C0_minao[k].conj().T.dot(sAB1)
                S1 = S1 + S1.conj().T
                S1 -= C0_minao[k].conj().T.dot(sAA1).dot(C0_minao[k])
                S1 = v.conj().T.dot(-S1).dot(v)
                S1 /= (w[:,None] + w)
                vw = v / w
                S1_lowdin = vw.dot(S1).dot(vw.conj().T)

                C1_minao = la.cho_solve(sAA_cd[k], sAB1 - sAA1.dot(C0_minao[k]))
                C1[k,n] = C1_minao.dot(S0_lowdin[k])
                C1[k,n] += C0_minao[k].dot(S1_lowdin)
        return C1
    return make_coeff

def _hubbard_U_deriv1(mf, dm=None, kpts=None):
    assert mf.alpha is None
    assert mf.C_ao_lo is None
    assert mf.minao_ref is not None
    if dm is None:
        dm = mf.make_rdm1()
    if kpts is None:
        kpts = mf.kpts.reshape(-1, 3)
    nkpts = len(kpts)
    cell = mf.cell

    # Construct orthogonal minao local orbitals.
    pcell = reference_mol(cell, mf.minao_ref)
    C_ao_lo = _make_minao_lo(cell, pcell, kpts=kpts)
    U_idx, U_val = _set_U(cell, pcell, mf.U_idx, mf.U_val)[:2]
    U_idx_stack = np.hstack(U_idx)
    C0 = [C_k[:,U_idx_stack] for C_k in C_ao_lo]

    ovlp0 = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    ovlp1 = cell.pbc_intor('int1e_ipovlp', kpts=kpts)
    C_inv = [C_k.conj().T.dot(S_k) for C_k, S_k in zip(C0, ovlp0)]
    dm_deriv0 = [C_k.dot(dm_k).dot(C_k.conj().T) for C_k, dm_k in zip(C_inv, dm)]
    f_local_ao = generate_first_order_local_orbitals(cell, pcell, kpts)

    ao_slices = cell.aoslice_by_atom()
    natm = cell.natm
    dE_U = np.zeros((natm, 3))
    weight = 1. / nkpts
    for atm_id, (p0, p1) in enumerate(ao_slices[:,2:]):
        C1 = f_local_ao(atm_id)
        for k in range(nkpts):
            C1_k = C1[k][:,:,U_idx_stack]
            SC1 = lib.einsum('pq,xqi->xpi', ovlp0[k], C1_k)
            SC1 -= lib.einsum('xqp,qi->xpi', ovlp1[k][:,p0:p1].conj(), C0[k][p0:p1])
            SC1[:,p0:p1] -= lib.einsum('xpq,qi->xpi', ovlp1[k][:,p0:p1], C0[k])
            dm_deriv1 = lib.einsum('pj,xjq->xpq', C_inv[k].dot(dm[k]), SC1)
            i0 = i1 = 0
            for idx, val in zip(U_idx, U_val):
                i0, i1 = i1, i1 + len(idx)
                P0 = dm_deriv0[k][i0:i1,i0:i1]
                P1 = dm_deriv1[:,i0:i1,i0:i1]
                dE_U[atm_id] += weight * (val * 0.5) * (
                    np.einsum('xii->x', P1).real * 2 # *2 for P1+P1.T
                    - np.einsum('xij,ji->x', P1, P0).real * 2)
    return dE_U

class Gradients(krks_grad.Gradients):
    def get_veff(self, dm=None, kpts=None):
        self._dE_U = _hubbard_U_deriv1(self.base, dm, kpts)
        return krks_grad.get_veff(self, dm, kpts)

    def extra_force(self, atom_id, envs):
        val = super().extra_force(atom_id, envs)
        return self._dE_U[atom_id] + val
