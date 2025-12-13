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
from pyscf import lib
from pyscf.pbc.dft.krkspu import _set_U, _make_minao_lo, reference_mol
from pyscf.pbc.grad import kuks as kuks_grad
from pyscf.pbc.grad.krkspu import generate_first_order_local_orbitals

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
    dm_deriv0 = [
        [C_k.dot(dm_k).dot(C_k.conj().T) for C_k, dm_k in zip(C_inv, dm_s)]
        for dm_s in dm
    ]
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
            for s in range(2):
                dm_deriv1 = lib.einsum('pj,xjq->xpq', C_inv[k].dot(dm[s][k]), SC1)
                i0 = i1 = 0
                for idx, val in zip(U_idx, U_val):
                    i0, i1 = i1, i1 + len(idx)
                    P0 = dm_deriv0[s][k][i0:i1,i0:i1]
                    P1 = dm_deriv1[:,i0:i1,i0:i1]
                    dE_U[atm_id] += weight * (val * 0.5) * (
                        np.einsum('xii->x', P1).real * 2 # *2 for P1+P1.T
                        - np.einsum('xij,ji->x', P1, P0).real * 4)
    return dE_U

class Gradients(kuks_grad.Gradients):
    def get_veff(self, dm=None, kpts=None):
        self._dE_U = _hubbard_U_deriv1(self.base, dm, kpts)
        return kuks_grad.get_veff(self, dm, kpts)

    def extra_force(self, atom_id, envs):
        val = super().extra_force(atom_id, envs)
        return self._dE_U[atom_id] + val
