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
from pyscf import lib
from pyscf.dft.ukspu import UKSpU
from pyscf.dft.rkspu import _set_U, _make_minao_lo, reference_mol
from pyscf.grad import uks as uks_grad
from pyscf.grad.rkspu import generate_first_order_local_orbitals

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
    dm_deriv0 = [C_inv.dot(dm[0]).dot(C_inv.conj().T),
                 C_inv.dot(dm[1]).dot(C_inv.conj().T)]
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
        for s in range(2):
            dm_deriv1 = lib.einsum('pj,xjq->xpq', C_inv.dot(dm[s]), SC1)
            i0 = i1 = 0
            for idx, val in zip(U_idx, U_val):
                i0, i1 = i1, i1 + len(idx)
                P0 = dm_deriv0[s][i0:i1,i0:i1]
                P1 = dm_deriv1[:,i0:i1,i0:i1]
                dE_U[atm_id] += (val * 0.5) * (
                    np.einsum('xii->x', P1).real * 2 # *2 for P1+P1.T
                    - np.einsum('xij,ji->x', P1, P0).real * 4)
    return dE_U

class Gradients(uks_grad.Gradients):
    def get_veff(self, mol=None, dm=None):
        self._dE_U = _hubbard_U_deriv1(self.base, dm)
        return uks_grad.get_veff(self, mol, dm)

    def extra_force(self, atom_id, envs):
        val = super().extra_force(atom_id, envs)
        return self._dE_U[atom_id] + val
