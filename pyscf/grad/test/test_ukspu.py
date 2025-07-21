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

import unittest
import numpy
from pyscf import gto, lib
from pyscf.dft import ukspu
from pyscf.grad import ukspu as ukspu_grad

class KnownValues(unittest.TestCase):
    def test_finite_diff_hubbard_U_grad(self):
        mol = gto.M(atom='C 0 1.6 0; O 0 0 1', spin=2, basis='ccpvdz', unit='B')
        U_idx = ["C 2p"]
        U_val = [5.0]
        mf = ukspu.UKSpU(mol, U_idx=U_idx, U_val=U_val)
        mf.__dict__.update(mol.UHF().run().__dict__)
        de = ukspu_grad._hubbard_U_deriv1(mf)

        mf.mol.set_geom_('C 0 1.6 0; O 0 0 1.001', unit='B')
        e1 = mf.get_veff().E_U

        mf.mol.set_geom_('C 0 1.6 0; O 0 0 0.999', unit='B')
        e2 = mf.get_veff().E_U
        self.assertAlmostEqual(de[1,2], (e1 - e2)/2e-3, 6)

    def test_finite_diff_ukspu_grad(self):
        mol = gto.M(atom='C 0 0 0; O 1 2 1', basis='ccpvdz', unit='B', verbose=0)
        U_idx = ["C 2p"]
        U_val = [5.0]
        mf = ukspu.UKSpU(mol, xc='pbe', U_idx=U_idx, U_val=U_val)
        mol = gto.M(atom='C 0 1.6 0; O 0 0 1', spin=2, basis='ccpvdz', unit='B', verbose=0)
        e, g = mf.nuc_grad_method().as_scanner()(mol)
        self.assertAlmostEqual(e, -112.76620903336628, 8)
        self.assertAlmostEqual(lib.fp(g), -1.0568299810814519, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('C 0 1.6 0; O 0 0 1.001', unit='B'))
        e2 = mf_scanner(mol1.set_geom_('C 0 1.6 0; O 0 0 0.999', unit='B'))
        self.assertAlmostEqual(g[1,2], (e1-e2)/2e-3, 4)

if __name__ == '__main__':
    print("Full Tests for UKS+U Gradients")
    unittest.main()
