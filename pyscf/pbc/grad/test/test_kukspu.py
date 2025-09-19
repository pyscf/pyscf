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
import pyscf
from pyscf import lib
from pyscf.pbc.dft import kukspu
from pyscf.pbc.grad import kukspu as kukspu_grad
from pyscf.data.nist import BOHR

class KnownValues(unittest.TestCase):
    def test_finite_diff_hubbard_U_grad(self):
        cell = pyscf.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            spin = 2,
            pseudo = 'gth-pbe')
        kpts = cell.make_kpts([3,1,1])
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = kukspu.KUKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao)
        mf.__dict__.update(cell.KUKS(kpts=kpts).run(max_cycle=1).__dict__)
        de = kukspu_grad._hubbard_U_deriv1(mf)

        mf.cell.set_geom_('C 0 0 0 0; O 0.5 0.801 1.1')
        e1 = mf.get_veff().E_U.real

        mf.cell.set_geom_('C 0 0 0 0; O 0.5 0.799 1.1')
        e2 = mf.get_veff().E_U.real
        self.assertAlmostEqual(de[1,1], (e1 - e2)/2e-3*BOHR, 6)

    def test_finite_diff_kukspu_grad(self):
        cell = pyscf.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; O 0.5,  0.8,  1.1',
            a = '''0.      1.7834  1.7834
                   1.7834  0.      1.7834
                   1.7834  1.7834  0.    ''',
            basis = [[0, [1.3, 1]], [1, [0.8, 1]]],
            pseudo = 'gth-pbe')
        kpts = cell.make_kpts([3,1,1])
        minao = 'gth-szv'

        U_idx = ['C 2p']
        U_val = [5]
        mf = kukspu.KUKSpU(cell, kpts=kpts, U_idx=U_idx, U_val=U_val, minao_ref=minao)
        e, g = mf.nuc_grad_method().as_scanner()(cell)
        self.assertAlmostEqual(e, -15.939464667807849, 8)
        self.assertAlmostEqual(lib.fp(g), -0.42370983409650914, 5)

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(cell.set_geom_('C 0 0 0 0; O 0.5 0.801 1.1'))
        e2 = mf_scanner(cell.set_geom_('C 0 0 0 0; O 0.5 0.799 1.1'))
        self.assertAlmostEqual(g[1,1], (e1-e2)/2e-3*BOHR, 5)

if __name__ == '__main__':
    print("Full Tests for KUKS+U Gradients")
    unittest.main()
