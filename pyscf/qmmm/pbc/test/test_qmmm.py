#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Chenghan Li <lch004218@gmail.com>
#

import unittest
import numpy as np
import pyscf
from pyscf.dft import rks
from pyscf.qmmm.pbc import itrf

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

bas='3-21g'
auxbasis='cc-pvdz-jkfit'
scf_tol = 1e-10
max_scf_cycles = 50
grids_level = 3

def setUpModule():
    global mol
    mol = pyscf.M(atom=atom, basis=bas, max_memory=32000)
    mol.output = '/dev/null'
    mol.build()
    mol.verbose = 1

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def run_dft(xc):
    mf = rks.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf = itrf.add_mm_charges(
        mf, [[1,2,-1],[3,4,5]], np.eye(3)*15, [-5,5], [0.8,1.2], rcut_ewald=8, rcut_hcore=6)
    mf.conv_tol = scf_tol
    mf.max_cycle = max_scf_cycles
    mf.grids.level = grids_level
    e_dft = mf.kernel()

    g = mf.nuc_grad_method()
    g.max_memory = 32000
    g.auxbasis_response = True
    g_qm = g.kernel()

    g_mm = g.grad_nuc_mm() + g.grad_hcore_mm(mf.make_rdm1()) + g.de_ewald_mm
    return e_dft, g_qm, g_mm

class KnownValues(unittest.TestCase):
    def test_rks_pbe0(self):
        print('-------- RKS PBE0 -------------')
        e_tot, g_qm, g_mm = run_dft('PBE0')
        assert abs(e_tot - -76.00178807) < 1e-7
        assert abs(g_qm - np.array([[ 0.03002572,  0.13947702, -0.09234864],
                                    [-0.00462601, -0.04602809,  0.02750759],
                                    [-0.01821532, -0.18473378, 0.04189843]])).max() < 1e-6
        assert abs(g_mm - np.array([[-0.00914559,  0.08992359,  0.02114633],
                                    [ 0.00196155,  0.00136132, 0.00179565]])).max() < 1e-6

if __name__ == "__main__":
    print("Full Tests for QMMM PBC")
    unittest.main()
