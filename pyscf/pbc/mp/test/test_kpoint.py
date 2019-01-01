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

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
import pyscf.pbc.mp
import pyscf.pbc.mp.kmp2


cell = pbcgto.Cell()
cell.unit = 'B'
L = 7
cell.atom.extend([['Be', (L/2.,  L/2., L/2.)]])
cell.a = 7 * np.identity(3)
cell.a[1,0] = 5.0

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade-q2'
cell.mesh = [12]*3
cell.verbose = 5
cell.output = '/dev/null'
cell.build()

def run_kcell(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

def run_kcell_complex(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    ekpt = kmf.scf()
    kmf.mo_coeff = [kmf.mo_coeff[i].astype(np.complex128) for i in range(np.prod(nk))]
    mp = pyscf.pbc.mp.kmp2.KMP2(kmf).run()
    return ekpt, mp.e_corr

class KnownValues(unittest.TestCase):
    def test_111(self):
        nk = (1, 1, 1)
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, -1.2061049658473704, 9)
        self.assertAlmostEqual(emp, -5.44597932944397e-06, 9)
        escf, emp = run_kcell_complex(cell,nk)
        self.assertAlmostEqual(emp, -5.44597932944397e-06, 9)

    def test_311_high_cost(self):
        nk = (3, 1, 1)
        escf, emp = run_kcell(cell,nk)
        self.assertAlmostEqual(escf, -1.0585001200928885, 9)
        self.assertAlmostEqual(emp, -7.9832274354253814e-06, 9)

    def test_h4_fcc_k2_frozen(self):
        '''Metallic hydrogen fcc lattice with frozen lowest lying occupied
        and highest lying virtual orbitals.  Checks versus a corresponding
        supercell calculation.

        NOTE: different versions of the davidson may converge to a different
        solution for the k-point IP/EA eom.  If you're getting the wrong
        root, check to see if it's contained in the supercell set of
        eigenvalues.'''
        cell = pbcgto.Cell()
        cell.atom = [['H', (0.000000000, 0.000000000, 0.000000000)],
                     ['H', (0.000000000, 0.500000000, 0.250000000)],
                     ['H', (0.500000000, 0.500000000, 0.500000000)],
                     ['H', (0.500000000, 0.000000000, 0.750000000)]]
        cell.unit = 'Bohr'
        cell.a = [[1.,0.,0.],[0.,1.,0],[0,0,2.2]]
        cell.verbose = 7
        cell.spin = 0
        cell.charge = 0
        cell.basis = [[0, [1.0, 1]],]
        cell.pseudo = 'gth-pade'
        cell.output = '/dev/null'
        cell.max_memory = 1000
        for i in range(len(cell.atom)):
            cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
        cell.build()

        nmp = [2, 1, 1]

        kmf = pbcscf.KRHF(cell)
        kmf.kpts = cell.make_kpts(nmp, scaled_center=[0.0,0.0,0.0])
        e = kmf.kernel()

        frozen = [[0, 3], []]
        mymp = pyscf.pbc.mp.kmp2.KMP2(kmf, frozen=frozen)
        ekmp2, _ = mymp.kernel()
        self.assertAlmostEqual(ekmp2, -0.022416773725207319, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        mysmp = pyscf.pbc.mp.kmp2.KMP2(mf, frozen=[0, 7])
        emp2, _ = mysmp.kernel()
        emp2 /= np.prod(nmp)
        self.assertAlmostEqual(emp2, -0.022416773725207319, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

