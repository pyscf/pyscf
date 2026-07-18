#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
Periodic KS-DFT with the pob-DZVP-rev2 / pob-TZVP-rev2 all-electron solid-state
basis sets (Oliveira, Peintinger, Laun, Bredow, J. Comput. Chem. 2019, 40, 2364).

System: LiH in the rocksalt structure (2-atom primitive cell), PBE, Gaussian
density fitting, 3x3x3 k-mesh.  The reference values below are pyscf regression
values.  They were cross-checked against TURBOMOLE riper (RI-J, jbas=universal)
on the identical cell and k-mesh: the two codes agree to sub-mHartree, confirming
the TURBOMOLE -> pyscf basis conversion.
'''

import unittest
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

# LiH rocksalt, conventional lattice constant a = 4.0834 Angstrom
A = 4.0834


def make_cell(basis):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.a = [[0., A / 2, A / 2],
              [A / 2, 0., A / 2],
              [A / 2, A / 2, 0.]]
    cell.atom = [['Li', [0.0, 0.0, 0.0]],
                 ['H',  [A / 2, 0.0, 0.0]]]
    cell.basis = basis
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()
    return cell


class KnownValues(unittest.TestCase):
    def test_pob_dzvp_rev2_krks_pbe(self):
        cell = make_cell('pob-DZVP-rev2')
        self.assertEqual(cell.nao_nr(), 11)   # Li(6) + H(5)
        mf = pbcdft.KRKS(cell, kpts=cell.make_kpts([3, 3, 3]))
        mf = mf.density_fit(auxbasis='weigend')
        mf.xc = 'PBE'
        mf.conv_tol = 1e-9
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(e, -8.127841401, 6)

    def test_pob_tzvp_rev2_krks_pbe(self):
        cell = make_cell('pob-TZVP-rev2')
        self.assertEqual(cell.nao_nr(), 13)   # Li(7) + H(6)
        mf = pbcdft.KRKS(cell, kpts=cell.make_kpts([3, 3, 3]))
        mf = mf.density_fit(auxbasis='weigend')
        mf.xc = 'PBE'
        mf.conv_tol = 1e-9
        e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(e, -8.129728186, 6)
        # triple-zeta must be variationally below double-zeta
        cell_dz = make_cell('pob-DZVP-rev2')
        mf_dz = pbcdft.KRKS(cell_dz, kpts=cell_dz.make_kpts([3, 3, 3]))
        mf_dz = mf_dz.density_fit(auxbasis='weigend')
        mf_dz.xc = 'PBE'
        mf_dz.conv_tol = 1e-9
        self.assertTrue(e < mf_dz.kernel())


if __name__ == '__main__':
    print("Full Tests for pob-*-rev2 periodic DFT")
    unittest.main()
