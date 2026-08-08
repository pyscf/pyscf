# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

try:
    from ase.build import bulk
    from pyscf.pbc.tools import pyscf_ase
    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


def _make_cell():
    '''Build a minimal diamond cell for testing via cell_from_ase.'''
    ase_atom = bulk('C', 'diamond', a=3.5668)
    cell = pyscf_ase.cell_from_ase(ase_atom)
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.build()
    return cell, ase_atom


def _make_symm_cell():
    '''Build a minimal diamond cell with space group symmetry enabled.'''
    ase_atom = bulk('C', 'diamond', a=3.5668)
    cell = pyscf_ase.cell_from_ase(ase_atom)
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.output = '/dev/null'
    cell.space_group_symmetry = True
    cell.symmorphic = False
    cell.build()
    return cell, ase_atom


@unittest.skipUnless(HAVE_ASE, "ASE is not installed")
class PySCFKptsWeightsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, cls.ase_atom = _make_cell()
        cls.symm_cell, cls.symm_ase_atom = _make_symm_cell()

    def _make_calc(self, kpts):
        '''Create a PySCF calculator wrapping a KRKS method.'''
        mf = self.cell.KRKS(xc='lda,vwn', kpts=kpts)
        self.ase_atom.calc = pyscf_ase.PySCF(method=mf)
        return self.ase_atom.calc

    def _make_symm_calc(self, kpts):
        '''Create a PySCF calculator using symmetry-enabled cell.'''
        mf = self.symm_cell.KRKS(xc='lda,vwn', kpts=kpts)
        self.symm_ase_atom.calc = pyscf_ase.PySCF(method=mf)
        return self.symm_ase_atom.calc

    def test_k_point_weights_no_symmetry(self):
        '''Uniform MP grid: all weights should be equal 1/nkpts.'''
        kpts = self.cell.make_kpts([2, 2, 2])
        calc = self._make_calc(kpts)
        w = calc.get_k_point_weights()
        nkpts = len(kpts)
        self.assertEqual(len(w), nkpts)
        self.assertAlmostEqual(np.sum(w), 1.0, 9)
        self.assertTrue(np.allclose(w, 1.0 / nkpts))

    def test_k_point_weights_with_symmetry(self):
        '''Symmetry-reduced k-points: weights should be non-uniform, sum to 1.'''
        kpts = self.symm_cell.make_kpts([2, 2, 2], space_group_symmetry=True)
        calc = self._make_symm_calc(kpts)
        w = calc.get_k_point_weights()
        self.assertGreater(len(w), 0)
        self.assertAlmostEqual(np.sum(w), 1.0, 9)

    def test_bz_k_points_no_symmetry(self):
        '''BZ k-points should be scaled coords matching the input grid.'''
        kpts = self.cell.make_kpts([2, 2, 2])
        calc = self._make_calc(kpts)
        bz_kpts = calc.get_bz_k_points()
        scaled = self.cell.get_scaled_kpts(kpts)
        self.assertEqual(bz_kpts.shape, (8, 3))
        self.assertTrue(np.allclose(bz_kpts, scaled))

    def test_bz_k_points_with_symmetry(self):
        '''BZ k-points from KPoints object should be full BZ scaled coords.'''
        kpts = self.symm_cell.make_kpts([2, 2, 2], space_group_symmetry=True)
        calc = self._make_symm_calc(kpts)
        bz_kpts = calc.get_bz_k_points()
        self.assertEqual(bz_kpts.shape, (kpts.nkpts, 3))
        self.assertTrue(np.allclose(bz_kpts, kpts.kpts_scaled))

    def test_bz_to_ibz_map_no_symmetry(self):
        '''Without symmetry, BZ-to-IBZ mapping should be identity.'''
        kpts = self.cell.make_kpts([2, 2, 2])
        calc = self._make_calc(kpts)
        bz2ibz = calc.get_bz_to_ibz_map()
        self.assertEqual(len(bz2ibz), 8)
        self.assertTrue(np.array_equal(bz2ibz, np.arange(8)))

    def test_bz_to_ibz_map_with_symmetry(self):
        '''With symmetry, BZ-to-IBZ mapping should match KPoints object.'''
        kpts = self.symm_cell.make_kpts([2, 2, 2], space_group_symmetry=True)
        calc = self._make_symm_calc(kpts)
        bz2ibz = calc.get_bz_to_ibz_map()
        self.assertEqual(len(bz2ibz), kpts.nkpts)
        self.assertTrue(np.array_equal(bz2ibz, kpts.bz2ibz))
        # Verify all IBZ indices are valid
        self.assertTrue(np.all(bz2ibz >= 0))
        self.assertTrue(np.all(bz2ibz < kpts.nkpts_ibz))


if __name__ == '__main__':
    print("Tests for pyscf_ase integration")
    unittest.main()
