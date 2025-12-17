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
try:
    from pyscf.geomopt import ase_solver
except ImportError:
    ase_solver = False

@unittest.skipIf(not ase_solver, "ASE library not found.")
class KnownValues(unittest.TestCase):
    def test_ase_optimize_cell(self):
        cell = pyscf.M(
            atom='''
            C 0.  0.  0.
            C 1.9 1.9 1.9
            ''', a='''
            0. , 3.8, 3.8
            3.8, 0. , 3.8
            3.8, 3.8, 0.
            ''', basis=[[0, [1.5, 1]], [0, [.5, 1]], [1, [1.3, 1]], [1, [.6, 1]]],
            pseudo='gth-pade', mesh=[21]*3, unit='Bohr',
            output='/dev/null', verbose=5)

        mf = cell.KRKS(xc='pbe')
        opt = mf.Gradients().optimizer().run()
        cell = opt.cell
        a = cell.lattice_vectors()
        atom_coords = cell.atom_coords()
        assert abs(atom_coords[0,0]) < 1e-5
        assert abs(atom_coords[1,0] - 1.8555577) < 5e-4
        assert abs(atom_coords[1,0]*2 - a[0,1]) < 1e-7

    def test_ase_optimize_mol(self):
        mol = pyscf.M(
            atom = '''
            O      0.000    0.    0.
            H     -0.757    0.    0.58
            H      0.757    0.    0.58
            ''', basis='def2-svp', output='/dev/null', verbose=5)

        mf = mol.RHF().density_fit()
        opt = mf.Gradients().optimizer(solver='ase').run()
        mol = opt.mol
        atom_coords = mol.atom_coords()
        assert abs(atom_coords[2,0] - 1.42162605) < 1e-5

if __name__ == "__main__":
    print("Tests for ase_solver")
    unittest.main()
