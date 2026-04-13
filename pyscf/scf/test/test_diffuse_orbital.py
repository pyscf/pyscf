# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
import pytest
import numpy as np
import pyscf
from pyscf import lib
from pyscf import scf, dft

def setUpModule():
    global mol
    mol = pyscf.M(
        atom = """
            H 0 0 0
            Li 1.0 0.1 0
        """,
        basis = """
            X    S
                0.3425250914E+01       0.1543289673E+00
                0.6239137298E+00       0.5353281423E+00
                0.1688554040E+00       0.4446345422E+00
            X    S
                0.1611957475E+02       0.1543289673E+00
                0.2936200663E+01       0.5353281423E+00
                0.7946504870E+00       0.4446345422E+00
            X    P
                0.010000 1.0
            X    P
                0.010001 1.0
        """,
        verbose = 5,
        output = '/dev/null',
    )

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backup = pyscf.scf.hf.remove_overlap_zero_eigenvalue
        pyscf.scf.hf.remove_overlap_zero_eigenvalue = True

    @classmethod
    def tearDownClass(cls):
        pyscf.scf.hf.remove_overlap_zero_eigenvalue = cls.backup

    def test_rhf(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.670162135801041) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.53027311e-01,  2.53027311e-02,  1.78111017e-19],
            [-2.53027311e-01, -2.53027311e-02, -1.78111017e-19],
        ]))) < 1e-5

        dipole = mf.dip_moment()
        assert np.max(np.abs(dipole - np.array([4.26375987e+00, 4.26375987e-01, 1.86659164e-16]))) < 1e-4

        e, c = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
        assert abs(e - mf.mo_energy).max() < 5e-7
        f = mf.get_fock()
        e1 = lib.einsum('pi,pq,qi->i', c.conj(), f, c)
        assert abs(e - e1).max() < 1e-12

    def test_rhf_soscf(self):
        mf = dft.RKS(mol, xc = "pbe")
        mf.grids.atom_grid = (99,590)
        mf.conv_tol = 1e-10
        mf = mf.newton()
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.748401087498843) < 1e-5

        gobj = mf.Gradients()
        gradient = gobj.kernel()
        assert np.abs(gradient - np.array(
            [[ 2.44273951e-01,  2.44377010e-02,  6.79546462e-17],
             [-2.44288315e-01, -2.44313901e-02, -1.66959137e-16]])).max() < 1e-5

    def test_uhf(self):
        mf = dft.RKS(mol, xc = "PBE")
        mf.grids.atom_grid = (50,194)
        mf = mf.density_fit(auxbasis = "def2-universal-jkfit")
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.748763949503415) < 1e-5

        gobj = mf.Gradients()
        gobj.grid_response = True
        gradient = gobj.kernel()
        assert np.max(np.abs(gradient - np.array([
            [ 2.44275992e-01,  2.44757818e-02,  3.22713915e-19],
            [-2.44275992e-01, -2.44757818e-02, -3.17524970e-19],
        ]))) < 1e-5

    def test_rohf(self):
        mf = dft.ROKS(mol, xc = "PBE0")
        mf.grids.level = 3
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        assert mf.converged
        assert np.abs(energy - -7.749335934429277) < 1e-5


if __name__ == "__main__":
    print("Tests for System with Diffuse Orbitals (Ill-conditioned Overlap Matrices)")
    unittest.main()
