#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
import copy
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.mp.dfmp2_fast import DFMP2


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = '''
C    0.000   0.000   1.266
C    0.000   0.000  -1.266
H    0.000   1.756   2.328
H    0.000  -1.756   2.328
H    0.000   1.756  -2.328
H    0.000  -1.756  -2.328
'''
mol.unit = 'Bohr'
mol.basis = 'def2-SVP'
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1.0e-12
mf.kernel()


def check_orth(obj, mol, mo_coeff, thresh=1.0e-12):
    sao = mol.intor_symmetric('int1e_ovlp')
    sno = numpy.linalg.multi_dot([mo_coeff.T, sao, mo_coeff])
    I = numpy.eye(mo_coeff.shape[1])
    obj.assertTrue(numpy.allclose(sno, I, atol=1.0e-12))


class KnownValues(unittest.TestCase):

    def setUp(self):
        self.assertTrue(mf.converged)
        self.mf = copy.copy(mf)
        self.mf.mol = mf.mol.copy()
        self.mf.mo_coeff = mf.mo_coeff.copy()
        self.mf.mo_occ = mf.mo_occ.copy()
        self.mf.mo_energy = mf.mo_energy.copy()

    def test_energy(self):
        with DFMP2(self.mf) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.280727901936, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -78.258178687361, delta=1.0e-8)

    def test_energy_fc(self):
        with DFMP2(self.mf, frozen=2) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.274767743344, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -78.252218528769, delta=1.0e-8)
    
    def test_energy_fclist(self):
        for arr in self.mf.mo_coeff.T, self.mf.mo_occ, self.mf.mo_energy:
            arr[[0, 2]] = arr[[2, 0]]
            arr[[1, 6]] = arr[[6, 1]]
        with DFMP2(self.mf, frozen=[2, 6]) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.274767743344, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -78.252218528769, delta=1.0e-8)
    
    def test_natorbs(self):
        mol = self.mf.mol
        with DFMP2(self.mf) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 1.9997941377, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.9384231532, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 0.0459829060, delta=1.0e-7)
            self.assertAlmostEqual(natocc[47], 0.0000761012, delta=1.0e-7)
    
    def test_natorbs_fc(self):
        mol = self.mf.mol
        with DFMP2(self.mf, frozen=2) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 2.0, delta=1.0e-12)
            self.assertAlmostEqual(natocc[1], 2.0, delta=1.0e-12)
            self.assertAlmostEqual(natocc[2], 1.9832413380, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.9384836199, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 0.0459325459, delta=1.0e-7)
            self.assertAlmostEqual(natocc[47], 0.0000751662, delta=1.0e-7)
    
    def test_natorbs_fclist(self):
        for arr in self.mf.mo_coeff.T, self.mf.mo_occ, self.mf.mo_energy:
            arr[[0, 5]] = arr[[5, 0]]
            arr[[1, 3]] = arr[[3, 1]]
        mol = self.mf.mol
        with DFMP2(self.mf, frozen=[3, 5]) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 2.0, delta=1.0e-12)
            self.assertAlmostEqual(natocc[1], 2.0, delta=1.0e-12)
            self.assertAlmostEqual(natocc[2], 1.9832413380, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.9384836199, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 0.0459325459, delta=1.0e-7)
            self.assertAlmostEqual(natocc[47], 0.0000751662, delta=1.0e-7)

    def test_memory(self):
        # Dummy class to set a very low memory limit.
        class fakeDFMP2(DFMP2):
            _mem_kb = 0
            @property
            def max_memory(self):
                return lib.current_memory()[0] + 1e-3 * self._mem_kb
            @max_memory.setter
            def max_memory(self, val):
                pass
        with fakeDFMP2(self.mf) as pt:
            E, natocc = None, None
            # Try very low amounts of memory (in kB) until there is no failure.
            # Assume it should certainly work before 1 MB is reached.
            for m in range(50, 1000, 50):
                pt._mem_kb = m
                try:
                    E = pt.kernel()
                except MemoryError:
                    pass
                else:
                    break
            for m in range(20, 1000, 20):
                pt._mem_kb = m
                try:
                    natocc = pt.make_natorbs()[0]
                except MemoryError:
                    pass
                else:
                    break
            self.assertAlmostEqual(E, -0.280727901936, delta=1.0e-8)
            self.assertAlmostEqual(natocc[7], 1.9384231532, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 0.0459829060, delta=1.0e-7)


if __name__ == "__main__":
    print("Full Tests for native DF-RMP2")
    unittest.main()
