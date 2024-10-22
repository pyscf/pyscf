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
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.mp.dfump2_native import DFUMP2, SCSUMP2


def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = '''
    O    0.000   0.000  -1.141
    O    0.000   0.000   1.141
    '''
    mol.unit = 'Bohr'
    mol.basis = 'def2-SVP'
    mol.spin = 2
    mol.build()

    mf = scf.UHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf


def check_orth(obj, mol, mo_coeff, thresh=1.0e-12):
    sao = mol.intor_symmetric('int1e_ovlp')
    sno = numpy.linalg.multi_dot([mo_coeff.T, sao, mo_coeff])
    I = numpy.eye(mo_coeff.shape[1])
    obj.assertTrue(numpy.allclose(sno, I, atol=1.0e-12))


class KnownValues(unittest.TestCase):

    def setUp(self):
        self.assertTrue(mf.converged)
        self.mf = mf.copy()
        self.mf.mol = mf.mol.copy()
        self.mf.mo_coeff = mf.mo_coeff.copy()
        self.mf.mo_occ = mf.mo_occ.copy()
        self.mf.mo_energy = mf.mo_energy.copy()

    def test_energy(self):
        with DFUMP2(self.mf) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.347887316046, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -149.838271530605, delta=1.0e-8)

    def test_energy_fc(self):
        with DFUMP2(self.mf, frozen=2) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.343816318675, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -149.834200533235, delta=1.0e-8)

    def test_energy_fclist(self):
        self.mf.mo_coeff[0, :, [1, 3]] = self.mf.mo_coeff[0, :, [3, 1]]
        self.mf.mo_energy[0, [1, 3]] = self.mf.mo_energy[0, [3, 1]]
        self.mf.mo_coeff[1, :, [0, 4]] = self.mf.mo_coeff[1, :, [4, 0]]
        self.mf.mo_energy[1, [0, 4]] = self.mf.mo_energy[1, [4, 0]]
        with DFUMP2(self.mf, frozen=[[0, 3], [1, 4]]) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.343816318675, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -149.834200533235, delta=1.0e-8)

    def test_natorbs(self):
        mol = self.mf.mol
        with DFUMP2(self.mf) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 1.9999191951, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9473283296, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0168954406, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0168954406, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0478262909, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002326288, delta=1.0e-7)

    def test_natorbs_fc(self):
        mol = self.mf.mol
        with DFUMP2(self.mf, frozen=2) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbital orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 1.9999987581, delta=1.0e-7)
            self.assertAlmostEqual(natocc[1], 1.9999987356, delta=1.0e-7)
            self.assertAlmostEqual(natocc[2], 1.9882629065, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9473585838, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0168965649, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0168965649, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0477790944, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002307322, delta=1.0e-7)

    def test_natorbs_fclist(self):
        self.mf.mo_coeff[0, :, [1, 3]] = self.mf.mo_coeff[0, :, [3, 1]]
        self.mf.mo_energy[0, [1, 3]] = self.mf.mo_energy[0, [3, 1]]
        self.mf.mo_coeff[1, :, [0, 4]] = self.mf.mo_coeff[1, :, [4, 0]]
        self.mf.mo_energy[1, [0, 4]] = self.mf.mo_energy[1, [4, 0]]
        with DFUMP2(self.mf, frozen=[[0, 3], [1, 4]]) as pt:
            # also check the density matrix
            rdm1 = pt.make_rdm1()
            self.assertAlmostEqual(rdm1[0, 0, 0], 1.0, delta=1.0e-12)
            self.assertAlmostEqual(rdm1[0, 3, 3], 1.0, delta=1.0e-12)
            self.assertAlmostEqual(rdm1[1, 1, 1], 1.0, delta=1.0e-12)
            self.assertAlmostEqual(rdm1[1, 4, 4], 1.0, delta=1.0e-12)
            # now calculate the natural orbitals
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbital orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 1.9999987581, delta=1.0e-7)
            self.assertAlmostEqual(natocc[1], 1.9999987356, delta=1.0e-7)
            self.assertAlmostEqual(natocc[2], 1.9882629065, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9473585838, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0168965649, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0168965649, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0477790944, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002307322, delta=1.0e-7)

    def test_natorbs_relaxed(self):
        mol = self.mf.mol
        with DFUMP2(self.mf) as pt:
            pt.cphf_tol = 1e-12
            natocc, natorb = pt.make_natorbs(relaxed=True)
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 1.9999198031, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9478407509, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0169668947, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0169668947, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0453923546, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002225494, delta=1.0e-7)

    def test_natorbs_relaxed_fc(self):
        mol = self.mf.mol
        with DFUMP2(self.mf, frozen=2) as pt:
            pt.cphf_tol = 1e-12
            natocc, natorb = pt.make_natorbs(relaxed=True)
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbital orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 2.0000050774, delta=1.0e-7)
            self.assertAlmostEqual(natocc[1], 2.0000042352, delta=1.0e-7)
            self.assertAlmostEqual(natocc[2], 1.9889171379, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9478689720, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0169674773, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0169674773, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0453427169, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002207476, delta=1.0e-7)

    def test_natorbs_relaxed_fclist(self):
        self.mf.mo_coeff[0, :, [0, 5]] = self.mf.mo_coeff[0, :, [5, 0]]
        self.mf.mo_energy[0, [0, 5]] = self.mf.mo_energy[0, [5, 0]]
        self.mf.mo_coeff[1, :, [1, 4]] = self.mf.mo_coeff[1, :, [4, 1]]
        self.mf.mo_energy[1, [1, 4]] = self.mf.mo_energy[1, [4, 1]]
        mol = self.mf.mol
        with DFUMP2(self.mf, frozen=[[1, 5], [0, 4]]) as pt:
            pt.cphf_tol = 1e-12
            natocc, natorb = pt.make_natorbs(relaxed=True)
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbital orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[0], 2.0000050774, delta=1.0e-7)
            self.assertAlmostEqual(natocc[1], 2.0000042352, delta=1.0e-7)
            self.assertAlmostEqual(natocc[2], 1.9889171379, delta=1.0e-7)
            self.assertAlmostEqual(natocc[6], 1.9478689720, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0169674773, delta=1.0e-7)
            self.assertAlmostEqual(natocc[8], 1.0169674773, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0453427169, delta=1.0e-7)
            self.assertAlmostEqual(natocc[27], 0.0002207476, delta=1.0e-7)

    def test_memory(self):
        # Dummy class to set a very low memory limit.
        class fakeDFUMP2(DFUMP2):
            _mem_kb = 0
            @property
            def max_memory(self):
                return lib.current_memory()[0] + 1e-3 * self._mem_kb
            @max_memory.setter
            def max_memory(self, val):
                pass
        with fakeDFUMP2(self.mf) as pt:
            E, natocc_ur, natocc_re = None, None, None
            pt.cphf_tol = 1e-12
            # Try very low amounts of memory (in kB) until there is no failure.
            # Assume it should certainly work before 1 MB is reached.
            for m in range(8, 1000, 8):
                pt._mem_kb = m
                try:
                    E = pt.kernel()
                except MemoryError:
                    pass
                else:
                    break
            for m in range(4, 1000, 4):
                pt._mem_kb = m
                try:
                    natocc_ur = pt.make_natorbs()[0]
                except MemoryError:
                    pass
                else:
                    break
            for m in range(20, 1000, 20):
                pt._mem_kb = m
                try:
                    natocc_re = pt.make_natorbs(relaxed=True)[0]
                except MemoryError:
                    pass
                else:
                    break
        self.assertAlmostEqual(E, -0.347887316046, delta=1.0e-8)
        self.assertAlmostEqual(natocc_ur[6], 1.9473283296, delta=1.0e-7)
        self.assertAlmostEqual(natocc_ur[7], 1.0168954406, delta=1.0e-7)
        self.assertAlmostEqual(natocc_ur[9], 0.0478262909, delta=1.0e-7)
        self.assertAlmostEqual(natocc_re[6], 1.9478407509, delta=1.0e-7)
        self.assertAlmostEqual(natocc_re[7], 1.0169668947, delta=1.0e-7)
        self.assertAlmostEqual(natocc_re[9], 0.0453923546, delta=1.0e-7)

    def test_scs_energy(self):
        with SCSUMP2(self.mf) as pt:
            pt.kernel()
            self.assertAlmostEqual(pt.e_corr, -0.324631353397, delta=1.0e-8)
            self.assertAlmostEqual(pt.e_tot, -149.815015567956, delta=1.0e-8)

    def test_scs_natorbs(self):
        mol = self.mf.mol
        with SCSUMP2(self.mf) as pt:
            natocc, natorb = pt.make_natorbs()
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[6], 1.9512132898, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0092563934, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0451631109, delta=1.0e-7)

    def test_scs_natorbs_relaxed(self):
        mol = self.mf.mol
        with SCSUMP2(self.mf) as pt:
            pt.cphf_tol = 1e-12
            natocc, natorb = pt.make_natorbs(relaxed=True)
            # number of electrons conserved
            self.assertAlmostEqual(numpy.sum(natocc), mol.nelectron, delta=1.0e-10)
            # orbitals orthogonal
            check_orth(self, mol, natorb)
            # selected values
            self.assertAlmostEqual(natocc[6], 1.9516484920, delta=1.0e-7)
            self.assertAlmostEqual(natocc[7], 1.0093068572, delta=1.0e-7)
            self.assertAlmostEqual(natocc[9], 0.0434261850, delta=1.0e-7)


if __name__ == "__main__":
    print("Full Tests for native DF-UMP2")
    unittest.main()
