#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from functools import reduce
import os
import numpy
from pyscf import gto
from pyscf import lo
from pyscf import scf
from pyscf.mp import cabs


def setUpModule():
    global mol, mf, mf1
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_find_cabs(self):
        auxmol = mol.copy()
        auxmol.basis = 'def2-tzvp'
        auxmol.build(False, False)
        cabs_mol, cabs_coeff = cabs.find_cabs(mol, auxmol)
        nao = mol.nao_nr()
        nca = cabs_coeff.shape[0]
        c1 = numpy.zeros((nca,nao))
        c1[:nao,:nao] = lo.orth.lowdin(mol.intor('int1e_ovlp_sph'))
        c = numpy.hstack((c1,cabs_coeff))
        s = reduce(numpy.dot, (c.T, cabs_mol.intor('int1e_ovlp_sph'), c))
        self.assertAlmostEqual(numpy.linalg.norm(s-numpy.eye(c.shape[1])), 0, 8)

    def test_rhf_cabs_singles(self):
        mol = gto.Mole(atom='''
            H   0.000000000   0.000000000   0.457870600
            F   0.000000000   0.000000000  -0.457870600
        ''', basis='cc-pvdz', verbose=0)
        mol.build()
        mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit')
        # MolPro: -0.033551214
        self.assertAlmostEqual(e, -0.033551214448, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=[0])
        self.assertAlmostEqual(e, -0.033551214448, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=0)
        # MRCC:   -0.033749160781
        self.assertAlmostEqual(e, -0.033749161358, 9)

    def test_uhf_cabs_singles(self):
        mol = gto.Mole(atom='''
            H   0.000000000   0.000000000   0.457870600
            O   0.000000000   0.000000000  -0.457870600
        ''', basis='cc-pvdz', spin=1, verbose=0)
        mol.build()
        mf = scf.UHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit')
        # no MolPro, it has only ROHF-MP2
        self.assertAlmostEqual(e, -0.022958881659, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=([0], [0]))
        self.assertAlmostEqual(e, -0.022958881659, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=0)
        # MRCC:   -0.023166900284
        self.assertAlmostEqual(e, -0.023166901553, 9)

    def test_rohf_cabs_singles(self):
        mol = gto.Mole(atom='''
            H   0.000000000   0.000000000   0.457870600
            O   0.000000000   0.000000000  -0.457870600
        ''', basis='cc-pvdz', spin=1, verbose=0)
        mol.build()
        mf = scf.ROHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').run()
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit')
        # MolPro:
        #                                       TOTAL          ALPHA          BETA
        # Singles Contributions MO        -0.002681874   -0.001339551   -0.001342323
        # Singles Contributions CABS      -0.022664571   -0.013326055   -0.009338516
        # Pure DF-RHF relaxation          -0.022020485
        #
        # One has to sum MO + CABS contributions:
        # -0.002681874 + -0.022664571 = -0.025346445
        self.assertAlmostEqual(e, -0.025343529823, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=([0], [0]))
        self.assertAlmostEqual(e, -0.025343529823, 9)
        e = cabs.energy_singles(mf, 'cc-pvdz-jkfit', frozen=0)
        # MolPro:
        #                                       TOTAL          ALPHA          BETA
        # Singles Contributions MO        -0.002709569   -0.001352954   -0.001356615
        # Singles Contributions CABS      -0.022873615   -0.013438077   -0.009435538
        # Pure DF-RHF relaxation          -0.022198754
        # Sum: -0.025583184
        self.assertAlmostEqual(e, -0.025580122540, 9)


if __name__ == "__main__":
    print("Full Tests for CABS")
    unittest.main()
