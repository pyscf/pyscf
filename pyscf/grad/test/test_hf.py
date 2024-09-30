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
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import grad
from pyscf import cc
from pyscf.grad import rks, uks, roks

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [1   , (0. , 0.1, .817)],
        ["F" , (0. , 0. , 0.)], ]
    mol.basis = {"H": '6-31g',
                 "F": '6-31g',}
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def fp(mat):
    return abs(mat).sum()

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-14
        rhf.scf()
        g = grad.RHF(rhf)
        self.assertAlmostEqual(fp(g.grad_elec()), 7.9210392362911595, 6)
        self.assertAlmostEqual(fp(g.kernel()), 0.367743084803, 6)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(mol)
        uhf.conv_tol = 1e-10
        uhf.scf()
        g = grad.DHF(uhf)
        self.assertAlmostEqual(fp(g.grad_elec()), 7.9216825870803245, 6)
        g.level = 'LLLL'
        self.assertAlmostEqual(fp(g.grad_elec()), 7.924684281032623, 6)

    def test_energy_nuc(self):
        rhf = scf.RHF(mol)
        rhf.scf()
        g = grad.RHF(rhf)
        self.assertAlmostEqual(fp(g.grad_nuc()), 8.2887823210941249, 9)

    def test_ccsd(self):
        rhf = scf.RHF(mol)
        rhf.set(conv_tol=1e-10).scf()
        mycc = cc.CCSD(rhf)
        mycc.kernel()
        mycc.solve_lambda()
        g1 = mycc.nuc_grad_method().kernel()
        self.assertAlmostEqual(fp(g1), 0.43305028391866857, 6)

    def test_rhf_scanner(self):
        mol1 = mol.copy()
        mol1.set_geom_('''
        H   0.   0.   0.9
        F   0.   0.1  0.''')
        mf_scanner = grad.RHF(scf.RHF(mol).set(conv_tol=1e-14)).as_scanner()
        e, de = mf_scanner(mol)
        self.assertAlmostEqual(fp(de), 0.367743084803, 6)
        e, de = mf_scanner(mol1)
        self.assertAlmostEqual(fp(de), 0.041822093538, 6)

    def test_rks_scanner(self):
        mol1 = mol.copy()
        mol1.set_geom_('''
        H   0.   0.   0.9
        F   0.   0.1  0.''')
        mf_scanner = rks.Grad(scf.RKS(mol).set(conv_tol=1e-14)).as_scanner()
        e, de = mf_scanner(mol)
        self.assertAlmostEqual(fp(de), 0.458572523892797, 5)
        e, de = mf_scanner(mol1)
        self.assertAlmostEqual(fp(de), 0.12763259021187467, 5)

    def test_ccsd_scanner(self):
        mycc = cc.CCSD(scf.RHF(mol).set(conv_tol=1e-14))
        cc_scanner = mycc.nuc_grad_method().as_scanner()
        e, de = cc_scanner(mol)
        self.assertAlmostEqual(fp(de), 0.4330503011412547, 5)

        mol1 = gto.M(atom='''
        H   0.   0.   0.9
        F   0.   0.1  0.''', verbose=0)
        e, de = cc_scanner(mol1)
        self.assertAlmostEqual(fp(de), 0.2618586029073042, 5)


if __name__ == "__main__":
    print("Full Tests for HF")
    unittest.main()
