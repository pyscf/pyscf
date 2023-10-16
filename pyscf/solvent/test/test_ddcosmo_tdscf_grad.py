#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import solvent

def setUpModule():
    global mol0, mol1, mol2
    mol0 = gto.M(atom='H  0.  0.  1.804; F  0.  0.  0.', verbose=0, unit='B')
    mol1 = gto.M(atom='H  0.  0.  1.803; F  0.  0.  0.', verbose=0, unit='B')
    mol2 = gto.M(atom='H  0.  0.  1.805; F  0.  0.  0.', verbose=0, unit='B')

def tearDownModule():
    global mol0, mol1, mol2
    del mol0, mol1, mol2

class KnownValues(unittest.TestCase):

    def test_rhf_tda(self):
        # TDA with equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

        # TDA without equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_rhf_tdhf(self):
        # TDHF with equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

        # TDHF without equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDHF().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDHF().ddCOSMO().run()
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDHF().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_rhf_tda_triplet(self):
        # TDA triplet with equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run(singlet=False, equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run(singlet=False, equilibrium_solvation=True)
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run(singlet=False, equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

        # TDA triplet without equilibrium_solvation
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run(singlet=False)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run(singlet=False)
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run(singlet=False)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_uhf_tda(self):
        # TDA with equilibrium_solvation
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

        # TDA without equilibrium_solvation
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

    def test_uhf_tdhf(self):
        # TDHF with equilibrium_solvation
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDHF().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

        # TDHF without equilibrium_solvation
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDHF().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDHF().ddCOSMO().run()
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDHF().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

    def test_lda_tda(self):
        # TDA lda with equilibrium_solvation
        mf = mol0.RKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.RKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

        # TDA lda without equilibrium_solvation
        mf = mol0.RKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.RKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_b3lyp_tda(self):
        # TDA gga with equilibrium_solvation
        mf = mol0.RKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.RKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

        # TDA gga without equilibrium_solvation
        mf = mol0.RKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.RKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_ulda_tda(self):
        # TDA lda with equilibrium_solvation
        mf = mol0.UKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.UKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

        # TDA lda without equilibrium_solvation
        mf = mol0.UKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.UKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_ub3lyp_tda(self):
        # TDA gga with equilibrium_solvation
        mf = mol0.UKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        mf = mol2.UKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA().ddCOSMO().run(equilibrium_solvation=True)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 3)

        # TDA gga without equilibrium_solvation
        mf = mol0.UKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA().ddCOSMO().run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA().ddCOSMO().run()
        mf = mol2.UKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA().ddCOSMO().run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 3)


if __name__ == "__main__":
    print("Full Tests for ddcosmo TDDFT gradients")
    unittest.main()
