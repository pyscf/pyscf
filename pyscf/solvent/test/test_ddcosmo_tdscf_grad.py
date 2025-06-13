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
from pyscf import tdscf

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
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_rhf_tdhf(self):
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDHF(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDHF(equilibrium_solvation=True).run()
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDHF(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_rhf_tda_triplet(self):
        mf = mol0.RHF().ddCOSMO().run()
        td = mf.TDA(equilibrium_solvation=True).run(singlet=False)
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RHF().ddCOSMO().run()
        td1 = mf.TDA(equilibrium_solvation=True).run(singlet=False)
        mf = mol2.RHF().ddCOSMO().run()
        td2 = mf.TDA(equilibrium_solvation=True).run(singlet=False)
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 6)

    def test_uhf_tda(self):
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

    def test_uhf_tdhf(self):
        mf = mol0.UHF().ddCOSMO().run()
        td = mf.TDHF(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UHF().ddCOSMO().run()
        td1 = mf.TDHF(equilibrium_solvation=True).run()
        mf = mol2.UHF().ddCOSMO().run()
        td2 = mf.TDHF(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 5)

    def test_lda_tda(self):
        mf = mol0.RKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.RKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_b3lyp_tda(self):
        mf = mol0.RKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.RKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.RKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_ulda_tda(self):
        mf = mol0.UKS().ddCOSMO().run(xc='svwn')
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='svwn')
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.UKS().ddCOSMO().run(xc='svwn')
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 4)

    def test_ub3lyp_tda(self):
        mf = mol0.UKS().ddCOSMO().run(xc='b3lyp')
        td = mf.TDA(equilibrium_solvation=True).run()
        g1 = td.nuc_grad_method().kernel()

        mf = mol1.UKS().ddCOSMO().run(xc='b3lyp')
        td1 = mf.TDA(equilibrium_solvation=True).run()
        mf = mol2.UKS().ddCOSMO().run(xc='b3lyp')
        td2 = mf.TDA(equilibrium_solvation=True).run()
        self.assertAlmostEqual((td2.e_tot[0]-td1.e_tot[0])/0.002, g1[0,2], 3)

    def test_scanner(self):
        mol = gto.M(atom='H 0 0 0; F .1 0 2.1', verbose=0, unit='B')
        td_ref = solvent.ddCOSMO(mol.RHF()).run().TDA(equilibrium_solvation=True).run()
        ref = td_ref.Gradients().kernel()
        td = mol0.RHF().ddCOSMO().run().TDA(equilibrium_solvation=True).Gradients()
        scan = td.as_scanner()
        e, de = scan('H 0 0 0; F .1 0 2.1')
        self.assertAlmostEqual(e, -98.20630203333705, 8)
        self.assertAlmostEqual(e, td_ref.e_tot[0], 7)
        self.assertAlmostEqual(de[0,0], 0.011059522, 5)
        self.assertAlmostEqual(abs(ref - de).max(), 0, 5)

        mol = gto.M(atom='H 0 0 0; F .1 0 2.1', verbose=0, unit='B')
        td_ref = solvent.ddPCM(mol.RHF()).run().TDA(equilibrium_solvation=True).run()
        ref = td_ref.Gradients().kernel()
        td = mol0.RHF().ddPCM().run().TDA(equilibrium_solvation=True).Gradients()
        scan = td.as_scanner()
        e, de = scan('H 0 0 0; F .1 0 2.1')
        # PCM results from gpu4pyscf -98.20379057832794
        self.assertAlmostEqual(e, -98.2065031603471, 8)
        self.assertAlmostEqual(e, td_ref.e_tot[0], 7)
        # PCM results from gpu4pyscf 0.011132973
        self.assertAlmostEqual(de[0,0], 0.011053227, 5)
        self.assertAlmostEqual(abs(ref - de).max(), 0, 5)

if __name__ == "__main__":
    print("Full Tests for ddcosmo TDDFT gradients")
    unittest.main()
