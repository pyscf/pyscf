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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df
from pyscf import mcscf

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = '6-31g',
        output = '/dev/null'
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_rhf_grad(self):
        gref = scf.RHF(mol).run().nuc_grad_method().kernel()
        g1 = scf.RHF(mol).density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 5)

    def test_rks_lda_grad(self):
        gref = mol.RKS(xc='lda,').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='lda,').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_rks_grad(self):
        gref = mol.RKS(xc='b3lyp').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='b3lyp').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_uhf_grad(self):
        gref = mol.UHF.run().nuc_grad_method().kernel()
        g1 = mol.UHF.density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 5)

    def test_uks_lda_grad(self):
        gref = mol.UKS.run(xc='lda,').nuc_grad_method().kernel()
        g1 = mol.UKS.density_fit().run(xc='lda,').nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_uks_grad(self):
        gref = mol.UKS.run(xc='b3lyp').nuc_grad_method().kernel()
        g1 = mol.UKS.density_fit().run(xc='b3lyp').nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_casscf_grad(self):
        gref = mcscf.CASSCF (mol.RHF.run (), 8, 6).run ().nuc_grad_method().kernel()
        g1 = mcscf.CASSCF (mol.RHF.density_fit().run(), 8, 6).run ().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_sacasscf_grad(self):
        mf = mol.RHF.run ()
        mc = mcscf.CASSCF (mf,8,6).state_average_([.5,.5]).run()
        gref = mc.nuc_grad_method ().kernel (state=1)
        mc = mcscf.CASSCF (mf.density_fit(),8,6).state_average_([.5,.5]).run()
        g1 = mc.nuc_grad_method ().kernel (state=1)
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for df.grad")
    unittest.main()

