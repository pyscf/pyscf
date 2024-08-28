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
from pyscf import dft
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
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_rhf_grad(self):
        gref = scf.RHF(mol).run().nuc_grad_method().kernel()
        g1 = scf.RHF(mol).density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

        pmol = mol.copy()
        mf = scf.RHF(pmol).density_fit(auxbasis='ccpvdz-jkfit').run()
        g = mf.Gradients().set(auxbasis_response=False).kernel()
        self.assertAlmostEqual(lib.fp(g), 0.005466630382488041, 7)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), 0.005516638190173352, 7)
        mfs = mf.as_scanner()
        e1 = mfs([['O' , (0. , 0.     , 0.001)],
                  [1   , (0. , -0.757 , 0.587)],
                  [1   , (0. , 0.757  , 0.587)] ])
        e2 = mfs([['O' , (0. , 0.     ,-0.001)],
                  [1   , (0. , -0.757 , 0.587)],
                  [1   , (0. , 0.757  , 0.587)] ])
        self.assertAlmostEqual((e1-e2)/0.002*lib.param.BOHR, g[0,2], 6)

    def test_rks_lda_grad(self):
        gref = mol.RKS(xc='lda,').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='lda,').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_rks_gga_grad(self):
        gref = mol.RKS(xc='b3lyp').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='b3lyp').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_rks_rsh_grad(self):
        gref = mol.RKS(xc='wb97').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='wb97').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_rks_mgga_grad(self):
        gref = mol.RKS(xc='m06').run().nuc_grad_method().kernel()
        g1 = mol.RKS(xc='m06').density_fit().run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_uhf_grad(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        mol.symmetry = True
        mol.verbose = 0
        mol.basis = '631g'
        mol.spin = 2
        mol.build()
        mf = scf.UHF(mol).density_fit().run(conv_tol=1e-12)
        g1 = mf.nuc_grad_method().kernel()
        gref = mol.UHF.run().nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

        g = mf.Gradients().set(auxbasis_response=False).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.19670644982746546, 7)
        g = mf.Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.19660674423263175, 7)
        mfs = mf.as_scanner()
        e1 = mfs([['O' , (0. , 0.     , 0.001)],
                  [1   , (0. , -0.757 , 0.587)],
                  [1   , (0. , 0.757  , 0.587)] ])
        e2 = mfs([['O' , (0. , 0.     ,-0.001)],
                  [1   , (0. , -0.757 , 0.587)],
                  [1   , (0. , 0.757  , 0.587)] ])
        self.assertAlmostEqual((e1-e2)/0.002*lib.param.BOHR, g[0,2], 6)

    def test_uks_lda_grad(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.charge = 1
        mol.spin = 1
        mol.build()
        mf = mol.UKS().density_fit().run(conv_tol=1e-12)
        gref = mol.UKS.run().nuc_grad_method().kernel()
        g1 = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

        g = mf.Gradients().set(auxbasis_response=False)
        self.assertAlmostEqual(lib.finger(g.kernel()), -0.12092643506961044, 7)
        g = mf.Gradients()
        self.assertAlmostEqual(lib.finger(g.kernel()), -0.12092884149543644, 7)
        g.grid_response = True
        self.assertAlmostEqual(lib.finger(g.kernel()), -0.12093220332146028, 7)

    def test_uks_gga_grad(self):
        gref = mol.UKS.run(xc='b3lyp').nuc_grad_method().kernel()
        g1 = mol.UKS.density_fit().run(xc='b3lyp').nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(gref - g1).max(), 0, 4)

    def test_uks_rsh_grad(self):
        mol = gto.Mole()
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.charge = 1
        mol.spin = 1
        mol.verbose = 0
        mol.build()
        gref = mol.UKS(xc='camb3lyp').run().nuc_grad_method().kernel()
        g1 = mol.UKS(xc='camb3lyp').density_fit().run().nuc_grad_method().kernel()
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
