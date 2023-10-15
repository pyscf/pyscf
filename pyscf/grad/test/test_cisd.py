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
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import ci
from pyscf import grad
from pyscf.grad import cisd as cisd_grad

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-8
    mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_cisd_grad(self):
        myci = ci.cisd.CISD(mf)
        myci.conv_tol = 1e-10
        myci.kernel()
        g1 = myci.nuc_grad_method().kernel(myci.ci, atmlst=[0,1,2])
# O     0.0000000000    -0.0000000000     0.0065498854
# H    -0.0000000000     0.0208760610    -0.0032749427
# H    -0.0000000000    -0.0208760610    -0.0032749427
        self.assertAlmostEqual(lib.fp(g1), -0.032562347119070523, 6)

    def test_cisd_grad_finite_diff(self):
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        ci_scanner = scf.RHF(mol).set(conv_tol=1e-14).apply(ci.CISD).as_scanner()
        e0 = ci_scanner(mol)
        e1 = ci_scanner(mol.set_geom_('H 0 0 0; H 0 0 1.704'))

        ci_scanner.nroots = 2
        ci_scanner(mol.set_geom_('H 0 0 0; H 0 0 1.705'))
        g1 = ci_scanner.nuc_grad_method().kernel()
        self.assertAlmostEqual(g1[0,2], (e1-e0)*500, 6)

    def test_cisd_grad_excited_state(self):
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        myci = scf.RHF(mol).set(conv_tol=1e-14).apply(ci.CISD).set(nroots=3)
        ci_scanner = myci.as_scanner()
        e0 = ci_scanner(mol)
        e1 = ci_scanner(mol.set_geom_('H 0 0 0; H 0 0 1.704'))

        g_scan = myci.nuc_grad_method().as_scanner(state=2)
        g1 = g_scan('H 0 0 0; H 0 0 1.705', atmlst=range(2))[1]
        self.assertAlmostEqual(g1[0,2], (e1[2]-e0[2])*500, 6)

    def test_frozen(self):
        myci = ci.cisd.CISD(mf)
        myci.frozen = [0,1,10,11,12]
        myci.max_memory = 1
        myci.kernel()
        g1 = cisd_grad.Gradients(myci).kernel(myci.ci)
# O    -0.0000000000     0.0000000000     0.0106763547
# H     0.0000000000    -0.0763194988    -0.0053381773
# H     0.0000000000     0.0763194988    -0.0053381773
        self.assertAlmostEqual(lib.fp(g1), 0.10224149952700579, 6)

    def test_as_scanner(self):
        myci = ci.cisd.CISD(mf)
        myci.frozen = [0,1,10,11,12]
        gscan = myci.nuc_grad_method().as_scanner().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -76.032220245016717, 9)
        self.assertAlmostEqual(lib.fp(g1), 0.10224149952700579, 6)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().CISD().run().Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), 0.11924457198332741, 7)


if __name__ == "__main__":
    print("Tests for CISD gradients")
    unittest.main()
