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
from pyscf import cc
from pyscf import grad
from pyscf.grad import ccsd as ccsd_grad

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


class KnownValues(unittest.TestCase):
    def test_ccsd_grad(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), -0.036999389889460096, 6)

        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mf0 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc0 = cc.ccsd.CCSD(mf0).run(conv_tol=1e-10)
        mol.set_geom_('H 0 0 0; H 0 0 1.704', unit='Bohr')
        mf1 = scf.RHF(mol).run(conv_tol=1e-14)
        mycc1= cc.ccsd.CCSD(mf1).run(conv_tol=1e-10)
        mol.set_geom_('H 0 0 0; H 0 0 1.705', unit='Bohr')
        mycc2 = cc.ccsd.CCSD(scf.RHF(mol))
        g_scanner = mycc2.nuc_grad_method().as_scanner()
        g1 = g_scanner(mol)[1]
        self.assertAlmostEqual(g1[0,2], (mycc1.e_tot-mycc0.e_tot)*500, 6)

    def test_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.max_memory = 1
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        g1 = ccsd_grad.kernel(mycc, t1, t2, l1, l2, mf_grad=grad.RHF(mf))
        self.assertAlmostEqual(lib.finger(g1), 0.10599503839207361, 6)

    def test_uccsd_grad(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.spin = 2
        mol.basis = '631g'
        mol.build(0, 0)
        mf = scf.UHF(mol)
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        mycc = cc.UCCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        l1, l2 = mycc.solve_lambda(eris=eris)
        g1 = mycc.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.finger(g1), -0.22892720804519961, 6)

        cc_scanner = mycc.as_scanner()
        mol.set_geom_([
            [8 , (0. , 0.     , 0.001)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]], unit='Ang')
        e1 = cc_scanner(mol)
        mol.set_geom_([
            [8 , (0. , 0.     ,-0.001)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]], unit='Ang')
        e2 = cc_scanner(mol)
        self.assertAlmostEqual(g1[0,2], (e1-e2)/.002*lib.param.BOHR, 5)


if __name__ == "__main__":
    print("Tests for CCSD gradients")
    unittest.main()

