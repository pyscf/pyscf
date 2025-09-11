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
import numpy
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import ao2mo
from pyscf import cc
from pyscf.cc import ccsd_t
from pyscf import grad
from pyscf.grad import ccsd_t as ccsd_t_grad
from pyscf.grad import ccsd_t_slow as ccsd_t_grad_slow
from pyscf.cc import ccsd_t_lambda, ccsd_t_lambda_slow

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
    def test_ccsd_t_grad(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.max_memory = 1
        mycc.conv_tol = 1e-10
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
        conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
        g1 = ccsd_t_grad.Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)

        conv, l1_slow, l2_slow = ccsd_t_lambda_slow.kernel(mycc, eris, t1, t2)
        g1_slow = ccsd_t_grad_slow.Gradients(mycc).kernel(t1, t2, l1_slow, l2_slow, eris=eris)
#[[ 1.43232988e-16 -1.28285681e-16  1.12044750e-02]
# [-2.57370534e-16  2.34464042e-02 -5.60223751e-03]
# [ 1.14137546e-16 -2.34464042e-02 -5.60223751e-03]]
        self.assertAlmostEqual(lib.fp(g1), -0.03843861359532726, 9)
        self.assertTrue(numpy.allclose(g1, g1_slow, rtol=1e-9, atol=1e-12))

        myccs = mycc.as_scanner()
        mol.atom[0] = ["O" , (0., 0., 0.001)]
        mol.build(0, 0)
        e1 = myccs(mol)
        e1 += myccs.ccsd_t()
        mol.atom[0] = ["O" , (0., 0.,-0.001)]
        mol.build(0, 0)
        e2 = myccs(mol)
        e2 += myccs.ccsd_t()
        fd = (e1-e2)/0.002*lib.param.BOHR
        self.assertAlmostEqual(g1[0,2], fd, 5)

    def test_ccsd_t_frozen(self):
        mycc = cc.ccsd.CCSD(mf)
        mycc.frozen = [0,1,10,11,12]
        mycc.diis_start_cycle = 1
        mycc.max_memory = 1
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
        conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, t1, t2)
        g1 = ccsd_t_grad.Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)

        conv, l1_slow, l2_slow = ccsd_t_lambda_slow.kernel(mycc, eris, t1, t2)
        g1_slow = ccsd_t_grad_slow.Gradients(mycc).kernel(t1, t2, l1_slow, l2_slow, eris=eris)
#[[ 1.07308036e-16 -2.51504538e-15  1.24240141e-02]
# [-2.21953002e-18 -7.92491838e-02 -6.21200703e-03]
# [-1.05088506e-16  7.92491838e-02 -6.21200703e-03]]
        self.assertAlmostEqual(lib.fp(g1), 0.10551838331192571, 9)
        self.assertTrue(numpy.allclose(g1, g1_slow, rtol=1e-9, atol=1e-12))

if __name__ == "__main__":
    print("Tests for CCSD(T) gradients")
    unittest.main()
