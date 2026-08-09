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
from pyscf import scf
from pyscf import cc
from pyscf.cc import uccsd_t_lambda
from pyscf.grad import uccsd_t as uccsd_t_grad

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.spin = 2
    mol.basis = '631g'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol_grad = 1e-8
    mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


def _fd_z(mycc, delta=0.001):
    '''Central finite difference of the CCSD(T) energy along the z coordinate
    of the first atom.'''
    myccs = mycc.as_scanner()
    atom0 = mol.atom[0]
    try:
        mol.atom[0] = ["O", (0., 0., delta)]
        mol.build(0, 0)
        e1 = myccs(mol)
        e1 += myccs.ccsd_t()
        mol.atom[0] = ["O", (0., 0., -delta)]
        mol.build(0, 0)
        e2 = myccs(mol)
        e2 += myccs.ccsd_t()
    finally:
        mol.atom[0] = atom0
        mol.build(0, 0)
    return (e1 - e2) / (2 * delta) * lib.param.BOHR


class KnownValues(unittest.TestCase):
    def test_uccsd_t_grad(self):
        mycc = cc.uccsd.UCCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.conv_tol_normt = 1e-8
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        conv, l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)
        g1 = uccsd_t_grad.Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)
#[[ 0.            0.            0.14809395]
# [ 0.            0.11228921   -0.07404698]
# [ 0.           -0.11228921   -0.07404698]]
        self.assertAlmostEqual(lib.fp(g1), -0.22991153901988648, 7)
        self.assertAlmostEqual(g1[0,2], _fd_z(mycc), 5)

    def test_uccsd_t_grad_frozen(self):
        mycc = cc.uccsd.UCCSD(mf, frozen=1)
        mycc.conv_tol = 1e-10
        mycc.conv_tol_normt = 1e-8
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)
        conv, l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)
        g1 = uccsd_t_grad.Gradients(mycc).kernel(t1, t2, l1, l2, eris=eris)
        self.assertAlmostEqual(lib.fp(g1), -0.22995398197034367, 7)
        self.assertAlmostEqual(g1[0,2], _fd_z(mycc), 5)

    def test_uccsd_t_grad_vs_rccsd_t(self):
        '''On a closed-shell molecule the UHF reference reduces to the RHF one,
        so both gradient codes must agree. This is the regression guard for the
        mixed-spin dvvVV compression bug of issue #3305, which affected only
        the UHF path.
        '''
        from pyscf.cc import ccsd_t_lambda
        from pyscf.grad import ccsd_t as ccsd_t_grad
        pmol = gto.M(atom=mol.atom, basis='631g', spin=0, verbose=0)

        rmf = scf.RHF(pmol).run(conv_tol=1e-12)
        rcc = cc.ccsd.CCSD(rmf)
        rcc.conv_tol = 1e-10
        rcc.conv_tol_normt = 1e-8
        reris = rcc.ao2mo()
        rcc.kernel(eris=reris)
        conv, rl1, rl2 = ccsd_t_lambda.kernel(rcc, reris, rcc.t1, rcc.t2)
        gr = ccsd_t_grad.Gradients(rcc).kernel(rcc.t1, rcc.t2, rl1, rl2, eris=reris)

        umf = scf.UHF(pmol).run(conv_tol=1e-12)
        ucc = cc.uccsd.UCCSD(umf)
        ucc.conv_tol = 1e-10
        ucc.conv_tol_normt = 1e-8
        ueris = ucc.ao2mo()
        ucc.kernel(eris=ueris)
        conv, ul1, ul2 = uccsd_t_lambda.kernel(ucc, ueris, ucc.t1, ucc.t2)
        gu = uccsd_t_grad.Gradients(ucc).kernel(ucc.t1, ucc.t2, ul1, ul2, eris=ueris)

        self.assertAlmostEqual(abs(gu - gr).max(), 0, 7)

if __name__ == "__main__":
    print("Tests for UCCSD(T) gradients")
    unittest.main()
