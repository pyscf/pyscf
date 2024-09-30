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
from pyscf import gto, scf, dft, lib
from pyscf import grad
from pyscf.grad import rks, uks, roks

def setUpModule():
    global h2o, h2o_n, h2o_p
    h2o = gto.Mole()
    h2o.verbose = 5
    h2o.output = '/dev/null'
    h2o.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = '6-31g'
    h2o.build()

    h2o_n = gto.Mole()
    h2o_n.verbose = 5
    h2o_n.output = '/dev/null'
    h2o_n.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o_n.charge = -1
    h2o_n.spin = 1
    h2o_n.basis = '6-31g'
    h2o_n.build()

    h2o_p = gto.Mole()
    h2o_p.verbose = 5
    h2o_p.output = '/dev/null'
    h2o_p.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o_p.charge = 1
    h2o_p.spin = 1
    h2o_p.basis = '6-31g'
    h2o_p.build()

def tearDownModule():
    global h2o, h2o_n, h2o_p
    h2o.stdout.close()
    h2o_n.stdout.close()
    h2o_p.stdout.close()
    del h2o, h2o_n, h2o_p


def fp(mat):
    return abs(mat).sum()

class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(h2o)
        rhf.conv_tol = 1e-14
        rhf.kernel()
        g = grad.RHF(rhf)
        self.assertAlmostEqual(fp(g.grad_elec()), 10.126405944938071, 6)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(h2o)
        uhf.conv_tol = 1e-12
        uhf.kernel()
        g = grad.DHF(uhf)
        self.assertAlmostEqual(fp(g.grad_elec()), 10.126445612578864, 6)

    def test_nr_uhf(self):
        mf = scf.UHF(h2o_n)
        mf.conv_tol = 1e-14
        mf.kernel()
        g = grad.UHF(mf)
        self.assertAlmostEqual(lib.fp(g.grad_elec()), 4.2250348208172541, 6)

    def test_nr_rohf(self):
        mf = scf.ROHF(h2o_n)
        mf.conv_tol = 1e-14
        mf.kernel()
        g = grad.ROHF(mf)
        self.assertAlmostEqual(lib.fp(g.grad_elec()), 4.230345200147354, 6)

    def test_energy_nuc(self):
        rhf = scf.RHF(h2o)
        g = grad.RHF(rhf)
        self.assertAlmostEqual(fp(g.grad_nuc()), 10.086972893020102, 9)

    def test_ccsd(self):
        from pyscf import cc
        rhf = scf.RHF(h2o)
        rhf.kernel()
        mycc = cc.CCSD(rhf)
        mycc.kernel()
        mycc.solve_lambda()
        g1 = grad.ccsd.Gradients(mycc).kernel()
        self.assertAlmostEqual(fp(g1), 0.065802850540912422, 6)

    def test_rks_lda(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-14, xc='lda,vwn')
        g = rks.Grad(mf)
        self.assertAlmostEqual(fp(g.grad()), 0.098438461959390822, 5)
        g.grid_response = True
        self.assertAlmostEqual(fp(g.grad()), 0.098441823256625829, 5)

    def test_rks_bp86(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-14, xc='b88,p86')
        g = rks.Grad(mf)
        self.assertAlmostEqual(fp(g.grad()), 0.10362532283229957, 5)
        g.grid_response = True
        self.assertAlmostEqual(fp(g.grad()), 0.10357804241970789, 5)

    def test_rks_b3lypg(self):
        mf = dft.RKS(h2o)
        mf.grids.prune = None
        mf.run(conv_tol=1e-14, xc='b3lypg')
        g = rks.Grad(mf)
        self.assertAlmostEqual(fp(g.grad()), 0.066541921001296467, 5)
        g.grid_response = True
        self.assertAlmostEqual(fp(g.grad()), 0.066543737224608879, 5)

    def test_uks_lda(self):
        mf = dft.UKS(h2o_p)
        mf.run(conv_tol=1e-14, xc='lda,vwn')
        g = uks.Grad(mf)
        self.assertAlmostEqual(lib.fp(g.grad()), -0.12090786418355501, 5)
        g.grid_response = True
        self.assertAlmostEqual(lib.fp(g.grad()), -0.12091122603875157, 5)

    def test_roks_lda(self):
        mf = dft.ROKS(h2o_p)
        mf.run(conv_tol=1e-14, xc='lda,vwn')
        g = roks.Grad(mf)
        self.assertAlmostEqual(lib.fp(g.grad()), -0.12051785975616186, 5)
        g.grid_response = True
        self.assertAlmostEqual(lib.fp(g.grad()), -0.12052121736985746, 5)

    def test_uks_b3lypg(self):
        mf = dft.UKS(h2o_n)
        mf.run(conv_tol=1e-14, xc='b3lypg')
        g = uks.Grad(mf)
        self.assertAlmostEqual(lib.fp(g.grad()), -0.1436034999176907, 5)
        g.grid_response = True
        self.assertAlmostEqual(lib.fp(g.grad()), -0.14360504586558553, 5)

    def test_roks_b3lypg(self):
        mf = dft.ROKS(h2o_n)
        mf.run(conv_tol=1e-14, xc='b3lypg')
        g = roks.Grad(mf)
        self.assertAlmostEqual(lib.fp(g.grad()), -0.14281412906618443, 5)
        g.grid_response = True
        self.assertAlmostEqual(lib.fp(g.grad()), -0.14281571291026954, 5)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
