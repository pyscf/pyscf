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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf import gto, dft
from pyscf import tdscf
from pyscf.grad import tdrks as tdrks_grad

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H' , (0. , 0. , 1.804)],
    ['F' , (0. , 0. , 0.)], ]
mol.unit = 'B'
mol.basis = '631g'
mol.build()
mf_lda = dft.RKS(mol).set(xc='LDA,')
mf_lda.grids.prune = False
mf_lda.kernel()
mf_gga = dft.RKS(mol).set(xc='b88,')
mf_gga.grids.prune = False
mf_gga.kernel()

def tearDownModule():
    global mol, mf_lda
    del mol, mf_lda

class KnownValues(unittest.TestCase):
    def test_tda_singlet_lda(self):
        td = tdscf.TDA(mf_lda).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdrks_grad.kernel(tdg, td.xy[2])
        g1 += tdg.grad_nuc()
        self.assertAlmostEqual(g1[0,2], -9.23916667e-02, 8)

#    def test_tda_triplet_lda(self):
#        td = tdscf.TDA(mf_lda).run(singlet=False, nstates=3)
#        tdg = td.nuc_grad_method()
#        g1 = tdg.kernel(state=3)
#        self.assertAlmostEqual(g1[0,2], -9.23916667e-02, 8)

    def test_tda_singlet_b88(self):
        td = tdscf.TDA(mf_gga).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -9.32506535e-02, 8)

#    def test_tda_triplet_b88(self):
#        td = tdscf.TDA(mf_gga).run(singlet=False, nstates=3)
#        tdg = td.nuc_grad_method()
#        g1 = tdg.kernel(state=3)
#        self.assertAlmostEqual(g1[0,2], -9.32506535e-02, 8)

    def test_tddft_lda(self):
        td = tdscf.TDDFT(mf_lda).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -1.31315477e-01, 8)

    def test_tddft_b3lyp_high_cost(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf._numint.libxc = dft.xcfun
        mf.grids.prune = False
        mf.scf()
        td = tdscf.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -1.55778110e-01, 7)

    def test_range_separated_high_cost(self):
        mol = gto.M(atom="H; H 1 1.", basis='631g', verbose=0)
        mf = dft.RKS(mol).set(xc='CAMB3LYP')
        mf._numint.libxc = dft.xcfun
        td = mf.apply(tdscf.TDA)
        tdg_scanner = td.nuc_grad_method().as_scanner().as_scanner()
        g = tdg_scanner(mol, state=3)[1]
        self.assertAlmostEqual(lib.finger(g), 0.60109310253094916, 7)
        smf = td.as_scanner()
        e1 = smf(mol.set_geom_("H; H 1 1.001"))[2]
        e2 = smf(mol.set_geom_("H; H 1 0.999"))[2]
        self.assertAlmostEqual((e1-e2)/0.002*lib.param.BOHR, g[1,0], 4)


if __name__ == "__main__":
    print("Full Tests for TD-RKS gradients")
    unittest.main()

