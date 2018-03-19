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
from pyscf import gto, dft
from pyscf import tddft

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['H' , (0. , 0. , 1.804)],
    ['F' , (0. , 0. , 0.)], ]
mol.unit = 'B'
mol.basis = '631g'
mol.build()

def finger(a):
    w = numpy.cos(numpy.arange(len(a)))
    return numpy.dot(w, a)

class KnownValues(unittest.TestCase):
    def test_tda_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'LDA'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -9.23916667e-02, 8)

    def test_tda_b88(self):
        mf = dft.RKS(mol)
        mf.xc = 'b88'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -9.32506535e-02, 8)

    def test_tddft_lda(self):
        mf = dft.RKS(mol)
        mf.xc = 'LDA'
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -1.31315477e-01, 8)

    def test_tddft_b3lyp(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf._numint.libxc = dft.xcfun
        mf.grids.prune = False
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=2)
        self.assertAlmostEqual(g1[0,2], -1.55778110e-01, 7)


if __name__ == "__main__":
    print("Full Tests for TD-RKS gradients")
    unittest.main()

