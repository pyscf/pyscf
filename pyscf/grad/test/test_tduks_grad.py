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
mol.charge = 2
mol.spin = 2
mol.basis = '631g'
mol.build()
pmol = mol.copy()

class KnownValues(unittest.TestCase):
    def test_tda_lda(self):
        mf = dft.UKS(mol).set(conv_tol=1e-12)
        mf.xc = 'LDA,'
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.40279473514282405, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tda_b88(self):
        mf = dft.UKS(mol).set(conv_tol=1e-12)
        mf.xc = 'b88,'
        mf.scf()
        td = tddft.TDA(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.8120037135120326, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tddft_lda(self):
        mf = dft.UKS(mol).set(conv_tol=1e-12)
        mf.xc = 'LDA,'
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.39791714992157035, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tddft_b3lyp(self):
        mf = dft.UKS(mol).set(conv_tol=1e-12)
        mf.xc = '.2*HF + .8*b88, vwn'
        mf.scf()
        td = tddft.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.80446691153291727, 6)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 4)


if __name__ == "__main__":
    print("Full Tests for TD-UKS gradients")
    unittest.main()

