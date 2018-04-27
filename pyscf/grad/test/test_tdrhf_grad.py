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
from pyscf import gto, scf
from pyscf import tdscf
from pyscf.grad import tdrhf as tdrhf_grad


mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'
mol.atom = [
    ['H' , (0. , 0. , 1.804)],
    ['F' , (0. , 0. , 0.)], ]
mol.unit = 'B'
mol.basis = '631g'
mol.build()
pmol = mol.copy()
mf = scf.RHF(mol).set(conv_tol=1e-12).run()

def tearDownModule():
    global mol, pmol, mf
    mol.stdout.close()
    del mol, pmol, mf

class KnownValues(unittest.TestCase):
    def test_tda_singlet(self):
        td = tdscf.TDA(mf).run(nstates=3)
        tdg = td.nuc_grad_method().as_scanner()
        g1 = tdg(mol.atom_coords(), state=3)[1]
        self.assertAlmostEqual(g1[0,2], -0.23226123352352346, 8)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 6)

        self.assertAlmostEqual(abs(tdg.kernel(state=0) -
                                   mf.nuc_grad_method().kernel()).max(), 0, 8)

    def test_tda_triplet(self):
        td = tdscf.TDA(mf).run(singlet=False, nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.47296513687621511, 8)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 5)

    def test_tdhf(self):
        td = tdscf.TDDFT(mf).run(nstates=3)
        tdg = td.nuc_grad_method()
        g1 = tdrhf_grad.kernel(tdg, td.xy[2])
        g1 += tdg.grad_nuc()
        self.assertAlmostEqual(g1[0,2], -0.25240005833657309, 8)

        td_solver = td.as_scanner()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual((e1[2]-e2[2])/.002, g1[0,2], 6)


if __name__ == "__main__":
    print("Full Tests for TD-RHF gradients")
    unittest.main()


