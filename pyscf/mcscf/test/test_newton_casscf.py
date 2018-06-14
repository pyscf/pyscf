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
from functools import reduce
import numpy
from pyscf import gto, scf, lib, fci
from pyscf.mcscf import newton_casscf

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ['H', ( 5.,-1.    , 1.   )],
    ['H', ( 0.,-5.    ,-2.   )],
    ['H', ( 4.,-0.5   ,-3.   )],
    ['H', ( 0.,-4.5   ,-1.   )],
    ['H', ( 3.,-0.5   ,-0.   )],
    ['H', ( 0.,-3.    ,-1.   )],
    ['H', ( 2.,-2.5   , 0.   )],
    ['H', ( 1., 1.    , 3.   )],
]
mol.basis = 'sto-3g'
mol.build()
mf = scf.RHF(mol).run(conv_tol=1e-14)
mc = newton_casscf.CASSCF(mf, 4, 4)
mc.fcisolver = fci.direct_spin1.FCI(mol)
mc.kernel()

def tearDownModule():
    global mol, mf, mc
    del mol, mf, mc


class KnownValues(unittest.TestCase):
    def test_gen_g_hop(self):
        numpy.random.seed(1)
        mo = numpy.random.random(mf.mo_coeff.shape)
        ci0 = numpy.random.random((6,6))
        ci0/= numpy.linalg.norm(ci0)
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(mc, mo, ci0, mc.ao2mo(mo))
        self.assertAlmostEqual(lib.finger(gall), 21.288022525148595, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -4.6864640132374618, 8)
        x = numpy.random.random(gall.size)
        u, ci1 = newton_casscf.extract_rotation(mc, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -412.9441873541524, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 73.358310983341198, 8)

    def test_get_grad(self):
        self.assertAlmostEqual(mc.e_tot, -3.6268060760105141, 9)
        self.assertAlmostEqual(abs(mc.get_grad()).max(), 0, 5)

if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()


