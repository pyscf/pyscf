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
from functools import reduce

from pyscf import gto
from pyscf import scf
from pyscf import cc

def setUpModule():
    global mol, rhf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_bz"

    mol.atom.extend([
        ["C", (-0.65830719,  0.61123287, -0.00800148)],
        ["C", ( 0.73685281,  0.61123287, -0.00800148)],
        ["C", ( 1.43439081,  1.81898387, -0.00800148)],
        ["C", ( 0.73673681,  3.02749287, -0.00920048)],
        ["C", (-0.65808819,  3.02741487, -0.00967948)],
        ["C", (-1.35568919,  1.81920887, -0.00868348)],
        ["H", (-1.20806619, -0.34108413, -0.00755148)],
        ["H", ( 1.28636081, -0.34128013, -0.00668648)],
        ["H", ( 2.53407081,  1.81906387, -0.00736748)],
        ["H", ( 1.28693681,  3.97963587, -0.00925948)],
        ["H", (-1.20821019,  3.97969587, -0.01063248)],
        ["H", (-2.45529319,  1.81939187, -0.00886348)],])


    mol.basis = {"H": '6-31g',
                 "C": '6-31g',}
    mol.build()

    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()

def tearDownModule():
    global mol, rhf
    del mol, rhf


class KnownValues(unittest.TestCase):
    def test_ccsd(self):
        mcc = cc.ccsd.CC(rhf)
        mcc.conv_tol = 1e-12
        mcc.conv_tol_normt = 1e-6
        mcc.kernel()
        self.assertTrue(numpy.allclose(mcc.t2,mcc.t2.transpose(1,0,3,2)))
        self.assertAlmostEqual(mcc.ecc, -0.5690403273511450, 8)
        self.assertAlmostEqual(abs(mcc.t2).sum(), 92.612789226948451, 4)


if __name__ == "__main__":
    print("Full Tests for C6H6")
    unittest.main()
