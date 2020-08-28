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
from pyscf.prop import nmr
from pyscf.data import nist
nist.ALPHA = 1./137.03599967994

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'

mol.atom = [
    [1   , (0. , 0. , .917)],
    ["F" , (0. , 0. , 0.)], ]
#mol.nucmod = {"F":2, "H":2}
mol.basis = {"H": 'cc_pvdz',
             "F": 'cc_pvdz',}
mol.build()

nrhf = scf.RHF(mol)
nrhf.conv_tol_grad = 1e-6
nrhf.conv_tol = 1e-12
nrhf.scf()

rhf = scf.DHF(mol)
rhf.conv_tol_grad = 1e-7
rhf.conv_tol = 1e-12
rhf.scf()

def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_common_gauge_ucpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = False
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1636.7413165636394, 5)

    def test_nr_common_gauge_cpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1562.3859473764551, 5)

    def test_nr_giao_ucpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = False
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1488.0948832784416, 5)

    def test_nr_giao_cpscf(self):
        m = nmr.RHF(nrhf)
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1358.9826064972372, 5)

    def test_rmb_common_gauge_ucpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = False
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1642.1872658333457, 4)

    def test_rmb_common_gauge_cpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1569.0400642905215, 4)

    def test_rmb_giao_ucpscf(self):
        m = nmr.DHF(rhf)
        m.cphf = False
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1493.7229929087348, 4)

    def test_rmb_giao_cpscf_high_cost(self):
        m = nmr.DHF(rhf)
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1365.4679007423506, 4)

    def test_rkb_giao_cpscf(self):
        m = nmr.DHF(rhf)
        m.mb = 'RKB'
        m.cphf = True
        m.gauge_orig = None
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1923.9092803444623, 4)

    def test_rkb_common_gauge_cpscf(self):
        m = nmr.DHF(rhf)
        m.mb = 'RKB'
        m.cphf = True
        m.gauge_orig = (1,1,1)
        msc = m.shielding()
        self.assertAlmostEqual(finger(msc), 1980.1179936444073, 4)

    def test_make_h10(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm0 = numpy.random.random((nao,nao))
        dm0 = dm0 + dm0.T
        h1 = nmr.rhf.make_h10(mol, dm0)
        self.assertAlmostEqual(numpy.linalg.norm(h1), 21.255203821714673, 8)
        h1 = nmr.rhf.make_h10(mol, dm0, gauge_orig=(0,0,0))
        self.assertAlmostEqual(numpy.linalg.norm(h1), 4.020198783142229, 8)
        n4c = mol.nao_2c()*2
        numpy.random.seed(1)
        dm0 = numpy.random.random((n4c,n4c))
        dm0 = dm0 + dm0.T.conj()
        h1 = nmr.dhf.make_h10(mol, dm0)
        self.assertAlmostEqual(numpy.linalg.norm(h1), 73.452535645731714, 8)
        h1 = nmr.dhf.make_h10(mol, dm0, gauge_orig=(0,0,0), mb='RKB')
        self.assertAlmostEqual(numpy.linalg.norm(h1), 7.3636964305440609, 8)



if __name__ == "__main__":
    print("Full Tests of RHF-MSC DHF-MSC for HF")
    unittest.main()
