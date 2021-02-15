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
from pyscf import gto
from pyscf import dft
from pyscf.prop import nmr
from pyscf.data import nist
nist.ALPHA = 1./137.03599967994

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null'

mol.atom = '''
     O      0.   0.       0.
     H      0.  -0.757    0.587
     H      0.   0.757    0.587'''
mol.basis = 'ccpvdz'
mol.build()

def finger(mat):
    w = numpy.cos(numpy.arange(mat.size))
    return numpy.dot(w, mat.ravel())

class KnowValues(unittest.TestCase):
    def test_nr_lda_common_gauge(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.grids.prune = False
        mf.xc = 'lda,vwn'
        mf.scf()
        m = nmr.RKS(mf)
        m.gauge_orig = (1,1,1)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 13.743109885011432, 5)

    def test_nr_b3lyp_common_gauge(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.grids.prune = False
        mf.xc = 'b3lypg'
        mf.scf()
        m = nmr.RKS(mf)
        m.gauge_orig = (1,1,1)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 15.205571299799631, 5)

    def test_nr_lda_giao(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.grids.prune = False
        mf.xc = 'lda,vwn'
        mf.scf()
        m = nmr.RKS(mf)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 58.642932758748856, 5)

    def test_nr_b3lyp_giao(self):
        mf = dft.RKS(mol)
        mf.conv_tol_grad = 1e-6
        mf.grids.prune = False
        mf.xc = 'b3lypg'
        mf.scf()
        m = nmr.RKS(mf)
        msc = m.kernel()
        self.assertAlmostEqual(finger(msc), 55.069383506691494, 5)



if __name__ == "__main__":
    print("Full Tests of RHF-MSC DHF-MSC for HF")
    unittest.main()

