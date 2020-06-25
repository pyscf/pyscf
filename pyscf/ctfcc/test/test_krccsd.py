#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Yang Gao <younggao1994@gmail.com>

import numpy
import unittest
from pyscf.pbc import gto, scf, cc
from pyscf.ctfcc import kccsd_rhf


cell = gto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.mesh = [13,13,13]
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000
'''
cell.verbose=4
cell.output = '/dev/null'
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KRHF(cell,kpts, exxdiv=None)
mf.max_cycle = 1
mf.kernel()

mycc = kccsd_rhf.KRCCSD(mf)
refcc = cc.KRCCSD(mf)

eris = mycc.ao2mo()
erisr = refcc.ao2mo()

class REFTEST(unittest.TestCase):

    def test_eris(self):
        # pyscf.pbc.cc.kccsd_rhf use physicist's integrals
        # pyscf.ctfcc.kccsd_rhf uses chemist's integrals
        self.assertTrue(numpy.linalg.norm(eris.oooo.transpose(0,2,1,3).array.to_nparray() - erisr.oooo)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.ooov.transpose(0,2,1,3).array.to_nparray() - erisr.ooov)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.oovv.transpose(0,2,1,3).array.to_nparray() - erisr.ovov)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.ovov.transpose(0,2,1,3).array.to_nparray() - erisr.oovv)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.vvvv.transpose(0,2,1,3).array.to_nparray() - erisr.vvvv)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.ovvo.transpose(1,3,0,2).conj().array.to_nparray() - erisr.voov)<1e-8)
        self.assertTrue(numpy.linalg.norm(eris.ovvv.transpose(2,0,3,1).array.to_nparray() - erisr.vovv)<1e-8)

    def test_update_amps(self):
        e1, t1, t2 = mycc.init_amps(eris=eris)
        e1r, t1r, t2r = refcc.init_amps(eris=erisr)
        self.assertTrue(abs(e1-e1r)<1e-8)
        self.assertTrue(numpy.linalg.norm(t1.array.to_nparray()-t1r)<1e-8)
        self.assertTrue(numpy.linalg.norm(t2.array.to_nparray()-t2r)<1e-8)
        t1new, t2new = mycc.update_amps(t1, t2, eris=eris)
        t1newr, t2newr = refcc.update_amps(t1r, t2r, eris=erisr)
        self.assertTrue(numpy.linalg.norm(t1new.array.to_nparray()-t1newr)<1e-8)
        self.assertTrue(numpy.linalg.norm(t2new.array.to_nparray()-t2newr)<1e-8)

if __name__ == '__main__':
    print("kccsd_rhf tests")
    unittest.main()
