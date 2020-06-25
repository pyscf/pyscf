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
from pyscf.ctfcc import kccsd_uhf


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
cell.verbose = 4
cell.output = '/dev/null'
cell.spin = 2
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KUHF(cell,kpts, exxdiv=None)
mf.max_cycle = 1
mf.kernel()

mycc = kccsd_uhf.KUCCSD(mf)
refcc = cc.KUCCSD(mf)

eris = mycc.ao2mo()
erisr = refcc.ao2mo()

class REFTEST(unittest.TestCase):

    def test_eris(self):
        for key in eris.__dict__.keys():
            if len(key.lower().replace('o','').replace('v',''))!=0:
                continue
            x = getattr(eris, key)
            y = getattr(erisr, key)
            if x is not None and y is not None:
                x=  x.array.to_nparray()
                self.assertTrue(numpy.linalg.norm(x-y)<1e-10)

    def test_update_amps(self):
        e1, t1, t2 = mycc.init_amps(eris=eris)
        e1r, t1r, t2r = refcc.init_amps(eris=erisr)
        self.assertTrue(abs(e1-e1r)<1e-10)
        self.assertTrue(numpy.linalg.norm(t1[0].array.to_nparray()-t1r[0])<1e-10)
        self.assertTrue(numpy.linalg.norm(t1[1].array.to_nparray()-t1r[1])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[0].array.to_nparray()-t2r[0])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[1].array.to_nparray()-t2r[1])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[2].array.to_nparray()-t2r[2])<1e-10)

        t1, t2 = mycc.update_amps(t1, t2, eris=eris)
        t1r, t2r = refcc.update_amps(t1r, t2r, eris=erisr)

        self.assertTrue(numpy.linalg.norm(t1[0].array.to_nparray()-t1r[0])<1e-10)
        self.assertTrue(numpy.linalg.norm(t1[1].array.to_nparray()-t1r[1])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[0].array.to_nparray()-t2r[0])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[1].array.to_nparray()-t2r[1])<1e-10)
        self.assertTrue(numpy.linalg.norm(t2[2].array.to_nparray()-t2r[2])<1e-10)

if __name__ == '__main__':
    print("kccsd_uhf tests")
    unittest.main()
