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
from pyscf.pbc import gto, scf

def setUpModule():
    global cell
    cell = gto.M(atom='''
    C 4.826006352031   3.412501814582   8.358888185226
    C 0.689429478862   0.487500259226   1.194126883604
                 ''',
    a='''
    4.136576868, 0.000000000, 2.388253772
    1.378858962, 3.900002074, 2.388253772
    0.000000000, 0.000000000, 4.776507525
                 ''',
    unit='B',
    precision=1e-14,
    basis='gth-tzv2p',
    pseudo='gth-lda',
    mesh=[15]*3,
    verbose=0)

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_rcut(self):
        kpts = cell.make_kpts([2,2,2])
        t0 = numpy.asarray(cell.pbc_intor('int1e_kin_sph', hermi=1, kpts=kpts))
        s0 = numpy.asarray(cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts))
        for i in range(1, 10):
            prec = 1e-13 * 10**i
            cell.rcut = max([cell.bas_rcut(ib, prec) for ib in range(cell.nbas)])
            t1 = numpy.asarray(cell.pbc_intor('int1e_kin_sph', hermi=1, kpts=kpts))
            s1 = numpy.asarray(cell.pbc_intor('int1e_ovlp_sph', hermi=1, kpts=kpts))
            #print prec, cell.rcut, abs(t1-t0).max(), abs(s1-s0).max()
            print(prec, 'error = ', abs(t1-t0).max(), abs(s1-s0).max())
            self.assertTrue(abs(t1-t0).max() < prec*1e-0)
            self.assertTrue(abs(s1-s0).max() < prec*1e-1)

if __name__ == '__main__':
    print("Test rcut and the errorsin pbc.gto.cell")
    unittest.main()
