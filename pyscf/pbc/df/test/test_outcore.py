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
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import dft as pdft
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
import pyscf.pbc
#pyscf.pbc.DEBUG = False

def setUpModule():
    global cell
    cell = pgto.Cell()
    cell.unit = 'B'
    cell.a = numpy.eye(3) * 4.
    cell.mesh = [11]*3
    cell.atom = 'He 0 1 1; He 1 1 0'
    cell.basis = { 'He': [[0, (0.8, 1.0)],
                          [0, (1.2, 1.0)]] }
    cell.verbose = 0
    cell.build(0, 0)

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_aux_e1(self):
        numpy.random.seed(1)
        kptij_lst = numpy.random.random((3,2,3))
        kptij_lst[0] = 0
        with tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfile:
            outcore.aux_e1(cell, cell, tmpfile.name, aosym='s2', comp=1,
                           kptij_lst=kptij_lst, verbose=0)
            refk = incore.aux_e2(cell, cell, aosym='s1', kptij_lst=kptij_lst)
            with h5py.File(tmpfile.name, 'r') as f:
                nao = cell.nao_nr()
                idx = numpy.tril_indices(nao)
                idx = idx[0] * nao + idx[1]
                self.assertTrue(numpy.allclose(refk[0,idx], f['eri_mo/0'][:].T))
                self.assertTrue(numpy.allclose(refk[1], f['eri_mo/1'][:].T))
                self.assertTrue(numpy.allclose(refk[2], f['eri_mo/2'][:].T))

if __name__ == '__main__':
    print("Full Tests for pbc.df.outcore")
    unittest.main()
