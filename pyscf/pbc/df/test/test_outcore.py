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
        with lib.NamedTemporaryFile(dir=lib.param.TMPDIR) as tmpfile:
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


    def make_storage_reference(self):
        self.cell = pgto.M(atom='He 0 0 0', a=numpy.eye(3)*3,
                         basis=[[0, [.8, 1]], [1, [.5, 1]]],
                         precision=1e-7, verbose=0)
        self.aux = incore.make_auxcell(self.cell, [[0, [.6, 1]], [1, [.4, 1]]])
        kpts = self.cell.make_kpts([2, 1, 1])
        self.pairs = numpy.array([[kpts[0], kpts[0]], [kpts[0], kpts[1]], [kpts[1], kpts[1]]])

    def test_auxiliary_last(self):
        self.make_storage_reference()
        ref = incore.aux_e2(self.cell, self.aux, aosym='s1', kptij_lst=self.pairs)
        with lib.H5TmpFile() as f:
            outcore._aux_e2(self.cell, self.aux, f, aosym='s1', kptij_lst=self.pairs, max_memory=.01)
            for k in range(3):
                group = f['eri_mo/%d' % k]
                actual = numpy.concatenate([group[str(i)][:] for i in range(len(group))], axis=-1)
                numpy.testing.assert_allclose(actual, ref[k], atol=1e-11)

    def test_auxiliary_first_derivative(self):
        self.make_storage_reference()
        ref = incore.aux_e2(self.cell, self.aux, intor='int3c2e_ip1', aosym='s1', kptij_lst=self.pairs)
        with lib.H5TmpFile() as f:
            outcore.aux_e1(self.cell, self.aux, f, intor='int3c2e_ip1', aosym='s1',
                           kptij_lst=self.pairs, max_memory=.01)
            for k in range(3):
                numpy.testing.assert_allclose(f['eri_mo/%d' % k][:], ref[k].transpose(0, 2, 1), atol=1e-11)

if __name__ == '__main__':
    print("Full Tests for pbc.df.outcore")
    unittest.main()
