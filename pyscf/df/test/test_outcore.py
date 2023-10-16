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
import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf import df

def setUpModule():
    global mol, auxmol
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = '/dev/null',
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

    auxmol = df.addons.make_auxmol(mol, 'weigend')

def tearDownModule():
    global mol, auxmol
    mol.stdout.close()
    del mol, auxmol


class KnownValues(unittest.TestCase):
    def test_outcore(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        cderi0 = df.incore.cholesky_eri(mol)
        df.outcore.cholesky_eri(mol, ftmp.name)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['j3c'], cderi0))

        df.outcore.cholesky_eri(mol, ftmp.name, max_memory=.05)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['j3c'], cderi0))

        nao = mol.nao_nr()
        naux = cderi0.shape[0]
        df.outcore.general(mol, (numpy.eye(nao),)*2, ftmp.name, max_memory=.02)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

        ####
        buf = numpy.zeros((naux,nao,nao))
        idx = numpy.tril_indices(nao)
        buf[:,idx[0],idx[1]] = cderi0
        buf[:,idx[1],idx[0]] = cderi0
        cderi0 = buf
        df.outcore.cholesky_eri(mol, ftmp.name, aosym='s1', max_memory=.05)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['j3c'], cderi0.reshape(naux,-1)))

        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        numpy.random.seed(1)
        co = numpy.random.random((nao,4))
        cv = numpy.random.random((nao,25))
        cderi0 = numpy.einsum('kpq,pi,qj->kij', cderi0, co, cv)
        df.outcore.general(mol, (co,cv), ftmp.name, max_memory=.05)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0.reshape(naux,-1)))

        cderi0 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1_sph',
                                  aosym='s1', comp=3).reshape(3,nao**2,-1)
        j2c = df.incore.fill_2c2e(mol, auxmol)
        low = scipy.linalg.cholesky(j2c, lower=True)
        cderi0 = [scipy.linalg.solve_triangular(low, j3c.T, lower=True)
                  for j3c in cderi0]
        nao = mol.nao_nr()
        df.outcore.general(mol, (numpy.eye(nao),)*2, ftmp.name,
                           int3c='int3c2e_ip1_sph', aosym='s1', int2c='int2c2e_sph',
                           comp=3, max_memory=.02)
        with h5py.File(ftmp.name, 'r') as feri:
            self.assertTrue(numpy.allclose(feri['eri_mo'], cderi0))

    def test_lindep(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        df.outcore.cholesky_eri(mol, ftmp.name, auxmol=auxmol, verbose=7)
        with h5py.File(ftmp.name, 'r') as f:
            cderi0 = f['j3c'][:]
        auxmol1 = auxmol.copy()
        auxmol1.basis = {'O': 'weigend', 'H': ('weigend', 'weigend')}
        auxmol1.build(0, 0)
        cderi1 = df.outcore.cholesky_eri(mol, ftmp.name, auxmol=auxmol1)
        with h5py.File(ftmp.name, 'r') as f:
            cderi1 = f['j3c'][:]
        eri0 = numpy.dot(cderi0.T, cderi0)
        eri1 = numpy.dot(cderi1.T, cderi1)
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 9)

#    def test_int3c2e_ip(self):
#        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
#        df.outcore.cholesky_eri(mol, ftmp.name, int3c='int3c2e_ip1',
#                                auxmol=auxmol, comp=3)
#        with h5py.File(ftmp.name, 'r') as f:
#            cderi0 = f['j3c'][:]


if __name__ == "__main__":
    print("Full Tests for df.outcore")
    unittest.main()
