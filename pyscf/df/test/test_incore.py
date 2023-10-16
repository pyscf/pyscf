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
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import df

def setUpModule():
    global mol, auxmol, atm, bas, env
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

    auxmol = df.addons.make_auxmol(mol, 'weigend')
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                 auxmol._atm, auxmol._bas, auxmol._env)

def tearDownModule():
    global mol, auxmol, atm, bas, env
    del mol, auxmol, atm, bas, env


class KnownValues(unittest.TestCase):
    def test_aux_e2(self):
        nao = mol.nao_nr()
        naoaux = auxmol.nao_nr()
        eri0 = numpy.empty((nao,nao,naoaux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('int3c2e_sph',
                                                         shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di

        j3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
        self.assertTrue(numpy.allclose(eri0, j3c.reshape(nao,nao,naoaux)))
        self.assertAlmostEqual(lib.finger(j3c), 45.27912877994409, 9)

        idx = numpy.tril_indices(nao)
        j3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(eri0[idx], j3c))
        self.assertAlmostEqual(lib.finger(j3c), 12.407403711205063, 9)

    def test_aux_e1(self):
        j3c1 = df.incore.aux_e1(mol, auxmol, intor='int3c2e', aosym='s2ij')
        j3c2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s2ij')
        self.assertAlmostEqual(abs(j3c1.T-j3c2).max(), 0, 12)

        j3c1 = df.incore.aux_e1(mol, auxmol, intor='int3c2e', aosym='s1')
        j3c2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s1')
        self.assertAlmostEqual(abs(j3c1.transpose(2,0,1)-j3c2).max(), 0, 12)

    def test_aux_e2_diff_bra_ket(self):
        mol1 = mol.copy()
        mol1.basis = 'sto3g'
        mol1.build(0, 0, verbose=0)
        atm1, bas1, env1 = gto.conc_env(atm, bas, env,
                                        mol1._atm, mol1._bas, mol1._env)
        ao_loc = gto.moleintor.make_loc(bas1, 'int3c2e_sph')
        shls_slice = (0, mol.nbas,
                      mol.nbas+auxmol.nbas, mol.nbas+auxmol.nbas+mol1.nbas,
                      mol.nbas, mol.nbas+auxmol.nbas)

        j3c = gto.moleintor.getints3c('int3c2e_sph', atm1, bas1, env1, comp=1,
                                      shls_slice=shls_slice, aosym='s1', ao_loc=ao_loc)

        nao = mol.nao_nr()
        naoj = mol1.nao_nr()
        naoaux = auxmol.nao_nr()
        eri0 = numpy.empty((nao,naoj,naoaux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas+auxmol.nbas, len(bas1)):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('int3c2e_sph',
                                                         shls, atm1, bas1, env1)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j3c))

    def test_cholesky_eri(self):
        j2c = df.incore.fill_2c2e(mol, auxmol)
        eri0 = numpy.empty_like(j2c)
        pi = 0
        for i in range(mol.nbas, len(bas)):
            pj = 0
            for j in range(mol.nbas, len(bas)):
                shls = (i, j)
                buf = gto.moleintor.getints_by_shell('int2c2e_sph',
                                                     shls, atm, bas, env)
                di, dj = buf.shape
                eri0[pi:pi+di,pj:pj+dj] = buf
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j2c))

        j3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s2ij')
        cderi = df.incore.cholesky_eri(mol)
        eri0 = numpy.einsum('pi,pk->ik', cderi, cderi)
        eri1 = numpy.einsum('ik,kl->il', j3c, numpy.linalg.inv(j2c))
        eri1 = numpy.einsum('ip,kp->ik', eri1, j3c)
        self.assertTrue(numpy.allclose(eri1, eri0))

        cderi1 = df.incore.cholesky_eri_debug(mol)
        self.assertAlmostEqual(abs(cderi-cderi1).max(), 0, 9)

    def test_r_incore(self):
        j3c = df.r_incore.aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym='s1')
        nao = mol.nao_2c()
        naoaux = auxmol.nao_nr()
        j3c = j3c.reshape(nao,nao,naoaux)

        eri0 = numpy.empty((nao,nao,naoaux), dtype=numpy.complex128)
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.moleintor.getints_by_shell('int3c2e_spinor',
                                                         shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di
        self.assertTrue(numpy.allclose(eri0, j3c))
        eri1 = df.r_incore.aux_e2(mol, auxmol, intor='int3c2e_spinor',
                                  aosym='s2ij')
        for i in range(naoaux):
            j3c[:,:,i] = lib.unpack_tril(eri1[:,i])
        self.assertTrue(numpy.allclose(eri0, j3c))

    def test_lindep(self):
        cderi0 = df.incore.cholesky_eri(mol, auxmol=auxmol)
        auxmol1 = auxmol.copy()
        auxmol1.basis = {'O': 'weigend', 'H': ('weigend', 'weigend')}
        auxmol1.build(0, 0)
        cderi1 = df.incore.cholesky_eri(mol, auxmol=auxmol1)
        eri0 = numpy.dot(cderi0.T, cderi0)
        eri1 = numpy.dot(cderi1.T, cderi1)
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 9)



if __name__ == "__main__":
    print("Full Tests for df.incore")
    unittest.main()
