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

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ["C", (-0.65830719,  0.61123287, -0.00800148)],
        ["C1", ( 0.73685281,  0.61123287, -0.00800148)],
        ["C2", ( 1.43439081,  1.81898387, -0.00800148)],
        ["C3", ( 0.73673681,  3.02749287, -0.00920048)],
        ["C4", (-0.65808819,  3.02741487, -0.00967948)],
        ["C5", (-1.35568919,  1.81920887, -0.00868348)],
        ["H", (-1.20806619, -0.34108413, -0.00755148)],
        ["H", ( 1.28636081, -0.34128013, -0.00668648)],
        ["H", ( 2.53407081,  1.81906387, -0.00736748)],
        ["H", ( 1.28693681,  3.97963587, -0.00925948)],
        ["H", (-1.20821019,  3.97969587, -0.01063248)],
        ["H", (-2.45529319,  1.81939187, -0.00886348)],]

    mol.basis = {'H': 'cc-pvdz',
                 'C1': 'CC PVDZ',
                 'C2': 'CC PVDZ',
                 'C3': 'cc-pVDZ',
                 'C4': gto.basis.parse('''
#BASIS SET: (9s,4p,1d) -> [3s,2p,1d]
    C    S
       6665.0000000              0.0006920             -0.0001460
       1000.0000000              0.0053290             -0.0011540
        228.0000000              0.0270770             -0.0057250
         64.7100000              0.1017180             -0.0233120
         21.0600000              0.2747400             -0.0639550
          7.4950000              0.4485640             -0.1499810
          2.7970000              0.2850740             -0.1272620
          0.5215000              0.0152040              0.5445290
    C    S
          0.1596000              1.0000000
    C    P
          9.4390000              0.0381090
          2.0020000              0.2094800
          0.5456000              0.5085570
    C    P
          0.1517000              1.0000000
    C    D
          0.5500000              1.0000000        '''),
                 'C': 'CC PVDZ',}
    mol.ecp = {'C1': 'LANL2DZ'}
    mol.build()

def tearDownModule():
    global mol
    del mol


def fp(mat):
    return abs(mat).sum()


class KnownValues(unittest.TestCase):
    def test_intor_nr(self):
        nao = mol.nao_nr()
        s0 = numpy.empty((3,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                buf = mol.intor_by_shell('int1e_ipovlp_sph', (i,j), comp=3)
                di,dj = buf.shape[1:]
                s0[:,ip:ip+di,jp:jp+dj] = buf
                jp += dj
            ip += di
        s = mol.intor('int1e_ipovlp_sph', comp=3)
        self.assertTrue(numpy.allclose(s,s0))
        self.assertAlmostEqual(fp(s), 960.67081839920604, 11)

    def test_intor_nr0(self):
        s = mol.intor('int1e_ovlp_sph')
        self.assertAlmostEqual(fp(s), 622.29059965181796, 11)

    def test_intor_nr1(self):
        s = mol.intor_symmetric('int1e_ovlp_sph')
        self.assertAlmostEqual(fp(s), 622.29059965181796, 11)

    def test_intor_nr2(self):
        s = mol.intor_asymmetric('int1e_ovlp_sph')
        self.assertAlmostEqual(fp(s), 622.29059965181796, 11)

    def test_intor_nr_cross(self):
        shls_slice = (0, mol.nbas//4, mol.nbas//4, mol.nbas)
        s = mol.intor('int1e_ovlp_sph', shls_slice=shls_slice)
        self.assertAlmostEqual(fp(s), 99.38188078749701, 11)

    def test_intor_r(self):
        s = mol.intor('int1e_ovlp_spinor')
        self.assertAlmostEqual(fp(s), 1592.2297864313475, 11)

    def test_intor_r1(self):
        s0 = mol.intor('int1e_ovlp_spinor')
        s1 = mol.intor_symmetric('int1e_ovlp_spinor')
        self.assertTrue(numpy.allclose(s1,s0))

    def test_intor_r2(self):
        s = mol.intor_asymmetric('int1e_ovlp_spinor')
        self.assertAlmostEqual(fp(s), 1592.2297864313475, 11)

    def test_intor_r_comp(self):
        s = mol.intor('int1e_ipkin_spinor')
        self.assertAlmostEqual(fp(s), 4409.86758420756, 10)
        s1 = mol.intor_asymmetric('int1e_ipkin_spinor')
        self.assertTrue(numpy.allclose(s1,s))

    def test_intor_nr2e(self):
        mol1 = gto.M(atom=[["O" , (0. , 0.     , 0.)],
                           [1   , (0. , -0.757 , 0.587)],
                           [1   , (0. , 0.757  , 0.587)]],
                     basis = '631g')
        nao = mol1.nao_nr()
        eri0 = numpy.empty((3,nao,nao,nao,nao))
        ip = 0
        for i in range(mol1.nbas):
            jp = 0
            for j in range(mol1.nbas):
                kp = 0
                for k in range(mol1.nbas):
                    lp = 0
                    for l in range(mol1.nbas):
                        buf = mol1.intor_by_shell('int2e_ip1_sph', (i,j,k,l), 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri0[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di
        buf = mol1.intor_by_shell('int2e_sph', (i,j,k,l))
        self.assertEqual(buf.ndim, 4)

        eri1 = mol1.intor('int2e_ip1_sph').reshape(3,13,13,13,13)
        self.assertTrue(numpy.allclose(eri0, eri1))

        idx = numpy.tril_indices(13)
        naopair = nao * (nao+1) // 2
        ref = eri0[:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s2ij')
        self.assertTrue(numpy.allclose(ref, eri1))

        idx = numpy.tril_indices(13)
        ref = eri0[:,:,:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s2kl')
        self.assertTrue(numpy.allclose(ref, eri1))

        idx = numpy.tril_indices(13)
        ref = eri0[:,idx[0],idx[1]][:,:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='4')
        self.assertTrue(numpy.allclose(ref, eri1))


        shls_slice = [0, 4, 2, 6, 1, 5, 3, 5]
        i0,i1,j0,j1,k0,k1,l0,l1 = mol1.ao_loc_nr()[shls_slice]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, shls_slice=shls_slice)
        eri1 = eri1.reshape(3,i1-i0,j1-j0,k1-k0,l1-l0)
        self.assertTrue(numpy.allclose(eri0[:,i0:i1,j0:j1,k0:k1,l0:l1], eri1))

        shls_slice = [5, 7, 7, 9, 1, 5, 4, 6]
        i0,i1,j0,j1,k0,k1,l0,l1 = mol1.ao_loc_nr()[shls_slice]
        idx = numpy.tril_indices(2)
        ref = eri0[:,i0:i1,j0:j1,k0:k1,l0:l1][:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s2ij', shls_slice=shls_slice)
        eri1 = eri1.reshape(3,-1,k1-k0,l1-l0)
        self.assertTrue(numpy.allclose(ref, eri1))

        shls_slice = [1, 5, 4, 6, 5, 7, 7, 9]
        i0,i1,j0,j1,k0,k1,l0,l1 = mol1.ao_loc_nr()[shls_slice]
        idx = numpy.tril_indices(2)
        ref = eri0[:,i0:i1,j0:j1,k0:k1,l0:l1][:,:,:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s2kl', shls_slice=shls_slice)
        eri1 = eri1.reshape(3,i1-i0,j1-j0,-1)
        self.assertTrue(numpy.allclose(ref, eri1))

        shls_slice = [5, 7, 7, 9, 7, 9, 5, 7]
        i0,i1,j0,j1,k0,k1,l0,l1 = mol1.ao_loc_nr()[shls_slice]
        idx = numpy.tril_indices(2)
        ref = eri0[:,i0:i1,j0:j1,k0:k1,l0:l1][:,idx[0],idx[1]][:,:,idx[0],idx[1]]
        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s4', shls_slice=shls_slice)
        self.assertTrue(numpy.allclose(ref, eri1))

        eri1 = mol1.intor('int2e_ip1_sph', comp=3, aosym='s1', shls_slice=(2,4,1,3))
        self.assertEqual(eri1.shape, (3,4,2,13,13))
        self.assertTrue(numpy.allclose(eri0[:,2:6,1:3], eri1))

    def test_intor_r2e(self):
        mol1 = gto.M(atom=[[1   , (0. , -0.7 , 0.)],
                           [1   , (0. ,  0.7 , 0.)]],
                     basis = '631g')
        nao = mol1.nao_2c()
        eri0 = numpy.empty((3,nao,nao,nao,nao), dtype=numpy.complex128)
        ip = 0
        for i in range(mol1.nbas):
            jp = 0
            for j in range(mol1.nbas):
                kp = 0
                for k in range(mol1.nbas):
                    lp = 0
                    for l in range(mol1.nbas):
                        buf = mol1.intor_by_shell('int2e_ip1_spinor', (i,j,k,l), comp=3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri0[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di

    def test_input_cint(self):
        '''Compatibility to old cint functions
        '''
        self.assertEqual(gto.moleintor.ascint3('cint2e'), 'int2e_spinor')
        self.assertEqual(gto.moleintor.ascint3('cint2e_sph'), 'int2e_sph')

    def test_rinv_with_zeta(self):
        with mol.with_rinv_orig((.2,.3,.4)), mol.with_rinv_zeta(2.2):
            v1 = mol.intor('int1e_rinv_sph')
        pmol = gto.M(atom='Ghost .2 .3 .4', unit='b', basis={'Ghost':[[0,(2.2*.5, 1)]]})
        pmol._atm, pmol._bas, pmol._env = \
            gto.conc_env(mol._atm, mol._bas, mol._env,
                         pmol._atm, pmol._bas, pmol._env)
        shls_slice = (pmol.nbas-1,pmol.nbas, pmol.nbas-1,pmol.nbas,
                      0, pmol.nbas, 0, pmol.nbas)
        v0 = pmol.intor('int2e_sph', shls_slice=shls_slice)
        nao = pmol.nao_nr()
        v0 = v0.reshape(nao,nao)[:-1,:-1]
        self.assertTrue(numpy.allclose(v0, v1))

    def test_intor_nr3c(self):
        mol = gto.M(verbose = 0, output = None,
            atom = '''H 0  1 .5
                      H 1 .8 1.1
                      H .2 1.8 0
                      H .5 .8 .1''',
            basis = 'cc-pvdz')
        nao = mol.nao_nr()
        eri0 = numpy.empty((nao,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    buf = mol.intor_by_shell('int3c1e_sph', (i,j,k))
                    di,dj,dk = buf.shape
                    eri0[ip:ip+di,jp:jp+dj,kp:kp+dk] = buf
                    kp += dk
                jp += dj
            ip += di
        buf = mol.intor_by_shell('int3c1e_sph', (i,j,k))
        self.assertEqual(buf.ndim, 3)

        eri1 = mol.intor('int3c1e_sph')
        self.assertTrue(numpy.allclose(eri0, eri1))

        idx = numpy.tril_indices(nao)
        eri1 = mol.intor('int3c1e_sph', aosym='s2ij')
        self.assertTrue(numpy.allclose(eri0[idx], eri1))

        eri1 = mol.intor('int3c1e_sph', aosym='s2ij',
                         shls_slice=(2,5,0,5,0,mol.nbas))
        self.assertTrue(numpy.allclose(eri0[idx][3:28], eri1))

        eri1 = mol.intor('int3c1e_sph', shls_slice=(2,5,4,9,0,mol.nbas))
        self.assertTrue(numpy.allclose(eri0[2:7,6:15], eri1))

        eri1 = mol.intor('int3c2e_ip1_sph', comp=3, shls_slice=(2,5,4,9,0,mol.nbas))
        self.assertEqual(eri1.shape, (3,5,9,nao))
        self.assertAlmostEqual(fp(eri1), 642.70512922279079, 11)

        eri1 = mol.intor('int3c2e_spinor')
        self.assertEqual(eri1.shape, (nao*2,nao*2,nao))
        self.assertAlmostEqual(fp(eri1), 9462.9659834308495, 11)

#        eri1 = mol.intor('int3c2e_spinor_ssc')
#        self.assertEqual(eri1.shape, (nao*2,nao*2,20))
#        self.assertAlmostEqual(fp(eri1), 1227.1971656655824, 11)

    def test_nr_s8(self):
        mol = gto.M(atom="He 0 0 0; Ne 3 0 0", basis='ccpvdz')
        eri0 = mol.intor('int2e_sph', aosym='s8')
        self.assertAlmostEqual(lib.fp(eri0), -10.685918926843847, 9)

    # FIXME
    #def test_nr_s8_skip(self):
    #    eri1 = mol.intor('int2e_yp_sph', aosym=8)
    #    self.assertAlmostEqual(lib.fp(eri1), -10.685918926843847, 9)
    #    self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 9)

    def test_unknonw(self):
        self.assertRaises(KeyError, mol.intor, 'int4c3e')

    def test_nr_2c2e(self):
        mat = mol.intor('int2c2e')
        self.assertAlmostEqual(lib.fp(mat), -460.83033192375615, 9)


if __name__ == "__main__":
    unittest.main()
