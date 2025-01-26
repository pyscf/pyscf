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
import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf, tools
from pyscf.pbc.tools import k2gamma

def setUpModule():
    global cell, mf, kpts
    cell = gto.Cell()
    cell.a = '''
         1.755000    1.755000    -1.755000
         1.755000    -1.755000    1.755000
         -1.755000    1.755000    1.755000'''
    cell.atom = '''Li      0.00000      0.00000      0.00000'''
    #same type of basis for different elements
    cell.basis = 'gth-szv'
    cell.pseudo = {'Li': 'GTH-PBE-q3'}
    cell.mesh = [20]*3
    cell.verbose = 6
    cell.output = '/dev/null'
    cell.build()

    kpts = cell.make_kpts([2,2,2])

    mf = scf.KUKS(cell, kpts)
    mf.xc = 'lda,vwn'
    mf.kernel()

def tearDownModule():
    global cell, mf
    cell.stdout.close()
    del cell, mf


class KnownValues(unittest.TestCase):
    def test_k2gamma(self):
        cell = gto.Cell()
        cell.a = '''
             1.755000    1.755000    -1.755000
             1.755000    -1.755000    1.755000
             -1.755000    1.755000    1.755000'''
        cell.atom = '''Li      0.00000      0.00000      0.00000'''
        cell.basis = 'gth-szv'
        cell.pseudo = {'Li': 'GTH-PBE-q3'}
        cell.mesh = [20]*3
        cell.verbose = 6
        cell.output = '/dev/null'
        cell.build()

        kpts = cell.make_kpts([2,2,2])
        mf = scf.KUKS(cell, kpts)
        mf.xc = 'lda,vwn'
        mf.kernel()

        popa, popb = mf.mulliken_meta()[0]
        self.assertAlmostEqual(lib.fp(popa), 1.2700920989, 7)
        self.assertAlmostEqual(lib.fp(popb), 1.2700920989, 7)

        popa, popb = k2gamma.k2gamma(mf).mulliken_meta()[0]
        self.assertAlmostEqual(lib.fp(popa), 0.8007278745, 7)
        self.assertAlmostEqual(lib.fp(popb), 0.8007278745, 7)

    def test_k2gamma_ksymm(self):
        cell = gto.Cell()
        cell.atom = '''
            He 0.  0. 0.
        '''
        cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
        cell.a = np.eye(3) * 2.
        cell.space_group_symmetry = True
        cell.build()

        kmesh = [2,2,1]
        kpts = cell.make_kpts(kmesh, space_group_symmetry=True)
        kmf = scf.KRKS(cell, kpts).density_fit()
        kmf.kernel()
        mf = k2gamma.k2gamma(kmf)
        c_g_ao = mf.mo_coeff

        scell = tools.super_cell(cell, kmesh)
        mf_sc = scf.RKS(scell).density_fit()
        self.assertEqual(mf.__class__, mf_sc.__class__)
        self.assertEqual(mf.xc, kmf.xc)

        s = mf_sc.get_ovlp()
        mf_sc.run()
        sc_mo = mf_sc.mo_coeff

        one = np.linalg.det(c_g_ao.T.conj().dot(s).dot(sc_mo))
        self.assertAlmostEqual(abs(one), 1., 9)

    def test_double_translation_indices(self):
        idx2 = k2gamma.translation_map(2)
        idx3 = k2gamma.translation_map(3)
        idx4 = k2gamma.translation_map(4)

        ref = np.empty((2, 3, 4, 2, 3, 4), dtype=int)
        for ix in range(2):
            for iy in range(3):
                for iz in range(4):
                    for jx in range(2):
                        for jy in range(3):
                            for jz in range(4):
                                ref[ix,iy,iz,jx,jy,jz] = idx2[ix,jx] * 12 + idx3[iy,jy] * 4 + idx4[iz,jz]

        result = k2gamma.double_translation_indices([2,3,4])
        self.assertEqual(abs(ref.reshape(24,24) - result).max(), 0)

    def test_kpts_to_kmesh(self):
        cell = gto.M(atom='He 0 0 0', basis=[[0, (.3, 1)]], a=np.eye(3), verbose=0)
        cell.rcut = 38
        self.assertEqual(cell.nimgs.tolist(), [21, 21, 21])
        scaled_kpts = np.array([
                [0. ,  0,  0],
                [0.5, -.5, .25],
                [0.5, .25, .333333],
        ])
        kpts = cell.get_abs_kpts(scaled_kpts)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        self.assertEqual(kmesh.tolist(), [2, 4, 12])

        cell.rcut = 9
        self.assertEqual(cell.nimgs.tolist(), [5, 5, 5])
        scaled_kpts = np.array([
                [0. ,  0,  0],
                [0.5, -.5, .25],
                [0.5, .25, .3333],
        ])
        kpts = cell.get_abs_kpts(scaled_kpts)
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        self.assertEqual(kmesh.tolist(), [2, 4, 11])

        cell = gto.M(atom='He 0 0 0', basis=[[0, (2.3, 1)]], a=np.eye(3)*3, verbose=0)
        kpts = cell.make_kpts([6,1,1])
        kmesh = k2gamma.kpts_to_kmesh(cell, kpts)
        self.assertEqual(kmesh.tolist(), [3, 1, 1])


if __name__ == '__main__':
    print("Full Tests for pbc.tools.k2gamma")
    unittest.main()
