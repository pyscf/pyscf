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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.UHF(mol)
mf.conv_tol = 1e-14
mf.scf()

n2sym = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = '''
        N     0    0    0
        N     0    0    1''',
    symmetry = 1,
    basis = 'cc-pvdz')
n2mf = scf.RHF(n2sym).set(conv_tol=1e-10).run()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = mf.init_guess_by_minao(mol, breaksym=False)
        self.assertAlmostEqual(abs(dm).sum(), 13.649710173723337, 9)
        dm = scf.uhf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 12.913908927027279, 9)

    def test_energy_tot(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, 57.122667754846844, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(numpy.linalg.norm(pop), 8.3342045408596057, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 12.322626374896178, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_nr_uhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.UHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

    def test_nr_uhf_symm_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        pmol.symmetry = 1
        pmol.build()
        mf = scf.UHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.uhf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 398.09239104094513, 9)

    def test_spin_square(self):
        self.assertAlmostEqual(mf.spin_square(mf.mo_coeff)[0], 0, 9)

    def test_rhf_to_uhf(self):
        scf.uhf.rhf_to_uhf(scf.RHF(mol))

    def test_uhf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.uhf_symm.UHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)

    def test_uhf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.uhf_symm.UHF(pmol)
        mf.irrep_nelec = {'B1':(2,1)}
        self.assertAlmostEqual(mf.scf(), -75.010623169610966, 9)

    def test_n2_symm(self):
        mf = scf.uhf_symm.UHF(n2sym)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)

        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.uhf_symm.UHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.34691774091894, 9)

    def test_n2_symm_uhf_fixnocc(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.uhf_symm.UHF(pmol)
        mf.irrep_nelec = {'A1g':6, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -108.22558478425401, 9)
        mf.irrep_nelec = {'A1g':(3,3), 'A1u':(2,1), 'E1ux':(1,1), 'E1uy':(1,1)}
        self.assertAlmostEqual(mf.scf(), -108.22558478425401, 9)

    def test_uhf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=8, spin=2)
        mf = scf.uhf.UHF(mol)
        energy = numpy.array(([-10, -1, 1, -2, 0, -3], [8, 2, 4, 3, 0, 5]))
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                                       ([1, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0])))
        pmol = n2sym.copy()
        pmol.spin = 2
        pmol.symmetry = False
        mf = scf.UHF(pmol).set(verbose = 0)
        energy = numpy.array([[34, 2 , 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7],
                              [29, 26, 13, 54, 18, 78, 85, 49, 84, 62, 42, 74, 20, 61, 51, 34, 2 , 33, 52, 3]])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [[0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))

    def test_uhf_symm_get_occ(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0)
        orbsym = numpy.array([[0 , 5 , 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5],
                              [5 , 0 , 6 , 7 , 5 , 10, 11, 0 , 5 , 0 , 5 , 5 , 6 , 7 , 0 , 2 , 3 , 0 , 2 , 3]])
        energy = numpy.array([[34, 2 , 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7],
                              [29, 26, 13, 54, 18, 78, 85, 49, 84, 62, 42, 74, 20, 61, 51, 34, 2 , 33, 52, 3]])
        mf.irrep_nelec = {'A1g':7, 'A1u':3, 'E1ux':2, 'E1uy':2}
        mo_coeff = lib.tag_array([numpy.eye(energy.size)]*2, orbsym=orbsym)
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                 [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]))
        mf.irrep_nelec = {'A1g':(5,2), 'A1u':(1,2)}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))

    def test_uhf_symm_dump_flags(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

    def test_det_ovlp(self):
        mf = scf.UHF(mol).run()
        s, x = mf.det_ovlp(mf.mo_coeff, mf.mo_coeff, mf.mo_occ, mf.mo_occ)
        self.assertAlmostEqual(s, 1.000000000, 9)
        self.assertAlmostEqual(numpy.trace(x[0]), mol.nelec[0]*1.000000000, 9)
        self.assertAlmostEqual(numpy.trace(x[0]), mol.nelec[1]*1.000000000, 9)

    def test_dip_moment(self):
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_get_wfnsym(self):
        self.assertEqual(n2mf.wfnsym, 0)

        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0).run()
        self.assertTrue(mf.wfnsym in (2, 3))

if __name__ == "__main__":
    print("Full Tests for uhf")
    unittest.main()

