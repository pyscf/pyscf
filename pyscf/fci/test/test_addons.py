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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

mol = gto.Mole()
mol.verbose = 0
mol.atom = '''
      H     1  -1.      0
      H     0  -1.     -1
      H     0  -0.5    -0
      H     0  -0.     -1
      H     1  -0.5     0
      H     0   1.      1'''
mol.basis = 'sto-3g'
mol.build()
m = scf.RHF(mol)
m.conv_tol = 1e-15
ehf = m.scf()
norb = m.mo_coeff.shape[1]
nelec = mol.nelectron
h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
g2e = ao2mo.incore.general(m._eri, (m.mo_coeff,)*4, compact=False)
na = fci.cistring.num_strings(norb, nelec//2)
e, ci0 = fci.direct_spin1.kernel(h1e, g2e, norb, nelec, tol=1e-15)

class KnownValues(unittest.TestCase):
    def test_large_ci(self):
        res = fci.addons.large_ci(ci0, norb, nelec, tol=.1)
        refstr =[('0b111'  , '0b111'  ),
                 ('0b111'  , '0b1011' ),
                 ('0b1011' , '0b111'  ),
                 ('0b1011' , '0b1011' ),
                 ('0b10101', '0b10101')]
        refci = [0.86848550920009038, 0.15130668599599939, 0.15130668599597297,
                 0.36620088911284837, 0.10306162063159749]
        self.assertTrue(numpy.allclose([abs(x[0]) for x in res], refci))
        self.assertEqual([x[1:] for x in res], refstr)

        res = fci.addons.large_ci(ci0, norb, nelec, tol=.1, return_strs=False)
        refa = numpy.array(((0,1,2), (0,1,2), (0,1,3), (0,1,3), (0,2,4)))
        refb = numpy.array(((0,1,2), (0,1,3), (0,1,2), (0,1,3), (0,2,4)))
        self.assertTrue(numpy.all([x[1] for x in res] == refa))
        self.assertTrue(numpy.all([x[2] for x in res] == refb))

    def test__init__file(self):
        c1 = fci.FCI(mol, m.mo_coeff)
        self.assertAlmostEqual(c1.kernel()[0], -2.8227809167209683, 9)

    def test_init_triplet(self):
        ci1 = fci.addons.initguess_triplet(norb, nelec, '0b1011')
        self.assertAlmostEqual(abs(ci1 + ci1.T).sum(), 0)
        self.assertTrue(ci1[0,1] < 0)

    def test_credes_ab(self):
        a4 = 10*numpy.arange(4)[:,None]
        a6 = 10*numpy.arange(6)[:,None]
        b4 = numpy.arange(4)
        b6 = numpy.arange(6)
        self.assertTrue(numpy.allclose(fci.addons.des_a(a4+b4, 4, (3,3), 0),
                                        [[  0.,  0.,  0.,  0.],
                                         [  0.,  0.,  0.,  0.],
                                         [  0.,  1.,  2.,  3.],
                                         [  0.,  0.,  0.,  0.],
                                         [ 10., 11., 12., 13.],
                                         [ 20., 21., 22., 23.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_a(a4+b4, 4, (3,3), 1),
                                        [[  0.,  0.,  0.,  0.],
                                         [  0., -1., -2., -3.],
                                         [  0.,  0.,  0.,  0.],
                                         [-10.,-11.,-12.,-13.],
                                         [  0.,  0.,  0.,  0.],
                                         [ 30., 31., 32., 33.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_a(a4+b4, 4, (3,3), 2),
                                        [[  0.,  1.,  2.,  3.],
                                         [  0.,  0.,  0.,  0.],
                                         [  0.,  0.,  0.,  0.],
                                         [-20.,-21.,-22.,-23.],
                                         [-30.,-31.,-32.,-33.],
                                         [  0.,  0.,  0.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_a(a4+b4, 4, (3,3), 3),
                                        [[ 10., 11., 12., 13.],
                                         [ 20., 21., 22., 23.],
                                         [ 30., 31., 32., 33.],
                                         [  0.,  0.,  0.,  0.],
                                         [  0.,  0.,  0.,  0.],
                                         [  0.,  0.,  0.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_b(a6+b4, 4, (2,3), 0),
                                        [[  0.,  0.,  0.,  0.,  1.,  2.],
                                         [  0.,  0., 10.,  0., 11., 12.],
                                         [  0.,  0., 20.,  0., 21., 22.],
                                         [  0.,  0., 30.,  0., 31., 32.],
                                         [  0.,  0., 40.,  0., 41., 42.],
                                         [  0.,  0., 50.,  0., 51., 52.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_b(a6+b4, 4, (2,3), 1),
                                        [[  0.,  0.,  0., -1.,  0.,  3.],
                                         [  0.,-10.,  0.,-11.,  0., 13.],
                                         [  0.,-20.,  0.,-21.,  0., 23.],
                                         [  0.,-30.,  0.,-31.,  0., 33.],
                                         [  0.,-40.,  0.,-41.,  0., 43.],
                                         [  0.,-50.,  0.,-51.,  0., 53.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_b(a6+b4, 4, (2,3), 2),
                                        [[  0.,  0.,  0., -2., -3.,  0.],
                                         [ 10.,  0.,  0.,-12.,-13.,  0.],
                                         [ 20.,  0.,  0.,-22.,-23.,  0.],
                                         [ 30.,  0.,  0.,-32.,-33.,  0.],
                                         [ 40.,  0.,  0.,-42.,-43.,  0.],
                                         [ 50.,  0.,  0.,-52.,-53.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.des_b(a6+b4, 4, (2,3), 3),
                                        [[  1.,  2.,  3.,  0.,  0.,  0.],
                                         [ 11., 12., 13.,  0.,  0.,  0.],
                                         [ 21., 22., 23.,  0.,  0.,  0.],
                                         [ 31., 32., 33.,  0.,  0.,  0.],
                                         [ 41., 42., 43.,  0.,  0.,  0.],
                                         [ 51., 52., 53.,  0.,  0.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_a(a6+b4, 4, (2,3), 0),
                                        [[ 20., 21., 22., 23.],
                                         [ 40., 41., 42., 43.],
                                         [ 50., 51., 52., 53.],
                                         [  0.,  0.,  0.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_a(a6+b4, 4, (2,3), 1),
                                        [[-10.,-11.,-12.,-13.],
                                         [-30.,-31.,-32.,-33.],
                                         [  0.,  0.,  0.,  0.],
                                         [ 50., 51., 52., 53.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_a(a6+b4, 4, (2,3), 2),
                                        [[  0.,  1.,  2.,  3.],
                                         [  0.,  0.,  0.,  0.],
                                         [-30.,-31.,-32.,-33.],
                                         [-40.,-41.,-42.,-43.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_a(a6+b4, 4, (2,3), 3),
                                        [[  0.,  0.,  0.,  0.],
                                         [  0.,  1.,  2.,  3.],
                                         [ 10., 11., 12., 13.],
                                         [ 20., 21., 22., 23.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_b(a6+b6, 4, (2,2), 0),
                                        [[  2.,  4.,  5.,  0.],
                                         [ 12., 14., 15.,  0.],
                                         [ 22., 24., 25.,  0.],
                                         [ 32., 34., 35.,  0.],
                                         [ 42., 44., 45.,  0.],
                                         [ 52., 54., 55.,  0.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_b(a6+b6, 4, (2,2), 1),
                                        [[ -1., -3.,  0.,  5.],
                                         [-11.,-13.,  0., 15.],
                                         [-21.,-23.,  0., 25.],
                                         [-31.,-33.,  0., 35.],
                                         [-41.,-43.,  0., 45.],
                                         [-51.,-53.,  0., 55.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_b(a6+b6, 4, (2,2), 2),
                                        [[  0.,  0., -3., -4.],
                                         [ 10.,  0.,-13.,-14.],
                                         [ 20.,  0.,-23.,-24.],
                                         [ 30.,  0.,-33.,-34.],
                                         [ 40.,  0.,-43.,-44.],
                                         [ 50.,  0.,-53.,-54.]]))
        self.assertTrue(numpy.allclose(fci.addons.cre_b(a6+b6, 4, (2,2), 3),
                                        [[  0.,  0.,  1.,  2.],
                                         [  0., 10., 11., 12.],
                                         [  0., 20., 21., 22.],
                                         [  0., 30., 31., 32.],
                                         [  0., 40., 41., 42.],
                                         [  0., 50., 51., 52.]]))
    def test_spin_squre(self):
        ss = fci.spin_op.spin_square(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 0, 9)
        ss = fci.spin_op.spin_square0(ci0, norb, nelec)
        self.assertAlmostEqual(ss[0], 0, 9)

    def test_fix_spin(self):
        mci = fci.FCI(mol, m.mo_coeff, False)
        mci = fci.addons.fix_spin_(mci, .2, 0)
        mci.kernel(nelec=(3,3))
        self.assertAlmostEqual(mci.spin_square(mci.ci, mol.nao_nr(), (3,3))[0], 0, 7)

    def test_fix_spin_high_cost(self):
        def check(mci):
            mci = fci.addons.fix_spin_(mci, .2, 0)
            mci.kernel(nelec=(8,8))
            self.assertAlmostEqual(mci.spin_square(mci.ci, mol.nao_nr(), 16)[0], 0, 7)

        mol = gto.M(atom='O 0 0 0; O 0 0 1.2', spin=2, basis='sto3g',
                    symmetry=1, verbose=0)
        mf = scf.RHF(mol).run()
        mci = fci.FCI(mol, mf.mo_coeff, False)
        mci.wfnsym = 'A1g'
        check(mci)
        mci.wfnsym = 'A2g'
        check(mci)

        mci = fci.FCI(mol, mf.mo_coeff, True)
        mci.wfnsym = 'A1g'
        check(mci)
        mci.wfnsym = 'A2g'
        check(mci)

        mol = gto.M(atom='O 0 0 0; O 0 0 1.2', spin=2, basis='sto3g',
                    verbose=0)
        mf = scf.RHF(mol).run()
        mci = fci.FCI(mol, mf.mo_coeff, False)
        check(mci)
        mci = fci.FCI(mol, mf.mo_coeff, True)
        check(mci)

    def test_transform_ci_for_orbital_rotation(self):
        numpy.random.seed(12)
        norb = nelec = 6
        u = numpy.linalg.svd(numpy.random.random((norb,norb)))[0]
        mo1 = m.mo_coeff.dot(u)
        h1e_new = reduce(numpy.dot, (mo1.T, m.get_hcore(), mo1))
        g2e_new = ao2mo.incore.general(m._eri, (mo1,)*4, compact=False)
        e1ref, ci1ref = fci.direct_spin1.kernel(h1e_new, g2e_new, norb, nelec, tol=1e-15)
        ci1 = fci.addons.transform_ci_for_orbital_rotation(ci0, norb, nelec, u)
        e1 = fci.direct_spin1.energy(h1e_new, g2e_new, ci1, norb, nelec)
        self.assertAlmostEqual(e1, e1ref, 9)
        self.assertAlmostEqual(abs(abs(ci1ref)-abs(ci1)).sum(), 0, 9)

    def test_guess_wfnsym(self):
        orbsym = [2,3,6,7]
        wfnsym = fci.addons.guess_wfnsym(numpy.array([-.5,0,0,0,0,0]),
                                         len(orbsym), (4,2), orbsym)
        self.assertEqual(wfnsym, 1)
        orbsym = [2,3,6,7]
        wfnsym = fci.addons.guess_wfnsym(numpy.array([-.5,0,.5,0,0,0]),
                                         len(orbsym), (4,2), orbsym)
        self.assertEqual(wfnsym, 1)


if __name__ == "__main__":
    print("Full Tests for fci.addons")
    unittest.main()


