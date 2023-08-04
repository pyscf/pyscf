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
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci

def setUpModule():
    global mol, m, h1e, g2e, ci0, norb, nelec
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

def tearDownModule():
    global mol, m, h1e, g2e, ci0
    del mol, m, h1e, g2e, ci0

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

        na = fci.cistring.num_strings(6, 3)
        numpy.random.seed(9)
        ci1 = numpy.random.random((na,na))
        ci1 /= numpy.linalg.norm(ci1)
        res = fci.addons.large_ci(ci1, 6, (3,3), tol=.2)
        self.assertEqual(res[0][1:], ('0b110100', '0b1101'))

    def test__init__file(self):
        c1 = fci.FCI(mol, m.mo_coeff, singlet=True)
        self.assertAlmostEqual(c1.kernel()[0], -2.8227809167209683, 9)

        c1 = fci.FCI(m, singlet=True)
        self.assertAlmostEqual(c1.kernel()[0], -2.8227809167209683, 9)

        c1 = fci.FCI(mol.UHF().run())
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

        mci = fci.addons.fix_spin_(mci, .2, ss=2)
        # Change initial guess to triplet state
        ci0 = fci.addons.initguess_triplet(norb, (3,3), '0b10011')
        mci.kernel(nelec=(3,3), ci0=ci0)
        self.assertAlmostEqual(mci.spin_square(mci.ci, mol.nao_nr(), (3,3))[0], 2, 7)

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
        mci.wfnsym = 'A1u'
        check(mci)

        mci = fci.FCI(mol, mf.mo_coeff, True)
        mci.wfnsym = 'A1g'
        check(mci)
        mci.wfnsym = 'A1u'
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
        norb = 6
        nelec = (4,2)
        u = numpy.linalg.svd(numpy.random.random((norb,norb)))[0]
        mo1 = m.mo_coeff.dot(u)
        h1e_new = reduce(numpy.dot, (mo1.T, m.get_hcore(), mo1))
        g2e_new = ao2mo.incore.general(m._eri, (mo1,)*4, compact=False)
        e1ref, ci1ref = fci.direct_spin1.kernel(h1e_new, g2e_new, norb, nelec, tol=1e-15)

        ci0 = fci.direct_spin1.kernel(h1e, g2e, norb, nelec)[1]
        ci1 = fci.addons.transform_ci_for_orbital_rotation(ci0, norb, nelec, u)
        e1 = fci.direct_spin1.energy(h1e_new, g2e_new, ci1, norb, nelec)
        self.assertAlmostEqual(e1, e1ref, 9)
        self.assertAlmostEqual(abs(abs(ci1ref)-abs(ci1)).sum(), 0, 9)

    def test_transform_ci(self):
        # transform for different orbital-space sizes
        numpy.random.seed(12)
        norb0, norb1 = 6, 7
        nelec = (4,2)
        na0 = fci.cistring.num_strings(norb0, nelec[0])
        nb0 = fci.cistring.num_strings(norb0, nelec[1])
        na1 = fci.cistring.num_strings(norb1, nelec[0])
        nb1 = fci.cistring.num_strings(norb1, nelec[1])
        ci0 = numpy.random.random((na0, nb0))
        u = numpy.zeros((norb0, norb1))
        u[:norb0,:norb0] = numpy.eye(norb0)

        ci1_ref = numpy.zeros((na1, nb1))
        ci1_ref[:na0,:na0] = ci0
        ci1 = fci.addons.transform_ci(ci0, nelec, u)
        self.assertAlmostEqual(abs(ci1 - ci1_ref).max(), 0, 14)

        u = numpy.linalg.svd(numpy.random.random((norb0, norb1)))[0]
        ci1 = fci.addons.transform_ci(ci0, nelec, u)
        ci0_new = fci.addons.transform_ci(ci1, nelec, u.T)
        self.assertAlmostEqual(abs(ci0 - ci0_new).max(), 0, 14)

    def test_overlap(self):
        numpy.random.seed(12)
        s = numpy.random.random((6,6))
        s = s.dot(s.T) / 3
        bra = numpy.random.random((15,15))
        ket = numpy.random.random((15,15))
        bra /= numpy.linalg.norm(bra)
        ket /= numpy.linalg.norm(ket)
        self.assertAlmostEqual(fci.addons.overlap(bra, ket, 6, 4), 0.7767249258737043, 9)
        self.assertAlmostEqual(fci.addons.overlap(bra, ket, 6, 4, (s,s)), 0.025906419720918766, 9)

        norb = 4
        nelec = (1,0)
        ua = numpy.linalg.svd(numpy.random.random((norb+1,norb+1)))[0]
        ub = numpy.linalg.svd(numpy.random.random((norb+1,norb+1)))[0]
        s = numpy.dot(ua[:,:norb].T, ub[:,:norb])
        ci0 = numpy.random.random((norb,1))
        ci0 /= numpy.linalg.norm(ci0)
        ci1 = numpy.random.random((norb,1))
        ci1 /= numpy.linalg.norm(ci1)
        ovlp = fci.addons.overlap(ci0, ci1, norb, nelec, s)
        self.assertAlmostEqual(ovlp, (ci0*ci1.T*s).sum(), 9)

    def test_det_overlap(self):
        numpy.random.seed(12)
        norb = 4
        nelec = (2,2)
        ua = numpy.linalg.svd(numpy.random.random((norb+1,norb+1)))[0]
        ub = numpy.linalg.svd(numpy.random.random((norb+1,norb+1)))[0]
        s = numpy.dot(ua[:,:norb].T, ub[:,:norb])

        strs = fci.cistring.make_strings(range(norb), nelec[0])
        na = len(strs)
        ci0 = numpy.random.random((na,na))
        ci0 /= numpy.linalg.norm(ci0)
        ci1 = numpy.random.random((na,na))
        ci1 /= numpy.linalg.norm(ci1)

        ovlpa = numpy.zeros((na,na))
        ovlpb = numpy.zeros((na,na))
        for ia in range(na):
            for ja in range(na):
                ovlpa[ia,ja] = fci.addons.det_overlap(strs[ia], strs[ja], norb, s)
        for ib in range(na):
            for jb in range(na):
                ovlpb[ib,jb] = fci.addons.det_overlap(strs[ib], strs[jb], norb, s)
        ovlp = numpy.einsum('ab,ij,ai,bj->', ci0, ci1, ovlpa, ovlpb)

        ref = fci.addons.overlap(ci0, ci1, norb, nelec, s)
        self.assertAlmostEqual(ovlp, ref, 9)

        s1 = numpy.random.seed(1)
        s1 = numpy.random.random((6,6))
        s1 = s1 + s1.T
        val = fci.addons.det_overlap(int('0b10011',2), int('0b011010',2), 6, s1)
        self.assertAlmostEqual(val, -0.273996425116, 12)

    def test_guess_wfnsym(self):
        orbsym = [2,3,6,7]
        wfnsym = fci.addons.guess_wfnsym(numpy.array([-.5,0,0,0,0,0]),
                                         len(orbsym), (4,2), orbsym)
        self.assertEqual(wfnsym, 1)
        orbsym = [2,3,6,7]
        wfnsym = fci.addons.guess_wfnsym(numpy.array([-.5,0,.5,0,0,0]),
                                         len(orbsym), (4,2), orbsym)
        self.assertEqual(wfnsym, 1)

    def test_cylindrical_init_guess(self):
        mol = gto.M(atom='O; O 1 1.2', spin=2, symmetry=True)
        orbsym = [6,7,2,3]
        ci0 = fci.addons.cylindrical_init_guess(mol, 4, (3,3), orbsym, wfnsym=10)
        ci0 = ci0[0].reshape(4,4)
        self.assertAlmostEqual(ci0[0,0],  .5**.5, 12)
        self.assertAlmostEqual(ci0[1,1], -.5**.5, 12)

        ci0 = fci.addons.cylindrical_init_guess(mol, 4, (3,3), orbsym, wfnsym=10, singlet=False)
        ci0 = ci0[0].reshape(4,4)
        self.assertAlmostEqual(ci0[0,1],  .5**.5, 12)
        self.assertAlmostEqual(ci0[1,0], -.5**.5, 12)

    def test_symmetrize_wfn(self):
        def finger(ci1):
            numpy.random.seed(1)
            fact = numpy.random.random(ci1.shape).ravel()
            return numpy.dot(ci1.ravel(), fact.ravel())
        norb = 6
        nelec = neleca, nelecb = 4,3
        na = fci.cistring.num_strings(norb, neleca)
        nb = fci.cistring.num_strings(norb, nelecb)
        ci = numpy.ones((na,nb))
        val = finger(fci.addons.symmetrize_wfn(ci, norb, nelec, [0,6,0,3,5,2], 2))
        self.assertAlmostEqual(val, 3.010642818688976, 12)

    def test_des_and_cre(self):
        a4 = 10*numpy.arange(4)[:,None]
        a6 = 10*numpy.arange(6)[:,None]
        b4 = numpy.arange(4)
        b6 = numpy.arange(6)

        self.assertAlmostEqual(lib.fp(fci.addons.des_a(a4+b4, 4, (3,3), 0)), -31.99739808931113, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_a(a4+b4, 4, (3,3), 1)), -68.97044878458135, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_a(a4+b4, 4, (3,3), 2)), -41.22836642162049, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_a(a4+b4, 4, (3,3), 3)), -29.88708752568659, 12)

        self.assertAlmostEqual(lib.fp(fci.addons.des_b(a6+b4, 4, (2,3), 0)), -163.5210711323742, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_b(a6+b4, 4, (2,3), 1)), -187.1999296644511, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_b(a6+b4, 4, (2,3), 2)), 285.3422683187559 , 12)
        self.assertAlmostEqual(lib.fp(fci.addons.des_b(a6+b4, 4, (2,3), 3)), 311.44080890546695, 12)

        self.assertAlmostEqual(lib.fp(fci.addons.cre_a(a6+b4, 4, (2,3), 0)), -39.48915822224921, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_a(a6+b4, 4, (2,3), 1)), 12.45125619610399 , 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_a(a6+b4, 4, (2,3), 2)), 12.016451871939289, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_a(a6+b4, 4, (2,3), 3)), 4.44581041782693  , 12)

        self.assertAlmostEqual(lib.fp(fci.addons.cre_b(a6+b6, 4, (2,2), 0)), -56.76161034968627, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_b(a6+b6, 4, (2,2), 1)), 23.167401126371875, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_b(a6+b6, 4, (2,2), 2)), 30.522245459279716, 12)
        self.assertAlmostEqual(lib.fp(fci.addons.cre_b(a6+b6, 4, (2,2), 3)), -57.04404450083064, 12)


if __name__ == "__main__":
    print("Full Tests for fci.addons")
    unittest.main()
