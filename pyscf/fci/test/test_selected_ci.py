#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import symm
from pyscf import fci
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci import selected_ci
from pyscf.fci import selected_ci_symm
from pyscf.fci import spin_op
from pyscf.fci import selected_ci_spin0
from pyscf.fci import selected_ci_spin0_symm
from pyscf.fci import selected_ci_slow

def setUpModule():
    global ci_strs, ci_coeff, civec_strs, eri, h1, spin0_ci_strs, spin0_ci_coeff
    global norb, nelec
    norb = 6
    nelec = 6
    na = cistring.num_strings(norb, nelec//2)
    ci_strs = [[0b111, 0b1011, 0b10101], [0b111, 0b1011, 0b1101]]
    numpy.random.seed(12)
    ci_coeff = (numpy.random.random((len(ci_strs[0]),len(ci_strs[1])))-.2)**3
    civec_strs = selected_ci._as_SCIvector(ci_coeff, ci_strs)
    nn = norb*(norb+1)//2
    eri = (numpy.random.random(nn*(nn+1)//2)-.2)**3
    h1 = numpy.random.random((norb,norb))
    h1 = h1 + h1.T

    spin0_ci_strs = [[0b111, 0b1011, 0b10101], [0b111, 0b1011, 0b10101]]
    spin0_ci_coeff = ci_coeff + ci_coeff.T

def tearDownModule():
    global ci_strs, ci_coeff, civec_strs, eri, h1, spin0_ci_strs, spin0_ci_coeff
    del ci_strs, ci_coeff, civec_strs, eri, h1, spin0_ci_strs, spin0_ci_coeff

class KnownValues(unittest.TestCase):
    def test_select_strs(self):
        myci = selected_ci.SCI()
        myci.select_cutoff = 1e-3
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .8
        strs = strs[mask]
        nn = norb*(norb+1)//2
        eri = (numpy.random.random(nn*(nn+1)//2)-.2)**3
        eri[eri<.1] *= 3e-3
        eri = ao2mo.restore(1, eri, norb)
        eri_pq_max = abs(eri.reshape(norb**2,-1)).max(axis=1).reshape(norb,norb)
        civec_max = numpy.random.random(len(strs))
        strs_add0 = select_strs(myci, eri, eri_pq_max, civec_max, strs, norb, nelec)
        strs_add1 = selected_ci.select_strs(myci, eri, eri_pq_max, civec_max,
                                            strs, norb, nelec)
        self.assertTrue(numpy.all(strs_add0 == strs_add1))

    def test_select_strs1(self):
        myci = selected_ci.SCI()
        myci.select_cutoff = .1
        myci.ci_coeff_cutoff = .01

        eri1 = ao2mo.restore(1, eri, norb)
        eri_pq_max = abs(eri1.reshape(norb**2,-1)).max(axis=1).reshape(norb,norb)
        civec_a_max = abs(ci_coeff).max(axis=1)
        civec_b_max = abs(ci_coeff).max(axis=0)
        strs_add0 = select_strs(myci, eri1, eri_pq_max, civec_a_max, ci_strs[0],
                                norb, nelec//2)
        strs_add1 = selected_ci.select_strs(myci, eri1, eri_pq_max, civec_a_max,
                                            ci_strs[0], norb, nelec//2)
        self.assertTrue(numpy.all(strs_add0 == strs_add1))

        strs_add0 = select_strs(myci, eri1, eri_pq_max, civec_b_max, ci_strs[1],
                                norb, nelec//2)
        strs_add1 = selected_ci.select_strs(myci, eri1, eri_pq_max, civec_b_max,
                                            ci_strs[1], norb, nelec//2)
        self.assertTrue(numpy.all(strs_add0 == strs_add1))

    def test_enlarge_space(self):
        myci = selected_ci.SCI()
        myci.select_cutoff = .1
        myci.ci_coeff_cutoff = .01
        cic = selected_ci.enlarge_space(myci, civec_strs, eri, norb, nelec)
        cis = cic._strs
        self.assertEqual((len(cis[0]), len(cis[1])), (17,18))  # 16,14
        self.assertEqual(list(cis[0]), [7,11,13,14,19,21,22,25,26,28,35,37,41,42,49,52,56])
        self.assertEqual(list(cis[1]), [7,11,13,14,19,21,22,25,28,35,37,38,41,44,49,50,52,56])
        self.assertAlmostEqual(abs(cic[[0,1,5]][:,[0,1,2]] - ci_coeff).sum(), 0, 12)

    def test_contract_2e(self):
        myci = selected_ci.SCI()
        ci0 = selected_ci.to_fci(civec_strs, norb, nelec)
        e1 = numpy.dot(civec_strs.ravel(), myci.contract_2e(eri, civec_strs, norb, nelec).ravel())
        eref = numpy.dot(ci0.ravel(), direct_spin1.contract_2e(eri, ci0, norb, nelec).ravel())
        self.assertAlmostEqual(e1, eref, 9)

        e2 = numpy.dot(civec_strs.ravel(), myci.contract_2e(eri, ci_coeff, norb, nelec).ravel())
        self.assertAlmostEqual(e2, eref, 9)

        dm1 = myci.make_rdm1(civec_strs, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm1), 0.70181046385686563, 9)
        dm1 = myci.trans_rdm1(civec_strs, civec_strs, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm1), 0.70181046385686563, 9)
        dm2 = myci.make_rdm2(civec_strs, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm2), -3.8397469683353962, 9)

    def test_contract_2e_1(self):
        myci = selected_ci.SCI()
        nelec = (4,3)
        strsa = cistring.make_strings(range(norb), nelec[0])
        strsb = cistring.make_strings(range(norb), nelec[1])
        ci0 = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), (strsa,strsb))
        h2 = ao2mo.restore(1, eri, norb)
        c1 = myci.contract_2e(h2, ci0, norb, nelec)
        c2 = direct_spin1.contract_2e(h2, ci0, norb, nelec)
        self.assertAlmostEqual(float(abs(c1-c2).sum()), 0, 9)
        dm1_1 = myci.make_rdm1(c1, norb, nelec)
        dm1_2 = direct_spin1.make_rdm1(c2, norb, nelec)
        self.assertAlmostEqual(abs(dm1_1 - dm1_2).sum(), 0, 9)
        dm2_1 = myci.make_rdm12(c1, norb, nelec)[1]
        dm2_2 = direct_spin1.make_rdm12(c2, norb, nelec)[1]
        self.assertAlmostEqual(abs(dm2_1 - dm2_2).sum(), 0, 9)

    def test_spin0_contract(self):
        myci = selected_ci_spin0.SCI()
        civec_strs = selected_ci._as_SCIvector(spin0_ci_coeff, spin0_ci_strs)
        ci0 = selected_ci.to_fci(civec_strs, norb, nelec)
        e1 = numpy.dot(civec_strs.ravel(), myci.contract_2e(eri, civec_strs, norb, nelec).ravel())
        eref = numpy.dot(ci0.ravel(), direct_spin1.contract_2e(eri, ci0, norb, nelec).ravel())
        self.assertAlmostEqual(e1, eref, 9)

        e2 = numpy.dot(civec_strs.ravel(), myci.contract_2e(eri, spin0_ci_coeff, norb, nelec).ravel())
        self.assertAlmostEqual(e2, eref, 9)

    def test_kernel(self):
        myci = selected_ci.SCI()
        e1, c1 = selected_ci.kernel(h1, eri, norb, nelec, nroots=2)
        e1 = e1[0]
        c1 = c1[0]
        e2, c2 = direct_spin1.kernel(h1, eri, norb, nelec)
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(abs(numpy.dot(c1.ravel(), c2.ravel())), 1, 9)
        dm1_1 = myci.make_rdm1(c1, norb, nelec)
        dm1_2 = direct_spin1.make_rdm1(c2, norb, nelec)
        self.assertAlmostEqual(abs(dm1_1 - dm1_2).sum(), 0, 4)
        dm2_1 = myci.make_rdm2(c1, norb, nelec)
        dm2_2 = direct_spin1.make_rdm12(c2, norb, nelec)[1]
        self.assertAlmostEqual(abs(dm2_1 - dm2_2).sum(), 0, 2)

        e1, c1 = selected_ci_spin0.kernel(h1, eri, norb, nelec)
        self.assertAlmostEqual(e1, e2, 9)
        self.assertAlmostEqual(abs(numpy.dot(c1.ravel(), c2.ravel())), 1, 9)

        e1, c1 = selected_ci_slow.kernel(h1, eri, norb, nelec)
        self.assertAlmostEqual(e1, e2, 9)

    def test_kernel_small_system(self):
        e1, c1 = selected_ci.kernel(h1[:2,:2], eri[:6], 2, (1,1), nroots=2)
        e2, c2 = direct_spin1.kernel(h1[:2,:2], eri[:6], 2, (1,1), nroots=2)
        self.assertAlmostEqual(abs(numpy.array(e1) - e2).max(), 0, 9)

        self.assertRaises(RuntimeError, selected_ci.kernel, h1[:2,:2], eri[:6], 2, (1,1), nroots=6)

    def test_hdiag(self):
        hdiag = selected_ci.make_hdiag(h1, eri, ci_strs, norb, nelec)
        self.assertAlmostEqual(lib.fp(hdiag), 8.2760894885437377, 9)

    def test_h8(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = None
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   ,-1.   )],
            ['H', ( 0.,-0.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0., 1.    , 1.   )],
            ['H', ( 1., 2.    , 3.   )],
            ['H', ( 1., 2.    , 4.   )],
        ]
        mol.basis = 'sto-3g'
        mol.build()

        m = scf.RHF(mol).run()
        norb = m.mo_coeff.shape[1]
        nelec = mol.nelectron
        h1e = reduce(numpy.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
        eri = ao2mo.kernel(m._eri, m.mo_coeff, compact=False)
        eri = eri.reshape(norb,norb,norb,norb)

        myci = selected_ci.SCI()
        myci.select_cutoff = 1e-3
        myci.ci_coeff_cutoff = 1e-3
        myci.dump_flags()
        e1, c1 = myci.kernel(h1e, eri, norb, nelec)
        self.assertAlmostEqual(e1, -11.894613845925514, 8)

        e, c = myci.kernel_fixed_space(h1e, eri, norb, nelec, c1._strs)
        self.assertAlmostEqual(e, -11.894613845925514, 8)

        res = myci.large_ci(c1, norb, nelec, .25)
        self.assertEqual([x[1] for x in res], ['0b1111', '0b1111', '0b10111', '0b10111'])
        self.assertEqual([x[2] for x in res], ['0b1111', '0b10111', '0b1111', '0b10111'])

        res = myci.large_ci(c1, norb, nelec, .25, return_strs=False)
        refa = numpy.array(((0,1,2,3), (0,1,2,3), (0,1,2,4), (0,1,2,4)))
        refb = numpy.array(((0,1,2,3), (0,1,2,4), (0,1,2,3), (0,1,2,4)))
        self.assertTrue(numpy.all([x[1] for x in res] == refa))
        self.assertTrue(numpy.all([x[2] for x in res] == refb))

        self.assertAlmostEqual(myci.spin_square(c1, norb, nelec)[0], 0, 2)

    def test_cre_des_linkstr(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .5
        strs = strs[mask]
        cd_index0 = cre_des_linkstr(strs, norb, nelec)
        cd_index1 = selected_ci.cre_des_linkstr(strs, norb, nelec)
        self.assertTrue(numpy.all(cd_index0 == cd_index1))
        cd_index0 = cre_des_linkstr_tril(strs, norb, nelec)
        cd_index1 = selected_ci.cre_des_linkstr_tril(strs, norb, nelec)
        cd_index1[:,:,1] = 0
        self.assertTrue(numpy.all(cd_index0 == cd_index1))

    def test_des_des_linkstr(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .4
        strs = strs[mask]
        dd_index0 = des_des_linkstr(strs, norb, nelec)
        dd_index1 = selected_ci.des_des_linkstr(strs, norb, nelec)
        self.assertTrue(numpy.all(dd_index0 == dd_index1))
        dd_index0 = des_des_linkstr_tril(strs, norb, nelec)
        dd_index1 = selected_ci.des_des_linkstr_tril(strs, norb, nelec)
        dd_index1[:,:,1] = 0
        self.assertTrue(numpy.all(dd_index0 == dd_index1))

    def test_des_linkstr(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .4
        strs = strs[mask]
        d_index0 = gen_des_linkstr(strs, norb, nelec)
        d_index1 = selected_ci.gen_des_linkstr(strs, norb, nelec)
        d_index1[:,:,0] = 0
        self.assertTrue(numpy.all(d_index0 == d_index1))

    def test_des_linkstr1(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .6
        strs = strs[mask]
        c_index0 = gen_cre_linkstr(strs, norb, nelec)
        c_index1 = selected_ci.gen_cre_linkstr(strs, norb, nelec)
        c_index1[:,:,1] = 0
        self.assertTrue(numpy.all(c_index0 == c_index1))

    def test_from_to_fci(self):
        ci0 = selected_ci.to_fci(civec_strs, norb, nelec)
        ci1 = selected_ci.from_fci(ci0, ci_strs, norb, nelec)
        self.assertAlmostEqual(abs(ci1-ci_coeff).sum(), 0, 12)

    def test_rdm(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .6
        strsa = strs[mask]
        mask = numpy.random.random(len(strs)) > .7
        strsb = strs[mask]
        ci_strs = (strsa, strsb)
        ci_coeff = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        ci0 = selected_ci.to_fci(ci_coeff, norb, (nelec,nelec))
        dm1ref, dm2ref = direct_spin1.make_rdm12s(ci0, norb, (nelec,nelec))
        dm1 = selected_ci.make_rdm1s(ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm1[0]-dm1ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm1[1]-dm1ref[1]).sum(), 0, 9)
        myci = selected_ci.SCI()
        dm2 = myci.make_rdm2s(ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm2[0]-dm2ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2[1]-dm2ref[1]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2[2]-dm2ref[2]).sum(), 0, 9)

        myci = selected_ci.SCI()
        ci1_coeff = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        ci1 = selected_ci.to_fci(ci1_coeff, norb, (nelec,nelec))
        dm1ref, dm2ref = direct_spin1.trans_rdm12s(ci1, ci0, norb, (nelec,nelec))
        dm1 = myci.trans_rdm1s(ci1_coeff, ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm1[0]-dm1ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm1[1]-dm1ref[1]).sum(), 0, 9)
        #dm2 = selected_ci.trans_rdm1s(ci1_coeff, ci_coeff, norb, (nelec,nelec))
        #self.assertAlmostEqual(abs(dm2[0]-dm2ref[0]).sum(), 0, 9)
        #self.assertAlmostEqual(abs(dm2[1]-dm2ref[1]).sum(), 0, 9)
        #self.assertAlmostEqual(abs(dm2[2]-dm2ref[2]).sum(), 0, 9)

    def test_rdm_2e(self):
        norb, nelec = 10, 1
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .6
        strsa = strs[mask]
        mask = numpy.random.random(len(strs)) > .7
        strsb = strs[mask]
        ci_strs = (strsa, strsb)
        ci_coeff = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        ci0 = selected_ci.to_fci(ci_coeff, norb, (nelec,nelec))
        dm1ref, dm2ref = direct_spin1.make_rdm12s(ci0, norb, (nelec,nelec))
        dm1 = selected_ci.make_rdm1s(ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm1[0]-dm1ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm1[1]-dm1ref[1]).sum(), 0, 9)
        dm2 = selected_ci.make_rdm2s(ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm2[0]-dm2ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2[1]-dm2ref[1]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm2[2]-dm2ref[2]).sum(), 0, 9)

        ci1_coeff = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        ci1 = selected_ci.to_fci(ci1_coeff, norb, (nelec,nelec))
        dm1ref, dm2ref = direct_spin1.trans_rdm12s(ci1, ci0, norb, (nelec,nelec))
        dm1 = selected_ci.trans_rdm1s(ci1_coeff, ci_coeff, norb, (nelec,nelec))
        self.assertAlmostEqual(abs(dm1[0]-dm1ref[0]).sum(), 0, 9)
        self.assertAlmostEqual(abs(dm1[1]-dm1ref[1]).sum(), 0, 9)
        #dm2 = selected_ci.trans_rdm1s(ci1_coeff, ci_coeff, norb, (nelec,nelec))
        #self.assertAlmostEqual(abs(dm2[0]-dm2ref[0]).sum(), 0, 9)
        #self.assertAlmostEqual(abs(dm2[1]-dm2ref[1]).sum(), 0, 9)
        #self.assertAlmostEqual(abs(dm2[2]-dm2ref[2]).sum(), 0, 9)

    def test_spin_square(self):
        norb, nelec = 10, 4
        strs = cistring.make_strings(range(norb), nelec)
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .6
        strsa = strs[mask]
        mask = numpy.random.random(len(strs)) > .7
        strsb = strs[mask]
        ci_strs = (strsa, strsb)
        ci_coeff = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        ci0 = selected_ci.to_fci(ci_coeff, norb, (nelec,nelec))
        ss0 = selected_ci.spin_square(ci_coeff, norb, (nelec,nelec))
        ss1 = spin_op.spin_square0(ci0, norb, (nelec,nelec))
        self.assertAlmostEqual(ss0[0], ss1[0], 9)

    def test_contract_2e_symm(self):
        norb, nelec = 7, (4,4)
        strs = cistring.make_strings(range(norb), nelec[0])
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .3
        strsa = strs[mask]
        mask = numpy.random.random(len(strs)) > .2
        strsb = strs[mask]
        ci_strs = (strsa, strsb)
        civec_strs = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        orbsym = (numpy.random.random(norb) * 4).astype(int)
        nn = norb*(norb+1)//2
        eri = ao2mo.restore(1, (numpy.random.random(nn*(nn+1)//2)-.2)**3, norb)
        oosym = orbsym[:,None] ^ orbsym
        oosym = oosym.reshape(-1,1) ^ oosym.ravel()
        eri[oosym.reshape([norb]*4)!=0] = 0
        ci0 = fci.selected_ci.to_fci(civec_strs, norb, nelec)
        ci0 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym)
        civec_strs = fci.selected_ci.from_fci(ci0, civec_strs._strs, norb, nelec)
        e1 = numpy.dot(civec_strs.ravel(), selected_ci_symm.contract_2e(eri, civec_strs, norb, nelec, orbsym=orbsym).ravel())
        e2 = numpy.dot(ci0.ravel(), direct_spin1_symm.contract_2e(eri, ci0, norb, nelec, orbsym=orbsym).ravel())
        self.assertAlmostEqual(e1, e2, 9)

    def test_spin0_contract_2e_symm(self):
        norb, nelec = 7, (4,4)
        strs = cistring.make_strings(range(norb), nelec[0])
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .3
        strsa = strs[mask]
        ci_strs = (strsa, strsa)
        na = len(strsa)
        ci_coeff = numpy.random.random((na,na))
        ci_coeff = ci_coeff + ci_coeff.T
        civec_strs = selected_ci._as_SCIvector(ci_coeff, ci_strs)
        orbsym = (numpy.random.random(norb) * 4).astype(int)
        nn = norb*(norb+1)//2
        eri = ao2mo.restore(1, (numpy.random.random(nn*(nn+1)//2)-.2)**3, norb)
        oosym = orbsym[:,None] ^ orbsym
        oosym = oosym.reshape(-1,1) ^ oosym.ravel()
        eri[oosym.reshape([norb]*4)!=0] = 0

        ci0 = fci.selected_ci.to_fci(civec_strs, norb, nelec)
        ci0 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym)
        civec_strs = fci.selected_ci.from_fci(ci0, civec_strs._strs, norb, nelec)

        myci = selected_ci_spin0_symm.SCI()
        e1 = numpy.dot(civec_strs.ravel(), myci.contract_2e(eri, civec_strs, norb, nelec, orbsym=orbsym).ravel())
        e2 = numpy.dot(ci0.ravel(), direct_spin1_symm.contract_2e(eri, ci0, norb, nelec, orbsym=orbsym).ravel())
        self.assertAlmostEqual(e1, e2, 9)

    def test_kernel_symm(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = None
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = 'sto-3g'
        mol.symmetry = 1
        mol.build()
        m = scf.RHF(mol).run()

        norb = m.mo_coeff.shape[1]
        nelec = mol.nelectron - 2
        h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
        eri = ao2mo.incore.full(m._eri, m.mo_coeff)
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)

        myci = selected_ci_symm.SCI().set(orbsym=orbsym, select_cutoff=.5e-3)
        e1, c1 = myci.kernel(h1e, eri, norb, nelec)
        myci = direct_spin1_symm.FCISolver().set(orbsym=orbsym)
        e2, c2 = myci.kernel(h1e, eri, norb, nelec)

        self.assertAlmostEqual(e1, e2, 6)
        c2_cut = selected_ci.from_fci(c2, c1._strs, norb, nelec)
        self.assertAlmostEqual(abs(numpy.dot(c1.ravel(), c2_cut.ravel())), 1, 6)

        myci = selected_ci_symm.SCI().set(orbsym=orbsym, select_cutoff=1e-5)
        e1, c1 = myci.kernel(h1e, eri, norb, nelec, ci0=c2_cut, wfnsym=0)
        self.assertAlmostEqual(e1, e2, 9)
        c2_cut = selected_ci.from_fci(c2, c1._strs, norb, nelec)
        self.assertAlmostEqual(abs(numpy.dot(c1.ravel(), c2_cut.ravel())), 1, 9)

    def test_gen_linkstr(self):
        sol = selected_ci.SCI()
        cd_index1a, dd_index1a, cd_index1b, dd_index1b = \
                sol.gen_linkstr(norb, 6, tril=True, ci_strs=ci_strs)
        cd_index1a[:,:,1] = 0
        cd_index1b[:,:,1] = 0
        cd_index2a, dd_index2a, cd_index2b, dd_index2b = \
                sol.gen_linkstr(norb, (3,3), tril=False, ci_strs=ci_strs)
        self.assertAlmostEqual(abs(cd_index1a - fci.cistring.reform_linkstr_index(cd_index2a)).max(), 0, 12)
        self.assertAlmostEqual(abs(cd_index1b - fci.cistring.reform_linkstr_index(cd_index2b)).max(), 0, 12)

    def test_contract_2e_vs_slow_version(self):
        myci = selected_ci.SCI()
        ci1 = myci.contract_2e(eri, civec_strs, norb, nelec)
        ci_and_str = (civec_strs, ci_strs)
        ciref = selected_ci_slow.contract_2e(eri, ci_and_str, norb, nelec)
        self.assertAlmostEqual(abs(ci1-ciref).max(), 0, 12)

    def test_rdm_vs_slow_version(self):
        ci_and_str = (civec_strs, ci_strs)
        dm1 = selected_ci_slow.make_rdm1(ci_and_str, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm1), 0.70181046385686563, 9)
        dm2 = selected_ci_slow.make_rdm2(ci_and_str, norb, nelec)
        self.assertAlmostEqual(lib.fp(dm2), -3.8397469683353962, 9)

    def test_guess_wfnsym(self):
        norb, nelec = 7, (4,4)
        strs = cistring.make_strings(range(norb), nelec[0])
        numpy.random.seed(11)
        mask = numpy.random.random(len(strs)) > .3
        strsa = strs[mask]
        mask = numpy.random.random(len(strs)) > .2
        strsb = strs[mask]
        ci_strs = (strsa, strsb)
        ci0 = selected_ci._as_SCIvector(numpy.random.random((len(strsa),len(strsb))), ci_strs)
        fake_mol = gto.M()
        fake_mol.groupname = 'C2v'
        cis = selected_ci_symm.SelectedCI(fake_mol)
        cis.orbsym = orbsym = (numpy.random.random(norb) * 4).astype(int)

        self.assertEqual(cis.guess_wfnsym(norb, nelec), 0)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0), 3)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0, wfnsym=0), 0)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0, wfnsym='B2'), 3)
        ci0[:] = 0
        self.assertRaises(RuntimeError, cis.guess_wfnsym, norb, nelec, ci0, wfnsym=1)

    # The edge case CAS(2, 2). issue #1808
    def test_cas_2_2(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1.')
        mc = mol.RHF().run().CASCI(2, 2)
        mc.fcisolver = fci.SCI(mol)
        mc.run()
        self.assertAlmostEqual(mc.e_tot, -1.1011503302326, 9)
        self.assertAlmostEqual(abs(mc.ci[0,0]), .984513596, 5)
        self.assertAlmostEqual(abs(mc.ci[1,1]), .175308242, 5)


def gen_des_linkstr(strs, norb, nelec):
    '''Given intermediates, the link table to generate input strs
    '''
    if nelec < 1:
        return None

    inter = []
    for str0 in strs:
        occ = [i for i in range(norb) if str0 & (1<<i)]
        for i in occ:
            inter.append(str0 ^ (1<<i))
    inter = sorted(set(inter))
    addrs = dict(zip(strs, range(len(strs))))

    nvir = norb - nelec + 1
    link_index = numpy.zeros((len(inter),nvir,4), dtype=numpy.int32)
    for i1, str1 in enumerate(inter):
        vir = [i for i in range(norb) if not str1 & (1<<i)]
        k = 0
        for i in vir:
            str0 = str1 | (1<<i)
            if str0 in addrs:
                sign = cistring.cre_sign(i, str1)
                link_index[i1,k] = (0, i, addrs[str0], sign)
                k += 1
    return link_index

def gen_cre_linkstr(strs, norb, nelec):
    '''Given intermediates, the link table to generate input strs
    '''
    if nelec == norb:
        return None
    inter = []
    for str0 in strs:
        vir = [i for i in range(norb) if not str0 & (1<<i)]
        for i in vir:
            inter.append(str0 | (1<<i))
    inter = sorted(set(inter))
    addrs = dict(zip(strs, range(len(strs))))

    link_index = numpy.zeros((len(inter),nelec+1,4), dtype=numpy.int32)
    for i1, str1 in enumerate(inter):
        occ = [i for i in range(norb) if str1 & (1<<i)]
        k = 0
        for i in occ:
            str0 = str1 ^ (1<<i)
            if str0 in addrs:
                sign = cistring.des_sign(i, str1)
                link_index[i1,k] = (i, 0, addrs[str0], sign)
                k += 1
    return link_index

def cre_des_linkstr(strs, norb, nelec):
    addrs = dict(zip(strs, range(len(strs))))
    nvir = norb - nelec
    link_index = numpy.zeros((len(addrs),nelec+nelec*nvir,4), dtype=numpy.int32)
    for i0, str1 in enumerate(strs):
        occ = []
        vir = []
        for i in range(norb):
            if str1 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        k = 0
        for i in occ:
            link_index[i0,k] = (i, i, i0, 1)
            k += 1
        for a in vir:
            for i in occ:
                str0 = str1 ^ (1<<i) | (1<<a)
                if str0 in addrs:
                    # [cre, des, targetddress, parity]
                    link_index[i0,k] = (a, i, addrs[str0], cistring.cre_des_sign(a, i, str1))
                    k += 1
    return link_index

def cre_des_linkstr_tril(strs, norb, nelec):
    addrs = dict(zip(strs, range(len(strs))))
    nvir = norb - nelec
    link_index = numpy.zeros((len(addrs),nelec+nelec*nvir,4), dtype=numpy.int32)
    for i0, str1 in enumerate(strs):
        occ = []
        vir = []
        for i in range(norb):
            if str1 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        k = 0
        for i in occ:
            link_index[i0,k] = (i*(i+1)//2+i, 0, i0, 1)
            k += 1
        for a in vir:
            for i in occ:
                str0 = str1 ^ (1<<i) | (1<<a)
                if str0 in addrs:
                    if a > i:
                        ai = a*(a+1)//2 + i
                    else:
                        ai = i*(i+1)//2 + a
                    # [cre, des, targetddress, parity]
                    link_index[i0,k] = (ai, 0, addrs[str0], cistring.cre_des_sign(a, i, str1))
                    k += 1
    return link_index

def des_des_linkstr(strs, norb, nelec):
    '''Given intermediates, the link table to generate input strs
    '''
    inter = []
    for str0 in strs:
        occ = [i for i in range(norb) if str0 & (1<<i)]
        for i1, i in enumerate(occ):
            for j in occ[:i1]:
                inter.append(str0 ^ (1<<i) ^ (1<<j))
    inter = sorted(set(inter))
    addrs = dict(zip(strs, range(len(strs))))

    nvir = norb - nelec + 2
    link_index = numpy.zeros((len(inter),nvir*nvir,4), dtype=numpy.int32)
    for i0, str1 in enumerate(inter):
        vir = [i for i in range(norb) if not str1 & (1<<i)]
        k = 0
        for i1, i in enumerate(vir):
            for j in vir[:i1]:
                str0 = str1 | (1<<i) | (1<<j)
                if str0 in addrs:
                    # from intermediate str1, create i, create j -> str0
                    # (str1 = des_i des_j str0)
                    # [cre_j, cre_i, targetddress, parity]
                    sign = cistring.cre_sign(i, str1)
                    sign*= cistring.cre_sign(j, str1|(1<<i))
                    link_index[i0,k] = (i, j, addrs[str0], sign)
                    link_index[i0,k+1] = (j, i, addrs[str0],-sign)
                    k += 2
    return link_index

def des_des_linkstr_tril(strs, norb, nelec):
    '''Given intermediates, the link table to generate input strs
    '''
    inter = []
    for str0 in strs:
        occ = [i for i in range(norb) if str0 & (1<<i)]
        for i1, i in enumerate(occ):
            for j in occ[:i1]:
                inter.append(str0 ^ (1<<i) ^ (1<<j))
    inter = sorted(set(inter))
    addrs = dict(zip(strs, range(len(strs))))

    nvir = norb - nelec + 2
    link_index = numpy.zeros((len(inter),nvir*nvir,4), dtype=numpy.int32)
    for i0, str1 in enumerate(inter):
        vir = [i for i in range(norb) if not str1 & (1<<i)]
        k = 0
        for i1, i in enumerate(vir):
            for j in vir[:i1]:
                str0 = str1 | (1<<i) | (1<<j)
                if str0 in addrs:
                    # from intermediate str1(i0), create i, create j -> str0
                    # (str1 = des_i des_j str0)
                    # [cre_j, cre_i, targetddress, parity]
                    sign = cistring.cre_sign(i, str1)
                    sign*= cistring.cre_sign(j, str1|(1<<i))
                    link_index[i0,k] = (i*(i-1)//2+j, 0, addrs[str0], sign)
                    k += 1
    return link_index

def select_strs(myci, eri, eri_pq_max, civec_max, strs, norb, nelec):
    strs_add = []
    for ia, str0 in enumerate(strs):
        occ = []
        vir = []
        for i in range(norb):
            if str0 & (1<<i):
                occ.append(i)
            else:
                vir.append(i)
        ca = civec_max[ia]
        for i1, i in enumerate(occ):
            for a1, a in enumerate(vir):
                if eri_pq_max[a,i]*ca > myci.select_cutoff:
                    str1 = str0 ^ (1<<i) | (1<<a)
                    strs_add.append(str1)

                    if i < nelec and a >= nelec:
                        for j in occ[:i1]:
                            for b in vir[a1+1:]:
                                if abs(eri[a,i,b,j])*ca > myci.select_cutoff:
                                    strs_add.append(str1 ^ (1<<j) | (1<<b))
    strs_add = sorted(set(strs_add) - set(strs))
    return numpy.asarray(strs_add, dtype=numpy.int64)

if __name__ == "__main__":
    print("Full Tests for selected_ci")
    unittest.main()
