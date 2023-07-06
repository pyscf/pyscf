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
import tempfile
import numpy
from functools import reduce

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import fci
from pyscf import ci
from pyscf.ci import cisd
from pyscf import ao2mo


class KnownValues(unittest.TestCase):
    def test_contract(self):
        mol = gto.M()
        mol.nelectron = 6
        nocc, nvir = mol.nelectron//2, 4
        nmo = nocc + nvir
        nmo_pair = nmo*(nmo+1)//2
        mf = scf.RHF(mol)
        numpy.random.seed(12)
        mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
        mf.mo_coeff = numpy.random.random((nmo,nmo))
        mf.mo_energy = numpy.arange(0., nmo)
        mf.mo_occ = numpy.zeros(nmo)
        mf.mo_occ[:nocc] = 2
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mol, dm)
        h1 = numpy.random.random((nmo,nmo)) * .1
        h1 = h1 + h1.T
        mf.get_hcore = lambda *args: h1

        myci = ci.CISD(mf)
        eris = myci.ao2mo(mf.mo_coeff)
        eris.ehf = (h1*dm).sum() + (vhf*dm).sum()*.5

        c2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
        c2 = c2 + c2.transpose(1,0,3,2)
        c1 = numpy.random.random(nocc*nvir+1) * .1
        c0, c1 = c1[0], c1[1:].reshape(nocc,nvir)
        civec = myci.amplitudes_to_cisdvec(c0, c1, c2)
        hcivec = ci.cisd.contract(myci, civec, eris)
        self.assertAlmostEqual(lib.finger(hcivec), 2059.5730673341673, 9)
        e2 = ci.cisd.dot(civec, hcivec+eris.ehf*civec, nmo, nocc)
        self.assertAlmostEqual(e2, 7226.7494656749295, 9)

        rdm2 = myci.make_rdm2(civec, nmo, nocc)
        self.assertAlmostEqual(lib.finger(rdm2), 2.0492023431953221, 9)

        def fcicontract(h1, h2, norb, nelec, ci0):
            g2e = fci.direct_spin1.absorb_h1e(h1, h2, norb, nelec, .5)
            ci1 = fci.direct_spin1.contract_2e(g2e, ci0, norb, nelec)
            return ci1
        ci0 = myci.to_fcivec(civec, nmo, mol.nelec)
        self.assertAlmostEqual(abs(civec-myci.from_fcivec(ci0, nmo, nocc*2)).max(), 0, 9)
        h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
        h1e = reduce(numpy.dot, (mf.mo_coeff.T, h1, mf.mo_coeff))
        ci1 = fcicontract(h1e, h2e, nmo, mol.nelec, ci0)
        ci2 = myci.to_fcivec(hcivec, nmo, mol.nelec)
        e1 = numpy.dot(ci1.ravel(), ci0.ravel())
        e2 = ci.cisd.dot(civec, hcivec+eris.ehf*civec, nmo, nocc)
        self.assertAlmostEqual(e1, e2, 9)

        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
                ci.cisd._gamma2_intermediates(myci, civec, nmo, nocc)
        self.assertAlmostEqual(lib.finger(numpy.array(dovov)), 0.02868859991188923, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(dvvvv)),-0.05524957311823144, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(doooo)), 0.01014399192065793, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(doovv)), 0.02761239887072825, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(dovvo)), 0.09971200182238759, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(dovvv)), 0.12777531252787638, 9)
        self.assertAlmostEqual(lib.finger(numpy.array(dooov)), 0.18667669732858014, 9)

    def test_from_fcivec(self):
        mol = gto.M()
        myci = scf.RHF(mol).apply(ci.CISD)
        nelec = (3,3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        myci.nocc = nocc
        myci.nmo = nmo
        civec = numpy.random.random(myci.vector_size())
        ci0 = myci.to_fcivec(civec, nmo, nelec)
        self.assertAlmostEqual(abs(civec-ci.cisd.from_fcivec(ci0, nmo, nelec)).sum(), 0, 9)

        ci0 = myci.to_fcivec(civec, nmo, sum(nelec))
        self.assertAlmostEqual(abs(civec-ci.cisd.from_fcivec(ci0, nmo, sum(nelec))).sum(), 0, 9)

    def test_h4(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.charge = 2
        mol.basis = '3-21g'
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        ecisd = ci.CISD(mf).kernel()[0]
        self.assertAlmostEqual(ecisd, -0.024780739973407784, 6)

        h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
        h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        eci = fci.direct_spin0.kernel(h1e, h2e, mf.mo_coeff.shape[1], mol.nelectron)[0]
        eci = eci + mol.energy_nuc() - mf.e_tot
        self.assertAlmostEqual(ecisd, eci, 9)

    def test_rdm(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = {'H': 'sto-3g',
                     'O': 'sto-3g',}
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.frozen = 1
        eris = myci.ao2mo()
        ecisd, civec = myci.kernel(eris=eris)
        self.assertAlmostEqual(ecisd, -0.048800218694077746, 6)

        nmo = myci.nmo
        nocc = myci.nocc
        strs = fci.cistring.gen_strings4orblist(range(nmo+1), nocc+1)
        mask = (strs & 1) != 0
        sub_strs = strs[mask]
        addrs = fci.cistring.strs2addr(nmo+1, nocc+1, sub_strs)
        na = len(strs)
        ci0 = numpy.zeros((na,na))
        ci0[addrs[:,None],addrs] = myci.to_fcivec(civec, nmo, nocc*2)
        ref1, ref2 = fci.direct_spin1.make_rdm12(ci0, (nmo+1), (nocc+1)*2)
        rdm1 = myci.make_rdm1(civec)
        rdm2 = myci.make_rdm2(civec)
        self.assertTrue(numpy.allclose(rdm2, ref2))
        self.assertAlmostEqual(abs(rdm2-rdm2.transpose(2,3,0,1)).sum(), 0, 9)
        self.assertAlmostEqual(abs(rdm2-rdm2.transpose(1,0,3,2)).sum(), 0, 9)
        dm1 = numpy.einsum('ijkk->ij', rdm2)/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1 - dm1).sum(), 0, 9)

        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo+1)
        e1 = numpy.einsum('ij,ji', h1, rdm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, rdm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, myci.e_tot, 7)

    def test_rdm1(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = '631g'
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.frozen = 1
        myci.kernel()

        nmo = myci.nmo
        nocc = myci.nocc
        d1 = cisd._gamma1_intermediates(myci, myci.ci, nmo, nocc)
        myci.max_memory = 0
        d2 = cisd._gamma2_intermediates(myci, myci.ci, nmo, nocc, True)
        dm2 = cisd.ccsd_rdm._make_rdm2(myci, d1, d2, with_dm1=True, with_frozen=True)
        dm1 = numpy.einsum('ijkk->ij', dm2)/(mol.nelectron-1)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo+1)
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, myci.e_tot, 7)

    def test_trans_rdm(self):
        numpy.random.seed(1)
        myci = ci.CISD(scf.RHF(gto.M()))
        myci.nmo = norb = 4
        myci.nocc = nocc = 2
        nvir = norb - nocc
        c2 = numpy.random.random((nocc,nocc,nvir,nvir))
        c2 = c2 + c2.transpose(1,0,3,2)
        cibra = numpy.hstack((numpy.random.random(1+nocc*nvir), c2.ravel()))
        c2 = numpy.random.random((nocc,nocc,nvir,nvir))
        c2 = c2 + c2.transpose(1,0,3,2)
        ciket = numpy.hstack((numpy.random.random(1+nocc*nvir), c2.ravel()))
        cibra /= ci.cisd.dot(cibra, cibra, norb, nocc)**.5
        ciket /= ci.cisd.dot(ciket, ciket, norb, nocc)**.5
        fcibra = ci.cisd.to_fcivec(cibra, norb, nocc*2)
        fciket = ci.cisd.to_fcivec(ciket, norb, nocc*2)

        fcidm1, fcidm2 = fci.direct_spin1.make_rdm12(fciket, norb, nocc*2)
        cidm1 = ci.cisd.make_rdm1(myci, ciket, norb, nocc)
        cidm2 = ci.cisd.make_rdm2(myci, ciket, norb, nocc)
        self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 12)
        self.assertAlmostEqual(abs(fcidm2-cidm2).max(), 0, 12)

        fcidm1 = fci.direct_spin1.trans_rdm1(fcibra, fciket, norb, nocc*2)
        cidm1  = ci.cisd.trans_rdm1(myci, cibra, ciket, norb, nocc)
        self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 12)

    def test_trans_rdm_with_frozen(self):
        mol = gto.M(atom='''
        O   0.   0.       .0
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''', basis='sto3g')
        mf = scf.RHF(mol).run()

        def check_frozen(frozen):
            myci = ci.CISD(mf)
            myci.frozen = frozen
            myci.nroots = 2
            myci.kernel()
            nocc = myci.nocc
            nmo = myci.nmo
            nfroz = len(frozen)
            cibra = (myci.ci[0] + myci.ci[1]) * numpy.sqrt(.5)
            fcibra = ci.cisd.to_fcivec(cibra, nmo+nfroz, mol.nelectron, myci.frozen)
            fciket = ci.cisd.to_fcivec(myci.ci[1], nmo+nfroz, mol.nelectron, myci.frozen)
            fcidm1 = fci.direct_spin1.trans_rdm1(fcibra, fciket, nmo+nfroz, mol.nelectron)
            cidm1  = myci.trans_rdm1(cibra, myci.ci[1], nmo, nocc)
            self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 12)

        check_frozen([5])
        check_frozen([3])
        check_frozen([1,3])
        check_frozen([2,5])
        #check_frozen([5,6])

    def test_dot(self):
        numpy.random.seed(12)
        nocc, nvir = 3, 5
        civec1 = numpy.random.random((1+nocc*nvir+nocc**2*nvir**2))
        civec2 = numpy.random.random((1+nocc*nvir+nocc**2*nvir**2))
        self.assertAlmostEqual(ci.cisd.dot(civec1, civec2, 8, nocc),
                               64.274937664180186, 13)

    def test_ao_direct(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = 'ccpvdz'
        mol.build()
        mf = scf.RHF(mol).set(conv_tol=1e-14).newton().run()
        myci = ci.CISD(mf)
        myci.max_memory = .1
        myci.nmo = 16
        myci.nocc = 5
        myci.direct = True
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, -0.1319371817220385, 6)

    def test_multi_roots(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.basis = '3-21g'
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.nroots = 3
        myci.run()
        myci.dump_chk()
        self.assertAlmostEqual(myci.e_tot[2], -1.6979890451316759, 6)

    def test_with_df(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.basis = '3-21g'
        mol.build()
        mf = scf.RHF(mol).density_fit('weigend').run(conv_tol=1e-14)
        myci = ci.cisd.RCISD(mf).run()
        self.assertAlmostEqual(myci.e_corr, -0.18730699567992737, 6)

    def test_scanner(self):
        mol = gto.M(atom='''
        O   0.   0.       .0
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''', basis='631g')
        ci_scanner = scf.RHF(mol).apply(ci.CISD).as_scanner()
        ci_scanner.conv_tol = 1e-8
        self.assertAlmostEqual(ci_scanner(mol), -76.114077022009468, 7)

        geom = '''
        O   0.   0.       .1
        H   0.   -0.757   0.587
        H   0.   0.757    0.587'''
        ci_scanner = ci_scanner.as_scanner()
        self.assertAlmostEqual(ci_scanner(geom), -76.104634289269356, 7)

    def test_dump_chk(self):
        mol = gto.M(atom='''
        O   0.   0.       .0
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''', basis='631g')
        mf = scf.RHF(mol).run()
        mf.chkfile = tempfile.NamedTemporaryFile().name
        ci_scanner = ci.CISD(mf).as_scanner()
        ci_scanner(mol)
        ci_scanner.nmo = mf.mo_energy.size
        ci_scanner.nocc = mol.nelectron // 2
        ci_scanner.dump_chk()
        myci = ci.CISD(mf)
        myci.__dict__.update(lib.chkfile.load(ci_scanner.chkfile, 'cisd'))
        self.assertAlmostEqual(abs(ci_scanner.ci-myci.ci).max(), 0, 9)

        ci_scanner.e_corr = -1
        ci_scanner.chkfile = None
        ci_scanner.dump_chk(frozen=2)
        self.assertEqual(lib.chkfile.load(mf.chkfile, 'cisd/e_corr'),
                         myci.e_corr)

    def test_tn_addrs_signs(self):
        norb = 12
        nelec = 6
        addrs, signs = ci.cisd.tn_addrs_signs(norb, nelec, 1)
        addrsr, signsr = t1_strs_ref(norb, nelec)
        self.assertTrue(numpy.all(addrsr == addrs))
        self.assertTrue(numpy.all(signsr == signs))

        addrs, signs = ci.cisd.tn_addrs_signs(norb, nelec, 2)
        addrsr, signsr = t2_strs_ref(norb, nelec)
        self.assertTrue(numpy.all(addrsr == addrs))
        self.assertTrue(numpy.all(signsr == signs))

        addrs, signs = ci.cisd.tn_addrs_signs(norb, nelec, 3)
        addrsr, signsr = t3_strs_ref(norb, nelec)
        self.assertTrue(numpy.all(addrsr == addrs))
        self.assertTrue(numpy.all(signsr == signs))

        addrs, signs = ci.cisd.tn_addrs_signs(norb, nelec, 4)
        addrsr, signsr = t4_strs_ref(norb, nelec)
        self.assertTrue(numpy.all(addrsr == addrs))
        self.assertTrue(numpy.all(signsr == signs))

    def test_overlap(self):
        numpy.random.seed(1)
        nmo = 6
        nocc = 3
        nvir = nmo - nocc
        c2 = numpy.random.rand(nocc,nocc,nvir,nvir)
        cibra = cisd.amplitudes_to_cisdvec(numpy.random.rand(1),
                                           numpy.random.rand(nocc,nvir),
                                           c2+c2.transpose(1,0,3,2))
        c2 = numpy.random.rand(nocc,nocc,nvir,nvir)
        ciket = cisd.amplitudes_to_cisdvec(numpy.random.rand(1),
                                           numpy.random.rand(nocc,nvir),
                                           c2+c2.transpose(1,0,3,2))
        fcibra = cisd.to_fcivec(cibra, nmo, nocc*2)
        fciket = cisd.to_fcivec(ciket, nmo, nocc*2)
        s_mo = numpy.random.random((nmo,nmo))
        s0 = fci.addons.overlap(fcibra, fciket, nmo, nocc*2, s_mo)
        s1 = cisd.overlap(cibra, ciket, nmo, nocc, s_mo)
        self.assertAlmostEqual(s1, s0, 9)

    def test_reset(self):
        mol = gto.M(atom='He')
        mol1 = gto.M(atom='C')
        myci = ci.CISD(scf.UHF(mol).newton())
        myci.reset(mol1)
        self.assertTrue(myci.mol is mol1)
        self.assertTrue(myci._scf.mol is mol1)

    def test_cisdvec_to_amplitudes_overwritten(self):
        mol = gto.M()
        myci = scf.RHF(mol).apply(ci.CISD)
        nelec = (3,3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        myci.nocc = nocc
        myci.nmo = nmo
        vec = numpy.zeros(myci.vector_size())
        vec_orig = vec.copy()
        c0, c1, c2 = myci.cisdvec_to_amplitudes(vec)
        c1[:] = 1
        c2[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    # issue 1362
    def test_cisd_hubbard(self):
        mol = gto.M(verbose=0)
        n, u = 6, 0.0
        mol.nelectron = n
        h1 = numpy.zeros((n,n))
        for i in range(n-1):
            h1[i,i+1] = h1[i+1,i] = -1.0
        eri = numpy.zeros((n,n,n,n))
        for i in range(1):
            eri[i,i,i,i] = u
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: numpy.eye(n)
        mf._eri = ao2mo.restore(8, eri, n)
        mf.kernel()
        myci = ci.CISD(mf)
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, 0, 9)

def t1_strs_ref(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for i in range(nocc):
        for a in range(nocc, norb):
            str1 = hf_str ^ (1 << i) | (1 << a)
            addrs.append(fci.cistring.str2addr(norb, nelec, str1))
            signs.append(fci.cistring.cre_des_sign(a, i, hf_str))
    return numpy.asarray(addrs), numpy.asarray(signs)

def t2_strs_ref(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for i in range(1, nocc):
        for j in range(i):
            for a in range(nocc, norb):
                for b in range(nocc, a):
                    str1 = hf_str ^ (1 << j) | (1 << b)
                    sign = fci.cistring.cre_des_sign(b, j, hf_str)
                    sign*= fci.cistring.cre_des_sign(a, i, str1)
                    str2 = str1 ^ (1 << i) | (1 << a)
                    addrs.append(fci.cistring.str2addr(norb, nocc, str2))
                    signs.append(sign)
    return numpy.asarray(addrs), numpy.asarray(signs)

def t3_strs_ref(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for i in range(2, nocc):
        for j in range(1, i):
            for k in range(j):
                for a in range(nocc, norb):
                    for b in range(nocc, a):
                        for c in range(nocc, b):
                            str1 = hf_str ^ (1 << k) | (1 << c)
                            str2 = str1 ^ (1 << j) | (1 << b)
                            sign = fci.cistring.cre_des_sign(c, k, hf_str)
                            sign*= fci.cistring.cre_des_sign(b, j, str1)
                            sign*= fci.cistring.cre_des_sign(a, i, str2)
                            str3 = str2 ^ (1 << i) | (1 << a)
                            addrs.append(fci.cistring.str2addr(norb, nocc, str3))
                            signs.append(sign)
    return numpy.asarray(addrs), numpy.asarray(signs)

def t4_strs_ref(norb, nelec):
    nocc = nelec
    hf_str = int('1'*nocc, 2)
    addrs = []
    signs = []
    for i in range(3, nocc):
        for j in range(2, i):
            for k in range(1, j):
                for l in range(k):
                    for a in range(nocc, norb):
                        for b in range(nocc, a):
                            for c in range(nocc, b):
                                for d in range(nocc, c):
                                    str1 = hf_str ^ (1 << l) | (1 << d)
                                    str2 = str1 ^ (1 << k) | (1 << c)
                                    str3 = str2 ^ (1 << j) | (1 << b)
                                    sign = fci.cistring.cre_des_sign(d, l, hf_str)
                                    sign*= fci.cistring.cre_des_sign(c, k, str1)
                                    sign*= fci.cistring.cre_des_sign(b, j, str2)
                                    sign*= fci.cistring.cre_des_sign(a, i, str3)
                                    str4 = str3 ^ (1 << i) | (1 << a)
                                    addrs.append(fci.cistring.str2addr(norb, nocc, str4))
                                    signs.append(sign)
    return numpy.asarray(addrs), numpy.asarray(signs)


if __name__ == "__main__":
    print("Full Tests for CISD")
    unittest.main()
