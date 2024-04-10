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
import scipy.linalg
from functools import reduce

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import fci
from pyscf import ci
from pyscf.ci import ucisd
from pyscf import ao2mo


class KnownValues(unittest.TestCase):
    def test_from_fcivec(self):
        myci = scf.UHF(gto.M()).apply(ci.CISD)
        nocca, noccb = nelec = (3,2)
        nvira, nvirb = 5, 6
        myci.nocc = nocc = nocca, noccb
        nmo = 8
        myci.nmo = (nmo,nmo)
        numpy.random.seed(12)
        civec = numpy.random.random(myci.vector_size())
        ci0 = ucisd.to_fcivec(civec, nmo, nelec)
        self.assertAlmostEqual(abs(civec-ucisd.from_fcivec(ci0, nmo, nelec)).max(), 0, 9)

        nocc = 3
        nvir = 5
        nmo = nocc + nvir
        c1a = numpy.random.random((nocc,nvir))
        c1b = numpy.random.random((nocc,nvir))
        c2aa = numpy.random.random((nocc,nocc,nvir,nvir))
        c2bb = numpy.random.random((nocc,nocc,nvir,nvir))
        c2ab = numpy.random.random((nocc,nocc,nvir,nvir))
        c1 = (c1a, c1b)
        c2 = (c2aa, c2ab, c2bb)
        cisdvec = ucisd.amplitudes_to_cisdvec(1., c1, c2)
        fcivec = ucisd.to_fcivec(cisdvec, nmo, nocc*2)
        cisdvec1 = ucisd.from_fcivec(fcivec, nmo, nocc*2)
        self.assertAlmostEqual(abs(cisdvec-cisdvec1).max(), 0, 12)
        ci1 = ucisd.to_fcivec(cisdvec1, nmo, (nocc,nocc))
        self.assertAlmostEqual(abs(fcivec-ci1).max(), 0, 12)

    def test_h4(self):
        mol = gto.Mole()
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.charge = 2
        mol.spin = 2
        mol.basis = '3-21g'
        mol.build()
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -0.50569591904536926, 8)

        mf = scf.UHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        ecisd = myci.kernel()[0]
        self.assertAlmostEqual(ecisd, -0.03598992307519909, 8)
        self.assertAlmostEqual(myci.e_tot, -0.50569591904536926, 8)

        eris = myci.ao2mo()
        ecisd = myci.kernel(eris=eris)[0]
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]])
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        efci, fcivec = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                             h1a.shape[0], mol.nelec)
        self.assertAlmostEqual(mf.e_tot+ecisd, efci+mol.energy_nuc(), 9)
        dm1ref, dm2ref = fci.direct_uhf.make_rdm12s(fcivec, h1a.shape[0], mol.nelec)
        rdm1 = myci.make_rdm1(myci.ci, myci.get_nmo(), myci.get_nocc())
        rdm2 = myci.make_rdm2(myci.ci, myci.get_nmo(), myci.get_nocc())
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[0]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[1]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[0]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[1]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[2]).max(), 0, 4)

    def test_h4_a(self):
        '''Compare to FCI'''
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.charge = -2
        mol.spin = 2
        mol.basis = '3-21g'
        mol.build()
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        ehf0 = mf.e_tot - mol.energy_nuc()
        myci = ci.CISD(mf)
        numpy.random.seed(10)
        nao = mol.nao_nr()
        mo = numpy.random.random((2,nao,nao))

        eris = myci.ao2mo(mo)
        self.assertAlmostEqual(lib.fp(myci.make_diagonal(eris)),
                               -838.45507742639279, 6)

        numpy.random.seed(12)
        nocca, noccb = mol.nelec
        nmo = mf.mo_occ[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        c1a  = .1 * numpy.random.random((nocca,nvira))
        c1b  = .1 * numpy.random.random((noccb,nvirb))
        c2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
        c2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
        c2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
        cisdvec = myci.amplitudes_to_cisdvec(1., (c1a, c1b), (c2aa, c2ab, c2bb))

        hcisd0 = myci.contract(myci.amplitudes_to_cisdvec(1., (c1a,c1b), (c2aa,c2ab,c2bb)), eris)
        self.assertAlmostEqual(lib.fp(hcisd0), 466.56620234351681, 6)
        eris = myci.ao2mo(mf.mo_coeff)
        hcisd0 = myci.contract(cisdvec, eris)
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]])
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        h2e = fci.direct_uhf.absorb_h1e((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                        h1a.shape[0], mol.nelec, .5)
        nmo = mf.mo_coeff[0].shape[1]
        fcivec = myci.to_fcivec(cisdvec, nmo, mol.nelec)
        hci1 = fci.direct_uhf.contract_2e(h2e, fcivec, h1a.shape[0], mol.nelec)
        hci1 -= ehf0 * fcivec
        hcisd1 = myci.from_fcivec(hci1, nmo, mol.nelec)
        self.assertAlmostEqual(abs(hcisd1-hcisd0).max(), 0, 8)

        ecisd = myci.kernel(eris=eris)[0]
        efci = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                     h1a.shape[0], mol.nelec)[0]
        self.assertAlmostEqual(ecisd, -0.037067274690894436, 8)
        self.assertTrue(myci.e_tot-mol.energy_nuc() - efci < 0.002)

    def test_rdm_h4(self):
        mol = gto.Mole()
        mol.verbose = 7
        mol.output = '/dev/null'
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.spin = 2
        mol.basis = 'sto-3g'
        mol.build()
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        myci = ucisd.UCISD(mf)
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, -0.033689623198003449, 8)

        nmoa = nmob = nmo = mf.mo_coeff[1].shape[1]
        nocc = (6,4)
        ci0 = myci.to_fcivec(civec, nmo, nocc)
        ref1, ref2 = fci.direct_uhf.make_rdm12s(ci0, nmo, nocc)
        rdm1 = myci.make_rdm1(civec)
        rdm2 = myci.make_rdm2(civec)
        self.assertAlmostEqual(abs(rdm1[0]-ref1[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm1[1]-ref1[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[0]-ref2[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[1]-ref2[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[2]-ref2[2]).max(), 0, 6)

        dm1a = numpy.einsum('ijkk->ji', rdm2[0]) / (mol.nelectron-1)
        dm1a+= numpy.einsum('ijkk->ji', rdm2[1]) / (mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1[0] - dm1a).max(), 0, 9)
        dm1b = numpy.einsum('kkij->ji', rdm2[2]) / (mol.nelectron-1)
        dm1b+= numpy.einsum('kkij->ji', rdm2[1]) / (mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1[1] - dm1b).max(), 0, 9)

        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0], compact=False).reshape([nmoa]*4)
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1], compact=False).reshape([nmob]*4)
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]], compact=False)
        eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        e2 = (numpy.einsum('ij,ji', h1a, rdm1[0]) +
              numpy.einsum('ij,ji', h1b, rdm1[1]) +
              numpy.einsum('ijkl,ijkl', eri_aa, rdm2[0]) * .5 +
              numpy.einsum('ijkl,ijkl', eri_ab, rdm2[1])      +
              numpy.einsum('ijkl,ijkl', eri_bb, rdm2[2]) * .5)
        e2 += mol.energy_nuc()
        self.assertAlmostEqual(myci.e_tot, e2, 9)

    def test_rdm12(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.basis = {'H': 'sto-3g',
                     'O': 'sto-3g',}
        mol.build()
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        myci = mf.CISD()
        eris = myci.ao2mo()
        ecisd, civec = myci.kernel(eris=eris)
        self.assertAlmostEqual(ecisd, -0.048878084082066106, 8)

        nmoa = mf.mo_energy[0].size
        nmob = mf.mo_energy[1].size
        rdm1 = myci.make_rdm1(civec)
        rdm2 = myci.make_rdm2(civec)
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0], compact=False).reshape([nmoa]*4)
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1], compact=False).reshape([nmob]*4)
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]], compact=False)
        eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        e2 = (numpy.einsum('ij,ji', h1a, rdm1[0]) +
              numpy.einsum('ij,ji', h1b, rdm1[1]) +
              numpy.einsum('ijkl,ijkl', eri_aa, rdm2[0]) * .5 +
              numpy.einsum('ijkl,ijkl', eri_ab, rdm2[1])      +
              numpy.einsum('ijkl,ijkl', eri_bb, rdm2[2]) * .5)
        self.assertAlmostEqual(ecisd + mf.e_tot - mol.energy_nuc(), e2, 8)

        from_dm2 = (numpy.einsum('ijkk->ji', rdm2[0]) +
                    numpy.einsum('ijkk->ji', rdm2[1]))/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1[0] - from_dm2).max(), 0, 8)
        from_dm2 = (numpy.einsum('ijkk->ji', rdm2[2]) +
                    numpy.einsum('kkij->ji', rdm2[1]))/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1[1] - from_dm2).sum(), 0, 8)

    def test_ao_direct(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O', ( 0., 0.    , 0.   )],
            ['H', ( 0., -0.757, 0.587)],
            ['H', ( 0., 0.757 , 0.587)],]
        mol.spin = 2
        mol.basis = 'ccpvdz'
        mol.build()
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.max_memory = .1
        myci.frozen = [[1,2],[1,2]]
        myci.direct = True
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, -0.04754878399464485, 8)

    def test_trans_rdm_with_frozen(self):
        mol = gto.M(atom='''
        O   0.   0.       .0
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''', basis='sto3g')
        mf = scf.UHF(mol).run()

        def check_frozen(frozen):
            myci = ci.UCISD(mf)
            myci.frozen = frozen
            myci.nroots = 2
            myci.kernel()
            nocc = myci.nocc
            nmo = myci.nmo
            norb = mf.mo_coeff[0].shape[1]
            nfroz = len(frozen[0])
            cibra = (myci.ci[0] + myci.ci[1]) * numpy.sqrt(.5)
            fcibra = ucisd.to_fcivec(cibra, norb, mol.nelec, myci.frozen)
            fciket = ucisd.to_fcivec(myci.ci[1], norb, mol.nelec, myci.frozen)
            fcidm1 = fci.direct_spin1.trans_rdm1s(fcibra, fciket, norb, mol.nelec)
            cidm1  = myci.trans_rdm1(cibra, myci.ci[1], nmo, nocc)
            self.assertAlmostEqual(abs(fcidm1[0]-cidm1[0]).max(), 0, 12)
            self.assertAlmostEqual(abs(fcidm1[1]-cidm1[1]).max(), 0, 12)

        check_frozen([[5], [6]])
        check_frozen([[3], [5]])
        check_frozen([[1,3], [2,5]])
        check_frozen([[2,5], [5]])

    def test_overlap(self):
        nmo = 8
        nocc = nocca, noccb = (4,3)
        numpy.random.seed(2)
        nvira, nvirb = nmo-nocca, nmo-noccb
        cibra = ucisd.amplitudes_to_cisdvec(numpy.random.rand(1)[0],
                                            (numpy.random.rand(nocca,nvira),
                                             numpy.random.rand(noccb,nvirb)),
                                            (numpy.random.rand(nocca,nocca,nvira,nvira),
                                             numpy.random.rand(nocca,noccb,nvira,nvirb),
                                             numpy.random.rand(noccb,noccb,nvirb,nvirb)))
        ciket = ucisd.amplitudes_to_cisdvec(numpy.random.rand(1)[0],
                                            (numpy.random.rand(nocca,nvira),
                                             numpy.random.rand(noccb,nvirb)),
                                            (numpy.random.rand(nocca,nocca,nvira,nvira),
                                             numpy.random.rand(nocca,noccb,nvira,nvirb),
                                             numpy.random.rand(noccb,noccb,nvirb,nvirb)))
        fcibra = ucisd.to_fcivec(cibra, nmo, nocc)
        fciket = ucisd.to_fcivec(ciket, nmo, nocc)
        s_mo = numpy.random.random((nmo,nmo))
        s0 = fci.addons.overlap(fcibra, fciket, nmo, nocc, s_mo)
        s1 = ucisd.overlap(cibra, ciket, nmo, nocc, (s_mo, s_mo))
        self.assertAlmostEqual(s1, s0, 9)

    def test_cisdvec_to_amplitudes_overwritten(self):
        mol = gto.M()
        myci = scf.UHF(mol).apply(ci.UCISD)
        nelec = (3, 3)
        nocc = nelec
        nmo = (5, 5)
        myci.nocc = nocc
        myci.nmo = nmo
        vec = numpy.zeros(myci.vector_size())
        vec_orig = vec.copy()
        c0, t1, t2 = myci.cisdvec_to_amplitudes(vec)
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        t1a[:] = 1
        t1b[:] = 1
        t2aa[:] = 1
        t2ab[:] = 1
        t2bb[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    def test_with_df_s0(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.basis = '631g'
        mol.build()
        rhf = scf.RHF(mol).density_fit(auxbasis='weigend')
        rhf.conv_tol_grad = 1e-8
        rhf.kernel()
        mf = scf.addons.convert_to_uhf(rhf)
        myci = ci.UCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -76.1131374309989, 8)

    def test_with_df_s2(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.basis = '631g'
        mol.spin = 2
        mol.build()
        mf = scf.UHF(mol).density_fit(auxbasis='weigend')
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        myci = ci.UCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -75.8307298990769, 8)


if __name__ == "__main__":
    print("Full Tests for UCISD")
    unittest.main()
