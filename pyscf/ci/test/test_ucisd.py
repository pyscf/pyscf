#!/usr/bin/env python
import unittest
import numpy
import scipy.linalg

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import fci
from pyscf import ci
from pyscf import ao2mo
from pyscf.ci import ucisd


class KnownValues(unittest.TestCase):
    def test_contract(self):
        '''cross check with GCISD'''
        mol = gto.M()
        mol.nelectron = 6
        nocc, nvir = mol.nelectron//2, 4
        nmo = nocc + nvir
        nmo_pair = nmo*(nmo+1)//2
        mf = scf.UHF(mol)
        numpy.random.seed(12)
        mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2) * .2
        mf.mo_coeff = numpy.random.random((2,nmo,nmo))
        mf.mo_energy = [numpy.arange(0., nmo)]*2
        mf.mo_occ = numpy.zeros((2,nmo))
        mf.mo_occ[:,:nocc] = 1
        h1 = numpy.random.random((nmo,nmo)) * .1
        h1 = h1 + h1.T + numpy.arange(nmo)
        mf.get_hcore = lambda *args: h1

        mf1 = scf.addons.convert_to_ghf(mf)
        mf1.get_hcore = lambda *args: scipy.linalg.block_diag(h1, h1)
        gci = ci.GCISD(mf1)
        c2 = numpy.random.random((nocc*2,nocc*2,nvir*2,nvir*2)) * .1 - .1
        c2 = c2 - c2.transpose(0,1,3,2)
        c2 = c2 - c2.transpose(1,0,2,3)
        c1 = numpy.random.random((nocc*2,nvir*2)) * .1
        c0 = .5
        civec = gci.amplitudes_to_cisdvec(c0, c1, c2)
        civecref = gci.contract(civec, gci.ao2mo())
        c0ref, c1ref, c2ref = gci.cisdvec_to_amplitudes(civecref)
        c1ref = gci.spin2spatial(c1ref)
        c2ref = gci.spin2spatial(c2ref)

        c1 = gci.spin2spatial(c1)
        c2 = gci.spin2spatial(c2)
        myci = ci.UCISD(mf)
        civec = myci.amplitudes_to_cisdvec(c0, c1, c2)
        cinew = myci.contract(civec, myci.ao2mo())
        c0new, c1new, c2new = myci.cisdvec_to_amplitudes(cinew)
        self.assertAlmostEqual(abs(c0new   -c0ref   ).max(), 0, 12)
        self.assertAlmostEqual(abs(c1new[0]-c1ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(c1new[1]-c1ref[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(c2new[0]-c2ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(c2new[1]-c2ref[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(c2new[2]-c2ref[2]).max(), 0, 12)
        self.assertAlmostEqual(lib.finger(cinew), -123.57726507299601, 9)

    def test_from_fcivec(self):
        mol = gto.M()
        nocca, noccb = nelec = (3,2)
        nvira, nvirb = 5, 6
        nmo = (8,8)
        numpy.random.seed(12)
        civec = numpy.random.random(1+nocca*nvira+noccb*nvirb
                                    +nocca*(nocca-1)//2*nvira*(nvira-1)//2
                                    +noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
                                    +nocca*noccb*nvira*nvirb)
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
        fcivec = ucisd.to_fcivec(cisdvec, (nmo,nmo), (nocc,nocc))
        cisdvec1 = ucisd.from_fcivec(fcivec, (nmo,nmo), (nocc,nocc))
        self.assertAlmostEqual(abs(cisdvec-cisdvec1).max(), 0, 12)
        ci1 = ucisd.to_fcivec(cisdvec1, (nmo,nmo), (nocc,nocc))
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
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[2]).max(), 0, 5)

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
        self.assertAlmostEqual(lib.finger(myci.make_diagonal(eris)),
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
        self.assertAlmostEqual(lib.finger(hcisd0), 466.56620234351681, 8)
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
        nmo = (mf.mo_coeff[0].shape[1],mf.mo_coeff[1].shape[1])
        fcivec = myci.to_fcivec(cisdvec, nmo, mol.nelec)
        hci1 = fci.direct_uhf.contract_2e(h2e, fcivec, h1a.shape[0], mol.nelec)
        hci1 -= ehf0 * fcivec
        hcisd1 = myci.from_fcivec(hci1, nmo, mol.nelec)
        self.assertAlmostEqual(abs(hcisd1-hcisd0).max(), 0, 9)

        ecisd = myci.kernel(eris=eris)[0]
        efci = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                     h1a.shape[0], mol.nelec)[0]
        self.assertAlmostEqual(ecisd, -0.037067274690894436, 9)
        self.assertTrue(myci.e_tot-mol.energy_nuc() - efci < 0.002)

    def test_rdm(self):
        mol = gto.Mole()
        mol.verbose = 5
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
        ci0 = myci.to_fcivec(civec, (nmoa,nmob), nocc)
        ref1, ref2 = fci.direct_uhf.make_rdm12s(ci0, nmo, nocc)
        rdm1 = myci.make_rdm1(civec)
        rdm2 = myci.make_rdm2(civec)
        self.assertAlmostEqual(abs(rdm1[0]-ref1[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm1[1]-ref1[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[0]-ref2[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[1]-ref2[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(rdm2[2]-ref2[2]).max(), 0, 6)

        dm1a = numpy.einsum('ijkk->ij', rdm2[0]) / (mol.nelectron-1)
        dm1a+= numpy.einsum('ijkk->ij', rdm2[1]) / (mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1[0] - dm1a).max(), 0, 9)
        dm1b = numpy.einsum('kkij->ij', rdm2[2]) / (mol.nelectron-1)
        dm1b+= numpy.einsum('kkij->ij', rdm2[1]) / (mol.nelectron-1)
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


if __name__ == "__main__":
    print("Full Tests for CISD")
    unittest.main()

