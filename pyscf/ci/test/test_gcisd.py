#!/usr/bin/env python
import unittest
import numpy
import scipy.linalg

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import fci
from pyscf import ci
from pyscf.ci import gcisd
from pyscf.ci import ucisd
from pyscf import ao2mo


class KnownValues(unittest.TestCase):
    def test_contract(self):
        '''cross check with UCISD'''
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
        numpy.random.seed(12)
        nocc = 3
        nvir = 5
        nmo = nocc + nvir

        orbspin = numpy.zeros(nmo*2, dtype=int)
        orbspin[1::2] = 1
        c1a = numpy.random.random((nocc,nvir))
        c1b = numpy.random.random((nocc,nvir))
        c2aa = numpy.random.random((nocc,nocc,nvir,nvir))
        c2bb = numpy.random.random((nocc,nocc,nvir,nvir))
        c2ab = numpy.random.random((nocc,nocc,nvir,nvir))
        c2ab = c2ab + c2ab.transpose(1,0,3,2)
        c1 = gcisd.spatial2spin((c1a, c1b), orbspin)
        c2 = gcisd.spatial2spin((c2aa, c2ab, c2bb), orbspin)
        cisdvec = gcisd.amplitudes_to_cisdvec(1., c1, c2)

        fcivec = gcisd.to_fcivec(cisdvec, nocc*2, orbspin)
        cisdvec1 = gcisd.from_fcivec(fcivec, nocc*2, orbspin)
        self.assertAlmostEqual(abs(cisdvec-cisdvec1).max(), 0, 12)
        ci1 = gcisd.to_fcivec(cisdvec1, nocc*2, orbspin)
        self.assertAlmostEqual(abs(fcivec-ci1).max(), 0, 12)

        vec1 = gcisd.from_rcisdvec(ucisd.amplitudes_to_cisdvec(1, (c1a,c1b), (c2aa,c2ab,c2bb)),
                                   nocc, orbspin)
        self.assertTrue(numpy.all(cisdvec == vec1))

        c1 = gcisd.spatial2spin((c1a, c1a), orbspin)
        c2aa = c2ab - c2ab.transpose(1,0,2,3)
        c2 = gcisd.spatial2spin((c2aa, c2ab, c2aa), orbspin)
        cisdvec = gcisd.amplitudes_to_cisdvec(1., c1, c2)
        vec1 = gcisd.from_rcisdvec(ci.cisd.amplitudes_to_cisdvec(1, c1a, c2ab), nocc, orbspin)
        self.assertTrue(numpy.all(cisdvec == vec1))

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
        mol.spin = 2
        mol.basis = '3-21g'
        mol.build()
        mf = scf.GHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 8)

        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 8)

        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['H', ( 1.,-1.    , 0.   )],
            ['H', ( 0.,-1.    ,-1.   )],
            ['H', ( 1.,-0.5   , 0.   )],
            ['H', ( 0.,-1.    , 1.   )],
        ]
        mol.charge = 2
        mol.spin = 0
        mol.basis = '3-21g'
        mol.build()
        mf = scf.UHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 8)

        mf = scf.UHF(mol).run(conv_tol=1e-14)
        gmf = scf.addons.convert_to_ghf(mf)
        ehf0 = mf.e_tot - mol.energy_nuc()
        myci = gcisd.GCISD(gmf)
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
        self.assertAlmostEqual(myci.e_tot-mol.energy_nuc(), efci, 9)
        dm1ref, dm2ref = fci.direct_uhf.make_rdm12s(fcivec, h1a.shape[0], mol.nelec)
        nmo = myci.nmo
        rdm1 = myci.make_rdm1(myci.ci, nmo, mol.nelectron)
        rdm2 = myci.make_rdm2(myci.ci, nmo, mol.nelectron)
        idxa = eris.orbspin == 0
        idxb = eris.orbspin == 1
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[idxa][:,idxa]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[idxb][:,idxb]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[idxa][:,idxa][:,:,idxa][:,:,:,idxa]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[idxa][:,idxa][:,:,idxb][:,:,:,idxb]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[idxb][:,idxb][:,:,idxb][:,:,:,idxb]).max(), 0, 6)

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
        gmf = scf.addons.convert_to_ghf(mf)
        myci = ci.GCISD(gmf)
        eris = myci.ao2mo()

        numpy.random.seed(12)
        nocca, noccb = mol.nelec
        nmo = mol.nao_nr()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        #cisdvec = myci.get_init_guess(eris)[1]
        c1a  = .1 * numpy.random.random((nocca,nvira))
        c1b  = .1 * numpy.random.random((noccb,nvirb))
        c2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
        c2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
        c2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))
        c1 = myci.spatial2spin((c1a, c1b))
        c2 = myci.spatial2spin((c2aa, c2ab, c2bb))
        cisdvec = myci.amplitudes_to_cisdvec(1., c1, c2)

        hcisd0 = myci.contract(cisdvec, eris)
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]])
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        h2e = fci.direct_uhf.absorb_h1e((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                        h1a.shape[0], mol.nelec, .5)
        fcivec = myci.to_fcivec(cisdvec, mol.nelectron, eris.orbspin)
        hci1 = fci.direct_uhf.contract_2e(h2e, fcivec, h1a.shape[0], mol.nelec)
        hci1 -= ehf0 * fcivec
        hcisd1 = myci.from_fcivec(hci1, mol.nelectron, eris.orbspin)
        self.assertAlmostEqual(abs(hcisd1-hcisd0).max(), 0, 9)

        hdiag0 = myci.make_diagonal(eris)
        hdiag0 = myci.to_fcivec(hdiag0, mol.nelectron, eris.orbspin).ravel()
        hdiag0 = myci.from_fcivec(hdiag0, mol.nelectron, eris.orbspin).ravel()
        hdiag1 = fci.direct_uhf.make_hdiag((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                           h1a.shape[0], mol.nelec)
        hdiag1 = myci.from_fcivec(hdiag1, mol.nelectron, eris.orbspin).ravel()
        self.assertAlmostEqual(abs(abs(hdiag0)-abs(hdiag1)).max(), 0, 9)

        ecisd = myci.kernel()[0]
        efci = fci.direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                     h1a.shape[0], mol.nelec)[0]
        self.assertAlmostEqual(ecisd, -0.037067274690894436, 9)
        self.assertTrue(myci.e_tot-mol.energy_nuc() - efci < 0.002)

    def test_rdm(self):
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
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        eris = myci.ao2mo()
        ecisd, civec = myci.kernel(eris=eris)
        self.assertAlmostEqual(ecisd, -0.035165114624046617, 8)

        nmo = eris.mo_coeff.shape[1]
        rdm1 = myci.make_rdm1(civec, nmo, mol.nelectron)
        rdm2 = myci.make_rdm2(civec, nmo, mol.nelectron)

        mo = eris.mo_coeff[:7] + eris.mo_coeff[7:]
        eri = ao2mo.kernel(mf._eri, mo, compact=False).reshape([nmo]*4)
        eri[eris.orbspin[:,None]!=eris.orbspin,:,:] = 0
        eri[:,:,eris.orbspin[:,None]!=eris.orbspin] = 0
        h1e = reduce(numpy.dot, (mo.T, mf.get_hcore(), mo))
        h1e[eris.orbspin[:,None]!=eris.orbspin] = 0
        e2 = (numpy.einsum('ij,ji', h1e, rdm1) +
              numpy.einsum('ijkl,jilk', eri, rdm2) * .5)
        e2 += mol.energy_nuc()
        self.assertAlmostEqual(myci.e_tot, e2, 9)

        dm1 = numpy.einsum('ijkk->ij', rdm2)/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1 - dm1).max(), 0, 9)

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
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        myci.max_memory = .1
        myci.frozen = [2,3,4,5]
        myci.direct = True
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, -0.048829195509732602, 8)


if __name__ == "__main__":
    print("Full Tests for GCISD")
    unittest.main()


