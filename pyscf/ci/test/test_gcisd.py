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
        h1 = h1 + h1.T + numpy.diag(numpy.arange(nmo))
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
        self.assertAlmostEqual(lib.fp(cinew), -102.17887236599671, 9)

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

        vec1 = gcisd.from_ucisdvec(ucisd.amplitudes_to_cisdvec(1, (c1a,c1b), (c2aa,c2ab,c2bb)),
                                   nocc*2, orbspin)
        self.assertAlmostEqual(abs(cisdvec - vec1).max(), 0, 12)

        c1 = gcisd.spatial2spin((c1a, c1a), orbspin)
        c2aa = c2ab - c2ab.transpose(1,0,2,3)
        c2 = gcisd.spatial2spin((c2aa, c2ab, c2aa), orbspin)
        cisdvec = gcisd.amplitudes_to_cisdvec(1., c1, c2)
        vec1 = gcisd.from_rcisdvec(ci.cisd.amplitudes_to_cisdvec(1, c1a, c2ab), nocc*2, orbspin)
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
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 6)

        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        myci.kernel()
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 6)

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
        self.assertAlmostEqual(myci.e_tot, -0.86423570617209888, 6)

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
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[idxa][:,idxa]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[idxb][:,idxb]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[idxa][:,idxa][:,:,idxa][:,:,:,idxa]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[idxa][:,idxa][:,:,idxb][:,:,:,idxb]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[idxb][:,idxb][:,:,idxb][:,:,:,idxb]).max(), 0, 5)

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
        self.assertEqual(cisdvec.size, myci.vector_size())

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
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.GCISD(mf)
        eris = myci.ao2mo()
        ecisd, civec = myci.kernel(eris=eris)
        self.assertAlmostEqual(ecisd, -0.035165114624046617, 6)

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
              numpy.einsum('ijkl,ijkl', eri, rdm2) * .5)
        e2 += mol.energy_nuc()
        self.assertAlmostEqual(myci.e_tot, e2, 9)

        dm1 = numpy.einsum('ijkk->ji', rdm2)/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1 - dm1).max(), 0, 9)

    def test_rdm_real(self):
        mol = gto.M()
        mol.verbose = 0
        nocc = 6
        nvir = 10
        mf = scf.GHF(mol)
        nmo = nocc + nvir
        npair = nmo*(nmo//2+1)//4
        numpy.random.seed(12)
        mf._eri = numpy.random.random(npair*(npair+1)//2)*.3
        hcore = numpy.random.random((nmo,nmo)) * .5
        hcore = hcore + hcore.T + numpy.diag(range(nmo))*2
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        mf.mo_coeff = numpy.eye(nmo)
        mf.mo_occ = numpy.zeros(nmo)
        mf.mo_occ[:nocc] = 1
        dm1 = mf.make_rdm1()
        mf.e_tot = mf.energy_elec()[0]
        myci = gcisd.GCISD(mf).run()
        dm1 = myci.make_rdm1()
        dm2 = myci.make_rdm2()

        nao = nmo // 2
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        eri  = ao2mo.kernel(mf._eri, mo_a)
        eri += ao2mo.kernel(mf._eri, mo_b)
        eri1 = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b))
        eri += eri1
        eri += eri1.T
        eri = ao2mo.restore(1, eri, nmo)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        self.assertAlmostEqual(e1, myci.e_tot, 7)

        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2).conj()).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(2,1,0,3)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(0,3,2,1)       ).max(), 0, 9)

    def test_rdm_complex(self):
        mol = gto.M()
        mol.verbose = 0
        nocc = 4
        nvir = 6
        mf = scf.GHF(mol)
        nmo = nocc + nvir
        numpy.random.seed(1)
        eri = (numpy.random.random((nmo,nmo,nmo,nmo)) +
               numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri = eri + eri.transpose(1,0,3,2).conj()
        eri = eri + eri.transpose(2,3,0,1)
        eri *= .1

        def get_jk(mol, dm, *args,**kwargs):
            vj = numpy.einsum('ijkl,lk->ij', eri, dm)
            vk = numpy.einsum('ijkl,jk->il', eri, dm)
            return vj, vk
        def get_veff(mol, dm, *args, **kwargs):
            vj, vk = get_jk(mol, dm)
            return vj - vk
        def ao2mofn(mos):
            return eri

        mf.get_jk = get_jk
        mf.get_veff = get_veff
        hcore = numpy.random.random((nmo,nmo)) * .2 + numpy.random.random((nmo,nmo))* .2j
        hcore = hcore + hcore.T.conj() + numpy.diag(range(nmo))*2
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        orbspin = numpy.zeros(nmo, dtype=int)
        orbspin[1::2] = 1
        mf.mo_coeff = lib.tag_array(numpy.eye(nmo) + 0j, orbspin=orbspin)
        mf.mo_occ = numpy.zeros(nmo)
        mf.mo_occ[:nocc] = 1
        mf.e_tot = mf.energy_elec(mf.make_rdm1(), hcore)[0]

        myci = gcisd.GCISD(mf)
        eris = gcisd.gccsd._make_eris_incore(myci, mf.mo_coeff, ao2mofn)
        myci.ao2mo = lambda *args, **kwargs: eris
        myci.kernel(eris=eris)
        dm1 = myci.make_rdm1()
        dm2 = myci.make_rdm2()

        e1 = numpy.einsum('ij,ji', hcore, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        self.assertAlmostEqual(e1, myci.e_tot, 7)

        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2).conj()).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(2,1,0,3)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(0,3,2,1)       ).max(), 0, 9)

    def test_rdm_vs_ucisd(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.basis = '631g'
        mol.spin = 2
        mol.build()
        mf = scf.UHF(mol).run()
        myuci = ucisd.UCISD(mf)
        myuci.frozen = 1
        myuci.kernel()
        udm1 = myuci.make_rdm1()
        udm2 = myuci.make_rdm2()

        mf = scf.addons.convert_to_ghf(mf)
        mygci = gcisd.GCISD(mf)
        mygci.frozen = 2
        mygci.kernel()
        dm1 = mygci.make_rdm1()
        dm2 = mygci.make_rdm2()

        nao = mol.nao_nr()
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        nmo = mo_a.shape[1]
        eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
        orbspin = mf.mo_coeff.orbspin
        sym_forbid = (orbspin[:,None] != orbspin)
        eri[sym_forbid,:,:] = 0
        eri[:,:,sym_forbid] = 0
        hcore = scf.RHF(mol).get_hcore()
        h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mygci.e_tot, 7)

        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        self.assertAlmostEqual(abs(dm1[idxa[:,None],idxa] - udm1[0]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm1[idxb[:,None],idxb] - udm1[1]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa] - udm2[0]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb] - udm2[1]).max(), 0, 4)
        self.assertAlmostEqual(abs(dm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb] - udm2[2]).max(), 0, 4)

        c0, c1, c2 = myuci.cisdvec_to_amplitudes(myuci.ci)
        ut1 = [0] * 2
        ut2 = [0] * 3
        ut0 = c0 + .2j
        ut1[0] = c1[0] + numpy.cos(c1[0]) * .2j
        ut1[1] = c1[1] + numpy.cos(c1[1]) * .2j
        ut2[0] = c2[0] + numpy.sin(c2[0]) * .8j
        ut2[1] = c2[1] + numpy.sin(c2[1]) * .8j
        ut2[2] = c2[2] + numpy.sin(c2[2]) * .8j
        civec = myuci.amplitudes_to_cisdvec(ut0, ut1, ut2)
        udm1 = myuci.make_rdm1(civec)
        udm2 = myuci.make_rdm2(civec)

        gt1 = mygci.spatial2spin(ut1)
        gt2 = mygci.spatial2spin(ut2)
        civec = mygci.amplitudes_to_cisdvec(ut0, gt1, gt2)
        gdm1 = mygci.make_rdm1(civec)
        gdm2 = mygci.make_rdm2(civec)

        self.assertAlmostEqual(abs(gdm1[idxa[:,None],idxa] - udm1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(gdm1[idxb[:,None],idxb] - udm1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(gdm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa] - udm2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(gdm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb] - udm2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(gdm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb] - udm2[2]).max(), 0, 9)

    def test_rdm_vs_rcisd(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.basis = '631g'
        mol.build()
        mf = scf.RHF(mol).run()
        myrci = ci.cisd.CISD(mf).run()
        rdm1 = myrci.make_rdm1()
        rdm2 = myrci.make_rdm2()

        mf = scf.addons.convert_to_ghf(mf)
        mygci = gcisd.GCISD(mf).run()
        dm1 = mygci.make_rdm1()
        dm2 = mygci.make_rdm2()

        nao = mol.nao_nr()
        mo_a = mf.mo_coeff[:nao]
        mo_b = mf.mo_coeff[nao:]
        nmo = mo_a.shape[1]
        eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
        orbspin = mf.mo_coeff.orbspin
        sym_forbid = (orbspin[:,None] != orbspin)
        eri[sym_forbid,:,:] = 0
        eri[:,:,sym_forbid] = 0
        hcore = scf.RHF(mol).get_hcore()
        h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mygci.e_tot, 7)

        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        trdm1 = dm1[idxa[:,None],idxa]
        trdm1+= dm1[idxb[:,None],idxb]
        trdm2 = dm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa]
        trdm2+= dm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb]
        dm2ab = dm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb]
        trdm2+= dm2ab
        trdm2+= dm2ab.transpose(2,3,0,1)
        self.assertAlmostEqual(abs(trdm1 - rdm1).max(), 0, 4)
        self.assertAlmostEqual(abs(trdm2 - rdm2).max(), 0, 4)

        c0, c1, c2 = myrci.cisdvec_to_amplitudes(myrci.ci)
        rt0 = c0 + .2j
        rt1 = c1 + numpy.cos(c1) * .2j
        rt2 = c2 + numpy.sin(c2) * .8j
        civec = myrci.amplitudes_to_cisdvec(rt0, rt1, rt2)
        rdm1 = myrci.make_rdm1(civec)
        rdm2 = myrci.make_rdm2(civec)

        gt1 = mygci.spatial2spin(rt1)
        gt2 = mygci.spatial2spin(rt2)
        civec = mygci.amplitudes_to_cisdvec(rt0, gt1, gt2)
        gdm1 = mygci.make_rdm1(civec)
        gdm2 = mygci.make_rdm2(civec)

        orbspin = mf.mo_coeff.orbspin
        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        trdm1 = gdm1[idxa[:,None],idxa]
        trdm1+= gdm1[idxb[:,None],idxb]
        trdm2 = gdm2[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa]
        trdm2+= gdm2[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb]
        dm2ab = gdm2[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb]
        trdm2+= dm2ab
        trdm2+= dm2ab.transpose(2,3,0,1)
        self.assertAlmostEqual(abs(trdm1 - rdm1).max(), 0, 9)
        self.assertAlmostEqual(abs(trdm2 - rdm2).max(), 0, 9)

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
        self.assertAlmostEqual(ecisd, -0.048829195509732602, 6)

    def test_trans_rdm1(self):
        numpy.random.seed(1)
        norb = 4
        nocc = 2
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

        fcidm1 = fci.direct_spin1.trans_rdm1s(fcibra, fciket, norb, nocc*2)
        myci1 = ci.GCISD(scf.GHF(gto.M()))
        myci1.nmo = norb = 8
        myci1.nocc = nocc = 4
        orbspin = numpy.zeros(norb, dtype=int)
        orbspin[1::2] = 1
        myci1.mo_coeff = lib.tag_array(numpy.eye(norb), orbspin=orbspin)
        myci1.mo_occ = numpy.zeros(norb)
        myci1.mo_occ[:nocc] = 1
        cibra = myci1.from_rcisdvec(cibra, (nocc//2,nocc//2), orbspin)
        ciket = myci1.from_rcisdvec(ciket)
        cidm1 = myci1.trans_rdm1(cibra, ciket, norb, nocc)
        self.assertAlmostEqual(abs(cidm1[0::2,0::2] - fcidm1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(cidm1[1::2,1::2] - fcidm1[1]).max(), 0, 12)

        cibra = myci1.to_ucisdvec(cibra, orbspin)
        ciket = myci1.to_ucisdvec(ciket)
        myci2 = ci.UCISD(scf.UHF(gto.M()))
        cidm1 = myci2.trans_rdm1(cibra, ciket, (norb//2,norb//2), (nocc//2,nocc//2))
        self.assertAlmostEqual(abs(cidm1[0] - fcidm1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(cidm1[1] - fcidm1[1]).max(), 0, 12)

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
        mf = scf.GHF(mol).run()
        myci = ci.GCISD(mf)
        myci.nroots = 3
        myci.run()
        self.assertAlmostEqual(myci.e_tot[2], -1.9802158893844912, 6)

    def test_trans_rdm_with_frozen(self):
        mol = gto.M(atom='''
        O   0.   0.       .0
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''', basis='sto3g')
        mf = scf.convert_to_ghf(scf.RHF(mol).run())
        orbspin = mf.mo_coeff.orbspin
        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        nmo_1c = mf.mo_coeff.shape[1]//2

        def check_frozen(frozen):
            myci = ci.GCISD(mf)
            myci.frozen = frozen
            myci.nroots = 3
            myci.kernel()
            nocc = myci.nocc
            nmo = myci.nmo
            nfroz = len(frozen)
            try:
                ket_id = 1
                fciket = gcisd.to_fcivec(myci.ci[ket_id], mol.nelectron, orbspin, myci.frozen)
            except RuntimeError:
                ket_id = 2
                fciket = gcisd.to_fcivec(myci.ci[ket_id], mol.nelectron, orbspin, myci.frozen)
                # spin-forbidden transition
                cidm1  = myci.trans_rdm1(myci.ci[0], myci.ci[1], nmo, nocc)
                self.assertAlmostEqual(abs(cidm1[idxa[:,None],idxa]).max(), 0, 7)
                self.assertAlmostEqual(abs(cidm1[idxb[:,None],idxb]).max(), 0, 7)

            cibra = (myci.ci[0] + myci.ci[ket_id]) * numpy.sqrt(.5)
            fcibra = gcisd.to_fcivec(cibra, mol.nelectron, orbspin, myci.frozen)
            fcidm1 = fci.direct_spin1.trans_rdm1s(fcibra, fciket, nmo_1c, mol.nelectron)
            cidm1  = myci.trans_rdm1(cibra, myci.ci[ket_id], nmo, nocc)
            self.assertAlmostEqual(abs(fcidm1[0]-cidm1[idxa[:,None],idxa]).max(), 0, 12)
            self.assertAlmostEqual(abs(fcidm1[1]-cidm1[idxb[:,None],idxb]).max(), 0, 12)

        check_frozen([10])
        check_frozen([10,3])

    def test_cisdvec_to_amplitudes_overwritten(self):
        mol = gto.M()
        myci = scf.GHF(mol).apply(ci.GCISD)
        nelec = (3,3)
        nocc, nvir = sum(nelec), 4
        nmo = nocc + nvir
        myci.nocc = nocc
        myci.nmo = nmo
        vec = numpy.zeros(myci.vector_size())
        vec_orig = vec.copy()
        c0, c1, c2 = myci.cisdvec_to_amplitudes(vec)
        c1[:] = 1
        c2[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)


if __name__ == "__main__":
    print("Full Tests for GCISD")
    unittest.main()
