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
from functools import reduce

from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import fci
from pyscf import ci
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
        nelec = (3,3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        numpy.random.seed(12)
        civec = numpy.random.random(1+nocc*nvir+nocc**2*nvir**2)
        ci0 = ci.cisd.to_fcivec(civec, nmo, nelec)
        self.assertAlmostEqual(abs(civec-ci.cisd.from_fcivec(ci0, nmo, nelec)).sum(), 0, 9)

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
        self.assertAlmostEqual(ecisd, -0.024780739973407784, 8)

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
        myci.frozen = None
        eris = myci.ao2mo()
        ecisd, civec = myci.kernel(eris=eris)
        self.assertAlmostEqual(ecisd, -0.048878084082066106, 8)

        nmo = mf.mo_coeff.shape[1]
        nocc = mol.nelectron//2
        ci0 = myci.to_fcivec(civec, nmo, nocc*2)
        ref1, ref2 = fci.direct_spin1.make_rdm12(ci0, nmo, nocc*2)
        rdm1 = myci.make_rdm1(civec)
        rdm2 = myci.make_rdm2(civec)
        self.assertTrue(numpy.allclose(rdm2, ref2))
        self.assertAlmostEqual(lib.finger(rdm1), 2.2685303884654933, 5)
        self.assertAlmostEqual(lib.finger(rdm2),-3.7944286346871299, 5)
        self.assertAlmostEqual(abs(rdm2-rdm2.transpose(2,3,0,1)).sum(), 0, 9)
        self.assertAlmostEqual(abs(rdm2-rdm2.transpose(1,0,3,2)).sum(), 0, 9)
        h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        h2e = ao2mo.kernel(mf._eri, mf.mo_coeff)
        h2e = ao2mo.restore(1, h2e, nmo)
        e2 = (numpy.dot(h1e.flatten(),rdm1.flatten()) +
              numpy.dot(h2e.flatten(),rdm2.flatten()) * .5)
        self.assertAlmostEqual(ecisd + mf.e_tot - mol.energy_nuc(), e2, 9)
        dm1 = numpy.einsum('ijkk->ij', rdm2)/(mol.nelectron-1)
        self.assertAlmostEqual(abs(rdm1 - dm1).sum(), 0, 9)

        doo, dov, dvo, dvv = \
                ci.cisd._gamma1_intermediates(myci, civec, nmo, nocc)
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
                ci.cisd._gamma2_intermediates(myci, civec, nmo, nocc)

        nvir = nmo - nocc
        fock0 = eris.fock
        tril2sq = lib.square_mat_in_trilu_indices(nvir)
        e2 =(numpy.einsum('ijkl,ijkl', doooo, eris.oooo)*2
            +numpy.einsum('acbd,acbd', dvvvv, ao2mo.restore(1, eris.vvvv, nvir))*2
            +numpy.einsum('jkia,iajk', dooov, eris.ovoo)*2
            +numpy.einsum('icba,icba', dovvv, eris.ovvv[:,:,tril2sq])*2
            +numpy.einsum('iajb,iabj', dovov, eris.ovvo)*2
            +numpy.einsum('jbai,jbai', dovvo, eris.ovvo)*2
            +numpy.einsum('ijab,ijab', doovv, eris.oovv)*2
            +numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc])*2
            +numpy.einsum('ia,ia', dov, fock0[:nocc,nocc:])*2
            +numpy.einsum('ai,ai', dvo, fock0[nocc:,:nocc])*2
            +numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:])*2
            )
        self.assertAlmostEqual(e2, ecisd, 9)

    def test_trans_rdm(self):
        numpy.random.seed(1)
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
        self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 9)
        self.assertAlmostEqual(abs(fcidm2-cidm2).max(), 0, 9)

        fcidm1 = fci.direct_spin1.trans_rdm1(fcibra, fciket, norb, nocc*2)
        cidm1  = ci.cisd.trans_rdm1(myci, cibra, ciket, norb, nocc)
        self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 9)

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
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        myci = ci.CISD(mf)
        myci.max_memory = .1
        myci.nmo = 16
        myci.nocc = 5
        myci.direct = True
        ecisd, civec = myci.kernel()
        self.assertAlmostEqual(ecisd, -0.1319371817220385, 8)

    def test_dump_chk(self):
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
        self.assertAlmostEqual(myci.e_corr, -0.18730699567992737, 8)

    def test_trans_rdm1(self):
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

        fcidm2 = fci.direct_spin1.make_rdm12(fciket, norb, nocc*2)[1]
        cidm2  = ci.cisd.make_rdm2(myci, ciket, norb, nocc)
        self.assertAlmostEqual(abs(fcidm2-cidm2).max(), 0, 12)

        fcidm1, fcidm2 = fci.direct_spin1.trans_rdm12(fcibra, fciket, norb, nocc*2)
        cidm1 = ci.cisd.trans_rdm1(myci, cibra, ciket, norb, nocc)
        self.assertAlmostEqual(abs(fcidm1-cidm1).max(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for CISD")
    unittest.main()

