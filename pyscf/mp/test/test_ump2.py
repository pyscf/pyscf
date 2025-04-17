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
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_ump2(self):
        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.16575150552336643, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.042627186675330754, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.12312431898078077, 8)

        pt.max_memory = 1
        pt.frozen = []
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.16575150552336643, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.042627186675330754, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.12312431898078077, 8)

    def test_ump2_dm(self):
        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel()
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()
        gpt = mp.GMP2(mf).run()
        dm1ref = gpt.make_rdm1()
        dm2ref = gpt.make_rdm2()
        ia = gpt._scf.mo_coeff.orbspin == 0
        ib = gpt._scf.mo_coeff.orbspin == 1
        mo_a, mo_b = mf.mo_coeff
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        nocca, noccb = mol.nelec

        self.assertTrue(numpy.allclose(dm1[0], dm1ref[ia][:,ia]))
        self.assertTrue(numpy.allclose(dm1[1], dm1ref[ib][:,ib]))
        self.assertTrue(numpy.allclose(dm2[0], dm2ref[ia][:,ia][:,:,ia][:,:,:,ia]))
        self.assertTrue(numpy.allclose(dm2[2], dm2ref[ib][:,ib][:,:,ib][:,:,:,ib]))
        self.assertTrue(numpy.allclose(dm2[1], dm2ref[ia][:,ia][:,:,ib][:,:,:,ib]))

        hcore = mf.get_hcore()
        eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
        eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
        eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
        eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
        h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
        e1 = numpy.einsum('ij,ji', h1a, dm1[0])
        e1+= numpy.einsum('ij,ji', h1b, dm1[1])
        e1+= numpy.einsum('ijkl,ijkl', eriaa, dm2[0]) * .5
        e1+= numpy.einsum('ijkl,ijkl', eriab, dm2[1])
        e1+= numpy.einsum('ijkl,ijkl', eribb, dm2[2]) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

        vhf = mf.get_veff(mol, mf.make_rdm1())
        h1a = reduce(numpy.dot, (mo_a.T, hcore+vhf[0], mo_a))
        h1b = reduce(numpy.dot, (mo_b.T, hcore+vhf[1], mo_b))
        dm1[0][numpy.diag_indices(nocca)] -= 1
        dm1[1][numpy.diag_indices(noccb)] -= 1
        e = numpy.einsum('pq,qp', h1a, dm1[0])
        e+= numpy.einsum('pq,qp', h1b, dm1[1])
        self.assertAlmostEqual(e, -emp2, 8)

    def test_ump2_contract_eri_dm(self):
        pt = mp.MP2(mf)
        pt.frozen = [[0,1,2,3],[1]]
        emp2, t2 = pt.kernel()
        mo_a, mo_b = mf.mo_coeff
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        dm1a,dm1b = pt.make_rdm1()
        dm2aa,dm2ab,dm2bb = pt.make_rdm2()
        eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
        eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
        eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
        eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
        hcore = mf.get_hcore()
        h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
        e1 = numpy.einsum('ij,ji', h1a, dm1a)
        e1+= numpy.einsum('ij,ji', h1b, dm1b)
        e1+= numpy.einsum('ijkl,ijkl', eriaa, dm2aa) * .5
        e1+= numpy.einsum('ijkl,ijkl', eriab, dm2ab)
        e1+= numpy.einsum('ijkl,ijkl', eribb, dm2bb) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

    def test_ump2_frozen(self):
        pt = mp.MP2(mf)
        pt.frozen = [1]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.11202141654451162, 8)

    def test_ump2_outcore_frozen(self):
        pt = mp.MP2(mf)
        pt.max_memory = 0
        pt.nmo = (12, 11)
        pt.frozen = [[4,5],[2,3]]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.033400699456971966, 8)

        pt = mp.MP2(mf)
        pt.nmo = (12, 11)
        pt.nocc = (4, 2)
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.033400699456971966, 8)

    def test_ump2_with_df(self):
        pt = mp.ump2.UMP2(mf.density_fit('weigend'))
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.11264162733420097, 8)

        pt = mp.dfump2.DFUMP2(mf.density_fit('weigend'))
        pt.frozen = [1]
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.11264162733420097, 8)

        pt = mp.dfump2.DFUMP2(mf)
        pt.frozen = [1]
        pt.with_df = mf.density_fit('weigend').with_df
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.11264162733420097, 8)

    def test_ump2_ao2mo_ovov(self):
        pt = mp.UMP2(mf)
        pt.frozen = 0
        nocca, noccb = mol.nelec
        orboa = mf.mo_coeff[0][:,:nocca]
        orbva = mf.mo_coeff[0][:,nocca:]
        orbob = mf.mo_coeff[1][:,:noccb]
        orbvb = mf.mo_coeff[1][:,noccb:]
        orbs = (orboa, orbva, orbob, orbvb)
        ftmp = lib.H5TmpFile()
        mp.ump2._ao2mo_ovov(pt, orbs, ftmp, 1)
        ovov = numpy.asarray(ftmp['ovov'])
        ovOV = numpy.asarray(ftmp['ovOV'])
        OVOV = numpy.asarray(ftmp['OVOV'])
        ovov_ref = ao2mo.general(mf._eri, (orboa,orbva,orboa,orbva))
        ovOV_ref = ao2mo.general(mf._eri, (orboa,orbva,orbob,orbvb))
        OVOV_ref = ao2mo.general(mf._eri, (orbob,orbvb,orbob,orbvb))
        self.assertAlmostEqual(numpy.linalg.norm(ovov_ref-ovov), 0, 8)
        self.assertAlmostEqual(numpy.linalg.norm(ovOV_ref-ovOV), 0, 8)
        self.assertAlmostEqual(numpy.linalg.norm(OVOV_ref-OVOV), 0, 8)

    def test_ump2_with_ao2mofn(self):
        pt = mp.ump2.UMP2(mf)
        mf_df = mf.density_fit('weigend')
        ao2mofn = mf_df.with_df.ao2mo
        pt.ao2mo = lambda *args: mp.ump2._make_eris(pt, *args, ao2mofn=ao2mofn)
        e1 = pt.kernel()[0]
        self.assertAlmostEqual(e1, -0.16607937629805458, 8)
        pt = mp.ump2.UMP2(mf.density_fit('weigend'))
        e2 = pt.kernel()[0]
        self.assertAlmostEqual(e1, e2, 8)

    def test_rdm_complex(self):
        mol = gto.M()
        mol.verbose = 0
        nocca,noccb = 3,2
        nvira,nvirb = 4,5
        mf = scf.UHF(mol)
        nmo = nocca + nvira
        numpy.random.seed(1)
        eri_aa = (numpy.random.random((nmo,nmo,nmo,nmo)) +
                  numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri_aa = eri_aa + eri_aa.transpose(1,0,3,2).conj()
        eri_aa = eri_aa + eri_aa.transpose(2,3,0,1)
        eri_aa *= .1
        eri_bb = (numpy.random.random((nmo,nmo,nmo,nmo)) +
                  numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri_bb = eri_bb + eri_bb.transpose(1,0,3,2).conj()
        eri_bb = eri_bb + eri_bb.transpose(2,3,0,1)
        eri_bb *= .1
        eri_ab = (numpy.random.random((nmo,nmo,nmo,nmo)) +
                  numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri_ab = eri_ab + eri_ab.transpose(1,0,3,2).conj()
        eri_ab *= .1

        eris = lambda: None
        eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].reshape(nocca*nvira,nocca*nvira)
        eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].reshape(noccb*nvirb,noccb*nvirb)
        eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].reshape(nocca*nvira,noccb*nvirb)

        mo_energy = [numpy.arange(nmo), numpy.arange(nmo)+.1]
        mo_occ = numpy.zeros((2,nmo))
        mo_occ[0,:nocca] = 1
        mo_occ[1,:noccb] = 1
        mf.make_rdm1 = lambda *args: [numpy.diag(mo_occ[0]), numpy.diag(mo_occ[1])]
        dm = mf.make_rdm1()
        vja = numpy.einsum('ijkl,lk->ij', eri_aa, dm[0])
        vja+= numpy.einsum('ijkl,lk->ij', eri_ab, dm[1])
        vjb = numpy.einsum('ijkl,lk->ij', eri_bb, dm[1])
        vjb+= numpy.einsum('klij,lk->ij', eri_ab, dm[0])
        vka = numpy.einsum('ijkl,jk->il', eri_aa, dm[0])
        vkb = numpy.einsum('ijkl,jk->il', eri_bb, dm[1])
        mf.get_veff = lambda *args: (vja - vka, vjb - vkb)
        vhf = mf.get_veff()
        hcore = (numpy.diag(mo_energy[0]) - vhf[0],
                 numpy.diag(mo_energy[1]) - vhf[1])
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        eris.mo_energy = mf.mo_energy = mo_energy
        mf.mo_coeff = [numpy.eye(nmo)]*2
        mf.mo_occ = mo_occ
        mf.e_tot = numpy.einsum('ij,ji', hcore[0], dm[0])
        mf.e_tot+= numpy.einsum('ij,ji', hcore[1], dm[1])
        mf.e_tot+= numpy.einsum('ij,ji', vhf[0], dm[0]) * .5
        mf.e_tot+= numpy.einsum('ij,ji', vhf[1], dm[1]) * .5
        mf.converged = True
        pt = mp.MP2(mf)
        pt.ao2mo = lambda *args, **kwargs: eris
        pt.kernel(eris=eris)
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()

        e1 = numpy.einsum('ij,ji', hcore[0], dm1[0])
        e1+= numpy.einsum('ij,ji', hcore[1], dm1[1])
        e1+= numpy.einsum('ijkl,ijkl', eri_aa, dm2[0]) * .5
        e1+= numpy.einsum('ijkl,ijkl', eri_ab, dm2[1])
        e1+= numpy.einsum('ijkl,ijkl', eri_bb, dm2[2]) * .5
        self.assertAlmostEqual(e1, pt.e_tot, 12)

        self.assertAlmostEqual(abs(dm2[0]-dm2[0].transpose(1,0,3,2).conj()).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2[0]-dm2[0].transpose(2,3,0,1)       ).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2[1]-dm2[1].transpose(1,0,3,2).conj()).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2[2]-dm2[2].transpose(1,0,3,2).conj()).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2[2]-dm2[2].transpose(2,3,0,1)       ).max(), 0, 8)

    def test_non_canonical_mp2(self):
        mf = scf.UHF(mol).run(max_cycle=1)
        pt = mp.MP2(mf)
        self.assertAlmostEqual(pt.kernel()[0], -0.1707921460057042, 7)



if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()
