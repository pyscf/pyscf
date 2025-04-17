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
    global mol, mf, mf1
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_mp2(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)

        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2 - t2ref0).max(), 0, 8)

        pt.max_memory = 1
        pt.frozen = []
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.204019967288338, 8)
        self.assertAlmostEqual(pt.e_corr_ss, -0.05153088565639835, 8)
        self.assertAlmostEqual(pt.e_corr_os, -0.15248908163191538, 8)
        self.assertAlmostEqual(abs(t2 - t2ref0).max(), 0, 8)

    def test_mp2_outcore(self):
        pt = mp.mp2.MP2(mf)
        pt.max_memory = .01
        e, t2 = pt.kernel()
        self.assertAlmostEqual(e, -0.20401996728747132, 8)
        self.assertAlmostEqual(numpy.linalg.norm(t2), 0.19379397642098622, 8)

    def test_mp2_dm(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        co = mf.mo_coeff[:,:nocc]
        cv = mf.mo_coeff[:,nocc:]
        g = ao2mo.incore.general(mf._eri, (co,cv,co,cv)).ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,3,1)

        pt = mp.mp2.MP2(mf)
        emp2, t2 = pt.kernel()

        t2s = numpy.zeros((nocc*2,nocc*2,nvir*2,nvir*2))
        t2s[ ::2, ::2, ::2, ::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
        t2s[1::2,1::2,1::2,1::2] = t2ref0 - t2ref0.transpose(0,1,3,2)
        t2s[ ::2,1::2,1::2, ::2] = t2ref0
        t2s[1::2, ::2, ::2,1::2] = t2ref0
        t2s[ ::2,1::2, ::2,1::2] = -t2ref0.transpose(0,1,3,2)
        t2s[1::2, ::2,1::2, ::2] = -t2ref0.transpose(0,1,3,2)
        dm1occ =-.5 * numpy.einsum('ikab,jkab->ij', t2s, t2s)
        dm1vir = .5 * numpy.einsum('ijac,ijbc->ab', t2s, t2s)
        dm1ref = numpy.zeros((nmo,nmo))
        dm1ref[:nocc,:nocc] = dm1occ[ ::2, ::2]+dm1occ[1::2,1::2]
        dm1ref[nocc:,nocc:] = dm1vir[ ::2, ::2]+dm1vir[1::2,1::2]
        for i in range(nocc):
            dm1ref[i,i] += 2
        dm1refao = reduce(numpy.dot, (mf.mo_coeff, dm1ref, mf.mo_coeff.T))
        rdm1 = mp.mp2.make_rdm1(pt, t2ref0, ao_repr=True)
        self.assertTrue(numpy.allclose(rdm1, dm1refao))
        self.assertTrue(numpy.allclose(pt.make_rdm1(), dm1ref))
        rdm1 = mp.mp2.make_rdm1(pt, ao_repr=True)
        self.assertTrue(numpy.allclose(rdm1, dm1refao))

        dm2ref = numpy.zeros((nmo*2,)*4)
        dm2ref[:nocc*2,nocc*2:,:nocc*2,nocc*2:] = t2s.transpose(0,3,1,2) * .5
        dm2ref[nocc*2:,:nocc*2,nocc*2:,:nocc*2] = t2s.transpose(3,0,2,1) * .5
        dm2ref = dm2ref[ ::2, ::2, ::2, ::2] + dm2ref[1::2,1::2,1::2,1::2] \
               + dm2ref[ ::2, ::2,1::2,1::2] + dm2ref[1::2,1::2, ::2, ::2]
        eris = ao2mo.restore(1, ao2mo.full(mf._eri, mf.mo_coeff), mf.mo_coeff.shape[1])
        self.assertAlmostEqual(numpy.einsum('iajb,iajb', eris, dm2ref)*.5, emp2, 8)

    def test_mp2_contract_eri_dm(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size

        pt = mp.mp2.MP2(mf)
        pt.frozen = 0
        emp2, t2 = pt.kernel()
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
        hcore = mf.get_hcore()
        rdm1 = pt.make_rdm1()
        rdm2 = pt.make_rdm2()
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, hcore, mf.mo_coeff))
        e1 = numpy.einsum('ij,ji', h1, rdm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, rdm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

        pt.frozen = 2
        pt.max_memory = 1
        emp2, t2 = pt.kernel(with_t2=False)
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
        hcore = mf.get_hcore()
        rdm1 = pt.make_rdm1()
        rdm2 = pt.make_rdm2()
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, hcore, mf.mo_coeff))
        e1 = numpy.einsum('ij,ji', h1, rdm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, rdm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

    def test_mp2_with_df(self):
        nocc = mol.nelectron//2
        nmo = mf.mo_energy.size
        nvir = nmo - nocc

        mf_df = mf.density_fit('weigend')
        pt = mp.dfmp2.DFMP2(mf_df)
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        eris = mp.mp2._make_eris(pt, mo_coeff=mf.mo_coeff, ao2mofn=mf_df.with_df.ao2mo)
        g = eris.ovov.ravel()
        eia = mf.mo_energy[:nocc,None] - mf.mo_energy[nocc:]
        t2ref0 = g/(eia.reshape(-1,1)+eia.reshape(-1)).ravel()
        t2ref0 = t2ref0.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        e, t2 = pt.kernel(mf.mo_energy, mf.mo_coeff)
        self.assertAlmostEqual(e, -0.20425449198334983, 8)
        self.assertAlmostEqual(abs(t2 - t2ref0).max(), 0, 8)

        pt = mp.MP2(mf.density_fit('weigend'))
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp.dfmp2.DFMP2(mf.density_fit('weigend'))
        e = pt.kernel(mf.mo_energy, mf.mo_coeff)[0]
        self.assertAlmostEqual(e, -0.20425449198334983, 8)

        pt.frozen = [1]
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)

        pt = mp.dfmp2.DFMP2(mf)
        pt.frozen = [1]
        pt.with_df = mf.density_fit('weigend').with_df
        e = pt.kernel()[0]
        self.assertAlmostEqual(e, -0.14708846352674113, 8)


    def test_mp2_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [0]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)
        pt.set_frozen()
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.20168270592254167, 8)

    def test_mp2_outcore_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.max_memory = 0
        pt.nmo = 12
        pt.frozen = [4]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.080686724178583275, 8)

        pt = mp.mp2.MP2(mf)
        pt.nmo = 12
        pt.nocc = 4
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.080686724178583275, 8)

    def test_mp2_ao2mo_ovov(self):
        pt = mp.mp2.MP2(mf)
        orbo = mf.mo_coeff[:,:8]
        orbv = mf.mo_coeff[:,8:]
        ftmp = lib.H5TmpFile()
        h5dat = mp.mp2._ao2mo_ovov(pt, orbo, orbv, ftmp, 1)
        ovov = numpy.asarray(h5dat)
        ovov_ref = ao2mo.general(mf._eri, (orbo,orbv,orbo,orbv))
        self.assertAlmostEqual(numpy.linalg.norm(ovov_ref-ovov), 0, 8)

    def test_mp2_with_ao2mofn(self):
        pt = mp.mp2.MP2(mf)
        mf_df = mf.density_fit('weigend')
        ao2mofn = mf_df.with_df.ao2mo
        pt.ao2mo = lambda *args: mp.mp2._make_eris(pt, *args, ao2mofn=ao2mofn)
        e1 = pt.kernel()[0]
        pt = mp.mp2.MP2(mf.density_fit('weigend'))
        e2 = pt.kernel()[0]
        self.assertAlmostEqual(e1, -0.20425449198652196, 8)
        self.assertAlmostEqual(e1, e2, 8)

    def test_rdm_complex(self):
        mol = gto.M()
        mol.verbose = 0
        nocc = 3
        nvir = 4
        mf = scf.RHF(mol)
        nmo = nocc + nvir
        numpy.random.seed(1)
        eri = (numpy.random.random((nmo,nmo,nmo,nmo)) +
               numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri = eri + eri.transpose(1,0,3,2).conj()
        eri = eri + eri.transpose(2,3,0,1)
        eri *= .1

        eris = lambda: None
        eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].reshape(nocc*nvir,nocc*nvir)

        mo_energy = numpy.arange(nmo)
        mo_occ = numpy.zeros(nmo)
        mo_occ[:nocc] = 2
        mf.make_rdm1 = lambda *args: numpy.diag(mo_occ)
        dm = mf.make_rdm1()
        mf.get_veff = lambda *args: numpy.einsum('ijkl,lk->ij', eri, dm) - numpy.einsum('ijkl,jk->il', eri, dm) * .5
        vhf = mf.get_veff()
        hcore = numpy.diag(mo_energy) - vhf
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        eris.mo_energy = mf.mo_energy = mo_energy
        mf.mo_coeff = numpy.eye(nmo)
        mf.mo_occ = mo_occ
        mf.e_tot = numpy.einsum('ij,ji', hcore, dm) + numpy.einsum('ij,ji', vhf, dm) *.5
        mf.converged = True
        pt = mp.MP2(mf)
        pt.ao2mo = lambda *args, **kwargs: eris
        pt.kernel(eris=eris)
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()

        e1 = numpy.einsum('ij,ji', hcore, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        self.assertAlmostEqual(e1, pt.e_tot, 12)

        #self.assertAlmostEqual(abs(numpy.einsum('ijkk->ji', dm2)/(nocc*2-1) - dm1).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2).conj()).max(), 0, 8)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)       ).max(), 0, 8)

    def test_init_mp2(self):
        mf0 = mf
        mf1 = scf.RHF(gto.M(atom='H', spin=1))
        self.assertTrue(isinstance(mp.MP2(mf0), mp.mp2.RMP2))
        self.assertTrue(isinstance(mp.MP2(mf1), mp.ump2.UMP2))
        self.assertTrue(isinstance(mp.MP2(mf0.density_fit()), mp.dfmp2.DFMP2))
        #self.assertTrue(isinstance(mp.MP2(mf1.density_fit()), mp.dfmp2.DFUMP2))
        self.assertTrue(isinstance(mp.MP2(mf0.newton()), mp.mp2.RMP2))
        self.assertTrue(isinstance(mp.MP2(mf1.newton()), mp.ump2.UMP2))

    def test_mp2_scanner(self):
        pt_scanner = mp.MP2(mf).as_scanner()
        e = pt_scanner(mol)
        self.assertAlmostEqual(e, mf.e_tot-0.204019967288338, 8)

    def test_reset(self):
        mol1 = gto.M(atom='C')
        pt = scf.RHF(mol).run().DFMP2()
        pt.reset(mol1)
        self.assertTrue(pt.mol is mol1)
        self.assertTrue(pt.with_df.mol is mol1)

    def test_non_canonical_mp2(self):
        mf = scf.RHF(mol).run(max_cycle=1)
        pt = mp.MP2(mf)
        self.assertAlmostEqual(pt.kernel()[0], -0.20447991367138338, 7)


if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()
