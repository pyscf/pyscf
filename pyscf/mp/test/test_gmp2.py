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
    global mol, mf, gmf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.scf()
    gmf = scf.GHF(mol)
    gmf.conv_tol = 1e-12
    gmf.scf()

def tearDownModule():
    global mol, mf, gmf
    mol.stdout.close()
    del mol, mf, gmf


class KnownValues(unittest.TestCase):
    def test_gmp2(self):
        pt = mp.GMP2(gmf)
        emp2, t2 = pt.kernel(gmf.mo_energy, gmf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.12886859466191491, 8)

        pt.max_memory = 1
        pt.frozen = None
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.12886859466191491, 8)

        mf1 = scf.addons.convert_to_ghf(mf)
        mf1.mo_coeff = numpy.asarray(mf1.mo_coeff)  # remove tag orbspin
        pt = mp.GMP2(mf1)
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.09625784206542846, 8)

        pt.max_memory = 1
        pt.frozen = None
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.09625784206542846, 8)

    def test_gmp2_contract_eri_dm(self):
        pt = mp.GMP2(mf)
        pt.frozen = 2
        emp2, t2 = pt.kernel()
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()

        nao = mol.nao_nr()
        mo_a = pt._scf.mo_coeff[:nao]
        mo_b = pt._scf.mo_coeff[nao:]
        nmo = mo_a.shape[1]
        eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
        orbspin = pt._scf.mo_coeff.orbspin
        sym_forbid = (orbspin[:,None] != orbspin)
        eri[sym_forbid,:,:] = 0
        eri[:,:,sym_forbid] = 0
        hcore = mf.get_hcore()
        h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
        h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))

        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

        pt = mp.GMP2(mf)
        emp2, t2 = pt.kernel()
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()
        #self.assertAlmostEqual(abs(numpy.einsum('ijkk->ji', dm2)/9 - dm1).max(), 0, 8)
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, pt.e_tot, 8)

        hcore = pt._scf.get_hcore()
        mo = pt._scf.mo_coeff
        vhf = pt._scf.get_veff(mol, pt._scf.make_rdm1())
        h1 = reduce(numpy.dot, (mo.T, hcore+vhf, mo))
        dm1[numpy.diag_indices(mol.nelectron)] -= 1
        e = numpy.einsum('pq,qp', h1, dm1)
        self.assertAlmostEqual(e, -emp2, 8)

    def test_gmp2_frozen(self):
        pt = mp.GMP2(gmf)
        pt.frozen = [0, 1]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.12783149583822068, 8)
        pt.set_frozen()
        pt.kernel(with_t2=False)
        self.assertEqual(pt.frozen, 2)
        self.assertAlmostEqual(pt.emp2, -0.12783149583822068, 8)

    def test_gmp2_outcore_frozen(self):
        pt = mp.GMP2(gmf)
        pt.max_memory = 0
        pt.nmo = 24
        pt.frozen = [8,9]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.098239933985213371, 8)

        pt = mp.GMP2(gmf)
        pt.nmo = 24
        pt.nocc = 8
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.098239933985213371, 8)

    def test_gmp2_with_ao2mofn(self):
        pt = mp.GMP2(gmf)
        mf_df = mf.density_fit('weigend')
        def ao2mofn(mo_coeffs):
            nao = mo_coeffs[0].shape[0]
            mo_a = [mo[:nao//2] for mo in mo_coeffs]
            mo_b = [mo[nao//2:] for mo in mo_coeffs]
            eri  = mf_df.with_df.ao2mo(mo_a)
            eri += mf_df.with_df.ao2mo(mo_b)
            eri += mf_df.with_df.ao2mo([mo_a[0], mo_a[1], mo_b[2], mo_b[3]])
            eri += mf_df.with_df.ao2mo([mo_b[0], mo_b[1], mo_a[2], mo_a[3]])
            return eri
        pt.ao2mo = lambda *args: mp.gmp2._make_eris_incore(pt, *args, ao2mofn=ao2mofn)
        e1 = pt.kernel()[0]
        self.assertAlmostEqual(e1, -0.12879040729543023, 7)
        # Should be quite close to emp2 without DF
        self.assertAlmostEqual(e1, -0.12886859466191491, 3)

#        pt = mp.GMP2(gmf.density_fit('weigend'))
#        e2 = pt.kernel()[0]
#        self.assertAlmostEqual(e1, e2, 8)

    def test_rdm_complex(self):
        mol = gto.M()
        mol.verbose = 0
        nocc = 6
        nvir = 8
        mf = scf.GHF(mol)
        nmo = nocc + nvir
        numpy.random.seed(1)
        eri = (numpy.random.random((nmo,nmo,nmo,nmo)) +
               numpy.random.random((nmo,nmo,nmo,nmo))* 1j - (.5+.5j))
        eri = eri + eri.transpose(1,0,3,2).conj()
        eri = eri + eri.transpose(2,3,0,1)
        eri *= .1
        eri0 = eri
        erip = eri - eri.transpose(0,3,2,1)

        eris = lambda: None
        eris.oovv = erip[:nocc,nocc:,:nocc,nocc:].transpose(0,2,1,3)

        mo_energy = numpy.arange(nmo)
        mo_occ = numpy.zeros(nmo)
        mo_occ[:nocc] = 1
        mf.make_rdm1 = lambda *args: numpy.diag(mo_occ)
        dm = mf.make_rdm1()
        mf.get_veff = lambda *args: numpy.einsum('ijkl,lk->ij', erip, dm)
        vhf = mf.get_veff()
        hcore = numpy.diag(mo_energy) - vhf
        mf.get_hcore = lambda *args: hcore
        mf.get_ovlp = lambda *args: numpy.eye(nmo)
        eris.mo_energy = mf.mo_energy = mo_energy
        mf.mo_coeff = numpy.eye(nmo)
        mf.mo_occ = mo_occ
        mf.e_tot = numpy.einsum('ij,ji', hcore, dm) + numpy.einsum('ij,ji', vhf, dm) *.5
        mf.converged = True
        pt = mp.GMP2(mf)
        pt.ao2mo = lambda *args, **kwargs: eris
        pt.kernel(eris=eris)
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()

        e1 = numpy.einsum('ij,ji', hcore, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri0, dm2) * .5
        self.assertAlmostEqual(e1, pt.e_tot, 12)

        #self.assertAlmostEqual(abs(numpy.einsum('ijkk->ji', dm2)/(nocc-1) - dm1).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2).conj()).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(2,1,0,3)       ).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2+dm2.transpose(0,3,2,1)       ).max(), 0, 9)

    def test_non_canonical_mp2(self):
        mf = scf.GHF(mol).run(max_cycle=1)
        pt = mp.MP2(mf)
        self.assertAlmostEqual(pt.kernel()[0], -0.12714840392411947, 7)

    def test_gmp2_with_df(self):
        pt = mp.GMP2(gmf).density_fit()
        emp2, t2 = pt.kernel(gmf.mo_energy, gmf.mo_coeff)
        self.assertAlmostEqual(emp2, -0.12884823204824902, 8)

        mf1 = scf.addons.convert_to_ghf(mf)
        mf1.mo_coeff = numpy.asarray(mf1.mo_coeff)  # remove tag orbspin
        pt = mp.GMP2(mf1).density_fit()
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.09624851692896723, 8)

        dm = gmf.get_init_guess() + .1j
        dm = 0.5*(dm + dm.T.conj())
        gmf.conv_tol = 1e-9
        gmf.kernel(dm0=dm)
        pt = mp.GMP2(gmf).density_fit()
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.12884823204824902, 8)


if __name__ == "__main__":
    print("Full Tests for gmp2")
    unittest.main()
