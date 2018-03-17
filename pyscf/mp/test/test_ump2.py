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
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mp

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
mf.conv_tol = 1e-14
ehf = mf.scf()


class KnownValues(unittest.TestCase):
    def test_ump2(self):
        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.16575150552336643, 9)

        pt.max_memory = 1
        pt.frozen = None
        emp2, t2 = pt.kernel()
        self.assertAlmostEqual(emp2, -0.16575150552336643, 9)

    def test_ump2_dm(self):
        pt = mp.MP2(mf)
        emp2, t2 = pt.kernel()
        dm1 = pt.make_rdm1()
        dm2 = pt.make_rdm2()
        gpt = mp.GMP2(mf).run()
        dm1ref = gpt.make_rdm1()
        ia = gpt._scf.mo_coeff.orbspin == 0
        ib = gpt._scf.mo_coeff.orbspin == 1
        mo_a, mo_b = mf.mo_coeff
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
        nocca, noccb = mol.nelec

        self.assertTrue(numpy.allclose(dm1[0], dm1ref[ia][:,ia]))
        self.assertTrue(numpy.allclose(dm1[1], dm1ref[ib][:,ib]))
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
        self.assertAlmostEqual(e1, pt.e_tot, 9)

        vhf = mf.get_veff(mol, mf.make_rdm1())
        h1a = reduce(numpy.dot, (mo_a.T, hcore+vhf[0], mo_a))
        h1b = reduce(numpy.dot, (mo_b.T, hcore+vhf[1], mo_b))
        dm1[0][numpy.diag_indices(nocca)] -= 1
        dm1[1][numpy.diag_indices(noccb)] -= 1
        e = numpy.einsum('pq,qp', h1a, dm1[0])
        e+= numpy.einsum('pq,qp', h1b, dm1[1])
        self.assertAlmostEqual(e, -emp2, 9)

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
        self.assertAlmostEqual(e1, pt.e_tot, 9)

    def test_ump2_frozen(self):
        pt = mp.MP2(mf)
        pt.frozen = [1]
        pt.kernel(with_t2=False)
        self.assertAlmostEqual(pt.emp2, -0.11202141654451162, 9)

    def test_ump2_outcore_frozen(self):
        pt = mp.MP2(mf)
        pt.max_memory = 0
        pt.nmo = (12, 11)
        pt.frozen = [[4,5],[2,3]]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.033400699456971966, 9)

        pt = mp.MP2(mf)
        pt.nmo = (12, 11)
        pt.nocc = (4, 2)
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.033400699456971966, 9)

    def test_ump2_with_df(self):
        pt = mp.ump2.UMP2(mf.density_fit('weigend'))
        pt.frozen = [1]
        e = pt.kernel(with_t2=False)[0]
        self.assertAlmostEqual(e, -0.11264162733420097, 9)

        #pt = mp.dfump2.DFUMP2(mf.density_fit('weigend'))
        #pt.frozen = [1]
        #e = pt.kernel()[0]
        #self.assertAlmostEqual(e, -0.11264162733420097, 9)

        #pt = mp.dfump2.DFUMP2(mf)
        #pt.frozen = [1]
        #pt.with_df = mf.density_fit('weigend').with_df
        #e = pt.kernel()[0]
        #self.assertAlmostEqual(e, -0.11264162733420097, 9)

    def test_ump2_ao2mo_ovov(self):
        pt = mp.UMP2(mf)
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
        self.assertAlmostEqual(numpy.linalg.norm(ovov_ref-ovov), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ovOV_ref-ovOV), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(OVOV_ref-OVOV), 0, 9)

    def test_ump2_with_ao2mofn(self):
        pt = mp.ump2.UMP2(mf)
        mf_df = mf.density_fit('weigend')
        ao2mofn = mf_df.with_df.ao2mo
        pt.ao2mo = lambda *args: mp.ump2._make_eris(pt, *args, ao2mofn=ao2mofn)
        e1 = pt.kernel()[0]
        pt = mp.ump2.UMP2(mf.density_fit('weigend'))
        e2 = pt.kernel()[0]
        self.assertAlmostEqual(e1, e2, 9)



if __name__ == "__main__":
    print("Full Tests for mp2")
    unittest.main()

