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
import copy
import numpy
import h5py
from functools import reduce

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import mp
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.cc import addons
from pyscf.cc import uccsd_rdm
from pyscf.fci import direct_uhf

def setUpModule():
    global mol, rhf, mf, myucc, mol_s2, mf_s2, eris
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol_grad = 1e-8
    rhf.kernel()
    mf = scf.addons.convert_to_uhf(rhf)

    myucc = cc.UCCSD(mf).run(conv_tol=1e-10)

    mol_s2 = gto.Mole()
    mol_s2.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol_s2.basis = '631g'
    mol_s2.spin = 2
    mol_s2.verbose = 5
    mol_s2.output = '/dev/null'
    mol_s2.build()
    mf_s2 = scf.UHF(mol_s2).run()
    eris = uccsd.UCCSD(mf_s2).ao2mo()

def tearDownModule():
    global mol, rhf, mf, myucc, mol_s2, mf_s2, eris
    mol.stdout.close()
    mol_s2.stdout.close()
    del mol, rhf, mf, myucc, mol_s2, mf_s2, eris

class KnownValues(unittest.TestCase):

    def test_with_df_s0(self):
        mf = scf.UHF(mol).density_fit(auxbasis='weigend').run()
        mycc = cc.UCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -76.118403942938741, 6)

    def test_with_df_s2(self):
        mf = scf.UHF(mol_s2).density_fit(auxbasis='weigend').run()
        mycc = cc.UCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -75.83360033370676, 6)

    def test_ERIS(self):
        ucc1 = cc.UCCSD(mf)
        nao,nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2,nao,nmo))
        eris = cc.uccsd._make_eris_incore(ucc1, mo_coeff)

        self.assertAlmostEqual(lib.fp(eris.oooo), 4.9638849382825754, 11)
        self.assertAlmostEqual(lib.fp(eris.ovoo),-1.3623681896983584, 11)
        self.assertAlmostEqual(lib.fp(eris.ovov), 125.81550684442163, 11)
        self.assertAlmostEqual(lib.fp(eris.oovv), 55.123681017639598, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvo), 133.48083527898248, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvv), 59.421927525288183, 11)
        self.assertAlmostEqual(lib.fp(eris.vvvv), 43.556602622204778, 11)
        self.assertAlmostEqual(lib.fp(eris.OOOO),-407.05319440524585, 11)
        self.assertAlmostEqual(lib.fp(eris.OVOO), 56.284299937160796, 11)
        self.assertAlmostEqual(lib.fp(eris.OVOV),-287.72899895597448, 11)
        self.assertAlmostEqual(lib.fp(eris.OOVV),-85.484299959144522, 11)
        self.assertAlmostEqual(lib.fp(eris.OVVO),-228.18996145476956, 11)
        self.assertAlmostEqual(lib.fp(eris.OVVV),-10.715902258877399, 11)
        self.assertAlmostEqual(lib.fp(eris.VVVV),-89.908425473958303, 11)
        self.assertAlmostEqual(lib.fp(eris.ooOO),-336.65979260175226, 11)
        self.assertAlmostEqual(lib.fp(eris.ovOO),-16.405125847288176, 11)
        self.assertAlmostEqual(lib.fp(eris.ovOV), 231.59042209500075, 11)
        self.assertAlmostEqual(lib.fp(eris.ooVV), 20.338077193028354, 11)
        self.assertAlmostEqual(lib.fp(eris.ovVO), 206.48662856981386, 11)
        self.assertAlmostEqual(lib.fp(eris.ovVV),-71.273249852220516, 11)
        self.assertAlmostEqual(lib.fp(eris.vvVV), 172.47130671068496, 11)
        self.assertAlmostEqual(lib.fp(eris.OVoo),-19.927660309103977, 11)
        self.assertAlmostEqual(lib.fp(eris.OOvv),-27.761433381797019, 11)
        self.assertAlmostEqual(lib.fp(eris.OVvo),-140.09648311337384, 11)
        self.assertAlmostEqual(lib.fp(eris.OVvv), 40.700983950220547, 11)

        uccsd.MEMORYMIN, bak = 0, uccsd.MEMORYMIN
        ucc1.max_memory = 0
        eris1 = ucc1.ao2mo(mo_coeff)
        uccsd.MEMORYMIN = bak
        self.assertAlmostEqual(abs(numpy.array(eris1.oooo)-eris.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovoo)-eris.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovov)-eris.ovov).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.oovv)-eris.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovvo)-eris.ovvo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovvv)-eris.ovvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.vvvv)-eris.vvvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OOOO)-eris.OOOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVOO)-eris.OVOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVOV)-eris.OVOV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OOVV)-eris.OOVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVVO)-eris.OVVO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVVV)-eris.OVVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.VVVV)-eris.VVVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ooOO)-eris.ooOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovOO)-eris.ovOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovOV)-eris.ovOV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ooVV)-eris.ooVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovVO)-eris.ovVO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovVV)-eris.ovVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.vvVV)-eris.vvVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVoo)-eris.OVoo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OOvv)-eris.OOvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVvo)-eris.OVvo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.OVvv)-eris.OVvv).max(), 0, 11)

        # Testing the complex MO integrals
        def ao2mofn(mos):
            if isinstance(mos, numpy.ndarray) and mos.ndim == 2:
                mos = [mos]*4
            nmos = [mo.shape[1] for mo in mos]
            eri_mo = ao2mo.kernel(mf._eri, mos, compact=False).reshape(nmos)
            return eri_mo * 1j
        eris1 = cc.uccsd._make_eris_incore(ucc1, mo_coeff, ao2mofn=ao2mofn)
        self.assertAlmostEqual(abs(eris1.oooo.imag-eris.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovoo.imag-eris.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovov.imag-eris.ovov).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.oovv.imag-eris.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovvo.imag-eris.ovvo).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.ovvv.imag-eris.ovvv).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.vvvv.imag-eris.vvvv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OOOO.imag-eris.OOOO).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OVOO.imag-eris.OVOO).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OVOV.imag-eris.OVOV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OOVV.imag-eris.OOVV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OVVO.imag-eris.OVVO).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.OVVV.imag-eris.OVVV).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.VVVV.imag-eris.VVVV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ooOO.imag-eris.ooOO).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovOO.imag-eris.ovOO).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovOV.imag-eris.ovOV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ooVV.imag-eris.ooVV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovVO.imag-eris.ovVO).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.ovVV.imag-eris.ovVV).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.vvVV.imag-eris.vvVV).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OVoo.imag-eris.OVoo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OOvv.imag-eris.OOvv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.OVvo.imag-eris.OVvo).max(), 0, 11)
        #self.assertAlmostEqual(abs(eris1.OVvv.imag-eris.OVvv).max(), 0, 11)

    def test_amplitudes_from_rccsd(self):
        e, t1, t2 = cc.RCCSD(rhf).set(conv_tol=1e-10).kernel()
        t1, t2 = myucc.amplitudes_from_rccsd(t1, t2)
        self.assertAlmostEqual(abs(t1[0]-myucc.t1[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(t1[1]-myucc.t1[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(t2[0]-myucc.t2[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(t2[1]-myucc.t2[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(t2[2]-myucc.t2[2]).max(), 0, 5)

    def test_uccsd_frozen(self):
        ucc1 = myucc.copy()
        ucc1.frozen = 1
        self.assertEqual(ucc1.nmo, (12,12))
        self.assertEqual(ucc1.nocc, (4,4))
        ucc1.set_frozen()
        self.assertEqual(ucc1.nmo, (12,12))
        self.assertEqual(ucc1.nocc, (4,4))
        ucc1.frozen = [0,1]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (3,3))
        ucc1.frozen = [[0,1], [0,1]]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (3,3))
        ucc1.frozen = [1,9]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (4,4))
        ucc1.frozen = [[1,9], [1,9]]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (4,4))
        ucc1.frozen = [9,10,12]
        self.assertEqual(ucc1.nmo, (10,10))
        self.assertEqual(ucc1.nocc, (5,5))
        ucc1.nmo = (13,12)
        ucc1.nocc = (5,4)
        self.assertEqual(ucc1.nmo, (13,12))
        self.assertEqual(ucc1.nocc, (5,4))

    def test_uccsd_frozen(self):
        # Freeze 1s electrons
        frozen = [[0,1], [0,1]]
        ucc = cc.UCCSD(mf_s2, frozen=frozen)
        ucc.diis_start_cycle = 1
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.07414978284611283, 8)

    def test_rdm(self):
        nocc = 5
        nvir = 7
        mol = gto.M()
        mf = scf.UHF(mol)
        mf.mo_occ = numpy.zeros((2,nocc+nvir))
        mf.mo_occ[:,:nocc] = 1
        mycc = uccsd.UCCSD(mf)

        def antisym(t2):
            t2 = t2 - t2.transpose(0,1,3,2)
            t2 = t2 - t2.transpose(1,0,2,3)
            return t2
        orbspin = numpy.zeros((nocc+nvir)*2, dtype=int)
        orbspin[1::2] = 1
        numpy.random.seed(1)
        t1 = numpy.random.random((2,nocc,nvir))*.1 - .1
        t2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
        t2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        t2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        t2 = (t2aa,t2ab,t2bb)
        l1 = numpy.random.random((2,nocc,nvir))*.1 - .1
        l2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
        l2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        l2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
        l2 = (l2aa,l2ab,l2bb)

        dm1a, dm1b = mycc.make_rdm1(t1, t2, l1, l2)
        dm2aa, dm2ab, dm2bb = mycc.make_rdm2(t1, t2, l1, l2)
        ia = orbspin == 0
        ib = orbspin == 1
        oa = orbspin[:nocc*2] == 0
        ob = orbspin[:nocc*2] == 1
        va = orbspin[nocc*2:] == 0
        vb = orbspin[nocc*2:] == 1

        t1 = addons.spatial2spin(t1, orbspin)
        t2 = addons.spatial2spin(t2, orbspin)
        l1 = addons.spatial2spin(l1, orbspin)
        l2 = addons.spatial2spin(l2, orbspin)
        mf1 = scf.GHF(mol)
        mf1.mo_occ = numpy.zeros((nocc+nvir)*2)
        mf.mo_occ[:,:nocc*2] = 1
        mycc1 = gccsd.GCCSD(mf1)
        dm1 = mycc1.make_rdm1(t1, t2, l1, l2)
        dm2 = mycc1.make_rdm2(t1, t2, l1, l2)
        self.assertAlmostEqual(abs(dm1[ia][:,ia]-dm1a).max(), 0, 9)
        self.assertAlmostEqual(abs(dm1[ib][:,ib]-dm1b).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[ia][:,ia][:,:,ia][:,:,:,ia]-dm2aa).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[ia][:,ia][:,:,ib][:,:,:,ib]-dm2ab).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2[ib][:,ib][:,:,ib][:,:,:,ib]-dm2bb).max(), 0, 9)

    def test_h2o_rdm(self):
        mol = mol_s2
        mf = mf_s2
        mycc = uccsd.UCCSD(mf)
        mycc.frozen = 2
        ecc, t1, t2 = mycc.kernel()
        l1, l2 = mycc.solve_lambda()
        dm1a,dm1b = mycc.make_rdm1(t1, t2, l1, l2)
        dm2aa,dm2ab,dm2bb = mycc.make_rdm2(t1, t2, l1, l2)
        mo_a = mf.mo_coeff[0]
        mo_b = mf.mo_coeff[1]
        nmoa = mo_a.shape[1]
        nmob = mo_b.shape[1]
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
        self.assertAlmostEqual(e1, mycc.e_tot, 6)

        d1 = uccsd_rdm._gamma1_intermediates(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2)
        mycc.max_memory = 0
        d2 = uccsd_rdm._gamma2_intermediates(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2, True)
        dm2 = uccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True)
        e1 = numpy.einsum('ij,ji', h1a, dm1a)
        e1+= numpy.einsum('ij,ji', h1b, dm1b)
        e1+= numpy.einsum('ijkl,ijkl', eriaa, dm2[0]) * .5
        e1+= numpy.einsum('ijkl,ijkl', eriab, dm2[1])
        e1+= numpy.einsum('ijkl,ijkl', eribb, dm2[2]) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 6)

    def test_h4_rdm(self):
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
        mol.basis = '6-31g'
        mol.build()
        mf = scf.UHF(mol).set(init_guess='1e').run(conv_tol=1e-14)
        ehf0 = mf.e_tot - mol.energy_nuc()
        mycc = uccsd.UCCSD(mf).run()
        mycc.solve_lambda()
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0])
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1])
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[0], mf.mo_coeff[0],
                                        mf.mo_coeff[1], mf.mo_coeff[1]])
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        efci, fcivec = direct_uhf.kernel((h1a,h1b), (eri_aa,eri_ab,eri_bb),
                                         h1a.shape[0], mol.nelec)
        dm1ref, dm2ref = direct_uhf.make_rdm12s(fcivec, h1a.shape[0], mol.nelec)
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.l1, mycc.l2
        rdm1 = mycc.make_rdm1(t1, t2, l1, l2)
        rdm2 = mycc.make_rdm2(t1, t2, l1, l2)
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[0]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[1]).max(), 0, 5)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[2]).max(), 0, 5)

    def test_eris_contract_vvvv_t2(self):
        mol = gto.Mole()
        nocca, noccb, nvira, nvirb = 5, 4, 12, 13
        nvira_pair = nvira*(nvira+1)//2
        nvirb_pair = nvirb*(nvirb+1)//2
        numpy.random.seed(9)
        t2 = numpy.random.random((nocca,noccb,nvira,nvirb))
        eris = uccsd._ChemistsERIs()
        eris.vvVV = numpy.random.random((nvira_pair,nvirb_pair))
        eris.mol = mol
        myucc.max_memory, bak = 0, myucc.max_memory
        vt2 = eris._contract_vvVV_t2(myucc, t2, eris.vvVV)
        myucc.max_memory = bak
        self.assertAlmostEqual(lib.fp(vt2), 12.00904827896089, 11)
        idxa = lib.square_mat_in_trilu_indices(nvira)
        idxb = lib.square_mat_in_trilu_indices(nvirb)
        vvVV = eris.vvVV[:,idxb][idxa]
        ref = lib.einsum('acbd,ijcd->ijab', vvVV, t2)
        self.assertAlmostEqual(abs(vt2 - ref).max(), 0, 11)

        # _contract_VVVV_t2, testing complex and real mixed contraction
        VVVV =(numpy.random.random((nvirb,nvirb,nvirb,nvirb)) +
               numpy.random.random((nvirb,nvirb,nvirb,nvirb))*1j - (.5+.5j))
        VVVV = VVVV + VVVV.transpose(1,0,3,2).conj()
        VVVV = VVVV + VVVV.transpose(2,3,0,1)
        eris.VVVV = VVVV
        t2 = numpy.random.random((noccb,noccb,nvirb,nvirb))
        t2 = t2 - t2.transpose(0,1,3,2)
        t2 = t2 - t2.transpose(1,0,3,2)
        myucc.max_memory, bak = 0, myucc.max_memory
        vt2 = eris._contract_VVVV_t2(myucc, t2, eris.VVVV)
        myucc.max_memory = bak
        self.assertAlmostEqual(lib.fp(vt2), 47.903883794299404-50.501573400833429j, 11)
        ref = lib.einsum('acbd,ijcd->ijab', eris.VVVV, t2)
        self.assertAlmostEqual(abs(vt2 - ref).max(), 0, 11)

    def test_update_amps1(self):
        mf = scf.UHF(mol_s2)
        numpy.random.seed(9)
        nmo = mf_s2.mo_occ[0].size
        mf.mo_coeff = numpy.random.random((2,nmo,nmo)) - 0.5
        mf.mo_occ = numpy.zeros((2,nmo))
        mf.mo_occ[0,:6] = 1
        mf.mo_occ[1,:5] = 1
        mycc = uccsd.UCCSD(mf)
        nocca, noccb = 6, 5
        nvira, nvirb = nmo-nocca, nmo-noccb
        nvira_pair = nvira*(nvira+1)//2
        nvirb_pair = nvirb*(nvirb+1)//2

        eris = mycc.ao2mo()
        fakeris = uccsd._ChemistsERIs()
        fakeris.mo_coeff = eris.mo_coeff
        fakeris.vvVV = eris.vvVV
        fakeris.mol = mol_s2
        t2ab = numpy.random.random((nocca,noccb,nvira,nvirb))
        t1a = numpy.zeros((nocca,nvira))
        t1b = numpy.zeros((noccb,nvirb))
        self.assertAlmostEqual(lib.fp(mycc._add_vvVV(None, t2ab, fakeris)), 21.652482203108928, 9)
        fakeris.vvVV = None
        mycc.direct = True
        mycc.max_memory = 0
        self.assertAlmostEqual(lib.fp(mycc._add_vvVV(None, t2ab, fakeris)), 21.652482203108928, 9)

        t1 = (numpy.random.random((nocca,nvira)), numpy.random.random((noccb,nvirb)))
        t2 = (numpy.random.random((nocca,nocca,nvira,nvira)),
              numpy.random.random((nocca,noccb,nvira,nvirb)),
              numpy.random.random((noccb,noccb,nvirb,nvirb)))
        t1, t2 = mycc.vector_to_amplitudes(mycc.amplitudes_to_vector(t1, t2))
        t1, t2 = mycc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(lib.fp(t1[0]),  49.912690337392938, 7)
        self.assertAlmostEqual(lib.fp(t1[1]),  74.596097348134776, 7)
        self.assertAlmostEqual(lib.fp(t2[0]), -41.784696524955393, 5)
        self.assertAlmostEqual(lib.fp(t2[1]), -9675.767769478574, 5)
        self.assertAlmostEqual(lib.fp(t2[2]),  270.75447826471577, 5)
        self.assertAlmostEqual(lib.fp(mycc.amplitudes_to_vector(t1, t2)), 4341.9623137256776, 5)

    def test_vector_to_amplitudes(self):
        t1, t2 = myucc.vector_to_amplitudes(myucc.amplitudes_to_vector(myucc.t1, myucc.t2))
        self.assertAlmostEqual(abs(t1[0]-myucc.t1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t1[1]-myucc.t1[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[0]-myucc.t2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[1]-myucc.t2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[2]-myucc.t2[2]).max(), 0, 12)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.UHF(mol).apply(cc.UCCSD)
        nelec = (3, 3)
        nocc = nelec
        nmo = (5, 5)
        mycc.nocc = nocc
        mycc.nmo = nmo
        vec = numpy.zeros(mycc.vector_size())
        vec_orig = vec.copy()
        t1, t2 = mycc.vector_to_amplitudes(vec)
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        t1a[:] = 1
        t1b[:] = 1
        t2aa[:] = 1
        t2ab[:] = 1
        t2bb[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    def test_vector_size(self):
        self.assertEqual(myucc.vector_size(), 2240)

    def test_update_amps2(self):  # compare to gccsd.update_amps
        mol = mol_s2
        mf = mf_s2
        myucc = uccsd.UCCSD(mf)
        nocca, noccb = 6,4
        nmo = mol.nao_nr()
        nvira,nvirb = nmo-nocca, nmo-noccb
        numpy.random.seed(9)
        t1 = [numpy.random.random((nocca,nvira))-.9,
              numpy.random.random((noccb,nvirb))-.9]
        t2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        t2[0] = t2[0] - t2[0].transpose(1,0,2,3)
        t2[0] = t2[0] - t2[0].transpose(0,1,3,2)
        t2[2] = t2[2] - t2[2].transpose(1,0,2,3)
        t2[2] = t2[2] - t2[2].transpose(0,1,3,2)

        mo_a = mf.mo_coeff[0] + numpy.sin(mf.mo_coeff[0]) * .01j
        mo_b = mf.mo_coeff[1] + numpy.sin(mf.mo_coeff[1]) * .01j
        nao = mo_a.shape[0]
        eri = ao2mo.restore(1, mf._eri, nao)
        eri0aa = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_a.conj(), mo_a, mo_a.conj(), mo_a)
        eri0ab = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_a.conj(), mo_a, mo_b.conj(), mo_b)
        eri0bb = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_b.conj(), mo_b, mo_b.conj(), mo_b)
        eri0ba = eri0ab.transpose(2,3,0,1)

        nvira = nao - nocca
        nvirb = nao - noccb
        eris = uccsd._ChemistsERIs(mol)
        eris.oooo = eri0aa[:nocca,:nocca,:nocca,:nocca].copy()
        eris.ovoo = eri0aa[:nocca,nocca:,:nocca,:nocca].copy()
        eris.oovv = eri0aa[:nocca,:nocca,nocca:,nocca:].copy()
        eris.ovvo = eri0aa[:nocca,nocca:,nocca:,:nocca].copy()
        eris.ovov = eri0aa[:nocca,nocca:,:nocca,nocca:].copy()
        eris.ovvv = eri0aa[:nocca,nocca:,nocca:,nocca:].copy()
        eris.vvvv = eri0aa[nocca:,nocca:,nocca:,nocca:].copy()

        eris.OOOO = eri0bb[:noccb,:noccb,:noccb,:noccb].copy()
        eris.OVOO = eri0bb[:noccb,noccb:,:noccb,:noccb].copy()
        eris.OOVV = eri0bb[:noccb,:noccb,noccb:,noccb:].copy()
        eris.OVVO = eri0bb[:noccb,noccb:,noccb:,:noccb].copy()
        eris.OVOV = eri0bb[:noccb,noccb:,:noccb,noccb:].copy()
        eris.OVVV = eri0bb[:noccb,noccb:,noccb:,noccb:].copy()
        eris.VVVV = eri0bb[noccb:,noccb:,noccb:,noccb:].copy()

        eris.ooOO = eri0ab[:nocca,:nocca,:noccb,:noccb].copy()
        eris.ovOO = eri0ab[:nocca,nocca:,:noccb,:noccb].copy()
        eris.ooVV = eri0ab[:nocca,:nocca,noccb:,noccb:].copy()
        eris.ovVO = eri0ab[:nocca,nocca:,noccb:,:noccb].copy()
        eris.ovOV = eri0ab[:nocca,nocca:,:noccb,noccb:].copy()
        eris.ovVV = eri0ab[:nocca,nocca:,noccb:,noccb:].copy()
        eris.vvVV = eri0ab[nocca:,nocca:,noccb:,noccb:].copy()

        eris.OOoo = eri0ba[:noccb,:noccb,:nocca,:nocca].copy()
        eris.OVoo = eri0ba[:noccb,noccb:,:nocca,:nocca].copy()
        eris.OOvv = eri0ba[:noccb,:noccb,nocca:,nocca:].copy()
        eris.OVvo = eri0ba[:noccb,noccb:,nocca:,:nocca].copy()
        eris.OVov = eri0ba[:noccb,noccb:,:nocca,nocca:].copy()
        eris.OVvv = eri0ba[:noccb,noccb:,nocca:,nocca:].copy()
        eris.VVvv = eri0ba[noccb:,noccb:,nocca:,nocca:].copy()

        eris.focka = numpy.diag(mf.mo_energy[0])
        eris.fockb = numpy.diag(mf.mo_energy[1])
        eris.mo_energy = mf.mo_energy

        t1[0] = t1[0] + numpy.sin(t1[0]) * .05j
        t1[1] = t1[1] + numpy.sin(t1[1]) * .05j
        t2[0] = t2[0] + numpy.sin(t2[0]) * .05j
        t2[1] = t2[1] + numpy.sin(t2[1]) * .05j
        t2[2] = t2[2] + numpy.sin(t2[2]) * .05j
        t1new_ref, t2new_ref = uccsd.update_amps(myucc, t1, t2, eris)

        nocc = nocca + noccb
        orbspin = numpy.zeros(nao*2, dtype=int)
        orbspin[1::2] = 1
        orbspin[nocc-1] = 0
        orbspin[nocc  ] = 1
        eri1 = numpy.zeros([nao*2]*4, dtype=numpy.complex128)
        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        eri1[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa] = eri0aa
        eri1[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb] = eri0ab
        eri1[idxb[:,None,None,None],idxb[:,None,None],idxa[:,None],idxa] = eri0ba
        eri1[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb] = eri0bb
        eri1 = eri1.transpose(0,2,1,3) - eri1.transpose(0,2,3,1)
        erig = gccsd._PhysicistsERIs()
        erig.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        erig.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        erig.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        erig.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        erig.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        erig.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        erig.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        mo_e = numpy.empty(nao*2)
        mo_e[orbspin==0] = mf.mo_energy[0]
        mo_e[orbspin==1] = mf.mo_energy[1]
        erig.fock = numpy.diag(mo_e)
        erig.mo_energy = mo_e.real

        myccg = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        t1 = myccg.spatial2spin(t1, orbspin)
        t2 = myccg.spatial2spin(t2, orbspin)
        t1new, t2new = gccsd.update_amps(myccg, t1, t2, erig)
        t1new = myccg.spin2spatial(t1new, orbspin)
        t2new = myccg.spin2spatial(t2new, orbspin)
        self.assertAlmostEqual(abs(t1new[0] - t1new_ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t1new[1] - t1new_ref[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2new[0] - t2new_ref[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2new[1] - t2new_ref[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2new[2] - t2new_ref[2]).max(), 0, 12)

    def test_mbpt2(self):
        myucc = uccsd.UCCSD(mf)
        e = myucc.kernel(mbpt2=True)[0]
        self.assertAlmostEqual(e, -0.12886859466216125, 8)
        emp2 = mp.MP2(mf).kernel()[0]
        self.assertAlmostEqual(e, emp2, 10)

        myucc = uccsd.UCCSD(mf_s2)
        e = myucc.kernel(mbpt2=True)[0]
        self.assertAlmostEqual(e, -0.096257842171487293, 8)
        emp2 = mp.MP2(mf_s2).kernel()[0]
        self.assertAlmostEqual(e, emp2, 10)

    def test_uintermediats(self):
        from pyscf.cc import uintermediates
        self.assertTrue(eris.get_ovvv().ndim == 4)
        self.assertTrue(eris.get_ovVV().ndim == 4)
        self.assertTrue(eris.get_OVvv().ndim == 4)
        self.assertTrue(eris.get_OVVV().ndim == 4)
        self.assertTrue(eris.get_ovvv(slice(None), slice(2,4)).ndim == 4)
        self.assertTrue(eris.get_ovVV(slice(None), slice(2,4)).ndim == 4)
        self.assertTrue(eris.get_OVvv(slice(None), slice(2,4)).ndim == 4)
        self.assertTrue(eris.get_OVVV(slice(None), slice(2,4)).ndim == 4)
        self.assertTrue(uintermediates._get_vvvv(eris).ndim == 4)
        self.assertTrue(uintermediates._get_vvVV(eris).ndim == 4)
        self.assertTrue(uintermediates._get_VVVV(eris).ndim == 4)

    def test_add_vvvv(self):
        myucc = uccsd.UCCSD(mf_s2)
        nocca, noccb = 6,4
        nmo = mf_s2.mo_occ[0].size
        nvira, nvirb = nmo-nocca, nmo-noccb
        numpy.random.seed(9)
        t1 = [numpy.zeros((nocca,nvira)),
              numpy.zeros((noccb,nvirb))]
        t2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        t2[0] = t2[0] - t2[0].transpose(1,0,2,3)
        t2[0] = t2[0] - t2[0].transpose(0,1,3,2)
        t2[2] = t2[2] - t2[2].transpose(1,0,2,3)
        t2[2] = t2[2] - t2[2].transpose(0,1,3,2)

        eris1 = copy.copy(eris)
        idxa = lib.square_mat_in_trilu_indices(nvira)
        idxb = lib.square_mat_in_trilu_indices(nvirb)
        ref =(lib.einsum('acbd,ijcd->ijab', eris1.vvvv[:,idxa][idxa], t2[0]),
              lib.einsum('acbd,ijcd->ijab', eris1.vvVV[:,idxb][idxa], t2[1]),
              lib.einsum('acbd,ijcd->ijab', eris1.VVVV[:,idxb][idxb], t2[2]))

        t2a = myucc._add_vvvv((t1[0]*0,t1[1]*0), t2, eris, t2sym=False)
        self.assertAlmostEqual(abs(ref[0]-t2a[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(ref[1]-t2a[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(ref[2]-t2a[2]).max(), 0, 12)

        myucc.direct = True
        eris1.vvvv = None  # == with_ovvv=True in the call below
        eris1.VVVV = None
        eris1.vvVV = None
        t1 = None
        myucc.mo_coeff, eris1.mo_coeff = eris1.mo_coeff, None
        t2b = myucc._add_vvvv(t1, t2, eris1)
        self.assertAlmostEqual(abs(ref[0]-t2b[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(ref[1]-t2b[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(ref[2]-t2b[2]).max(), 0, 12)

    def test_add_vvVV(self):
        myucc = uccsd.UCCSD(mf_s2)
        nocca, noccb = 6,4
        nmo = mf_s2.mo_occ[0].size
        nvira, nvirb = nmo-nocca, nmo-noccb
        numpy.random.seed(9)
        t1 = [numpy.zeros((nocca,nvira)),
              numpy.zeros((noccb,nvirb))]
        t2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        t2[0] = t2[0] - t2[0].transpose(1,0,2,3)
        t2[0] = t2[0] - t2[0].transpose(0,1,3,2)
        t2[2] = t2[2] - t2[2].transpose(1,0,2,3)
        t2[2] = t2[2] - t2[2].transpose(0,1,3,2)

        eris1 = copy.copy(eris)
        idxa = lib.square_mat_in_trilu_indices(nvira)
        idxb = lib.square_mat_in_trilu_indices(nvirb)
        vvVV = eris1.vvVV[:,idxb][idxa]
        ref = lib.einsum('acbd,ijcd->ijab', vvVV, t2[1])

        t2a = myucc._add_vvVV((t1[0]*0,t1[1]*0), t2[1], eris)
        self.assertAlmostEqual(abs(ref-t2a).max(), 0, 12)

        myucc.direct = True
        eris1.vvvv = None  # == with_ovvv=True in the call below
        eris1.VVVV = None
        eris1.vvVV = None
        t1 = None
        myucc.mo_coeff, eris1.mo_coeff = eris1.mo_coeff, None
        t2b = myucc._add_vvVV(t1, t2[1], eris1)
        self.assertAlmostEqual(abs(ref-t2b).max(), 0, 12)

    def test_zero_beta_electrons(self):
        mol = gto.M(atom='H', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]),
                    spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        mycc = uccsd.UCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_corr, 0, 9)

        mol = gto.M(atom='He', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]),
                    spin=2, verbose=0)
        mf = scf.UHF(mol).run()
        mycc = uccsd.UCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_corr, -2.6906675843462455e-05, 9)
        self.assertEqual(mycc.t1[1].size, 0)
        self.assertEqual(mycc.t2[1].size, 0)
        self.assertEqual(mycc.t2[2].size, 0)

    def test_reset(self):
        mycc = cc.CCSD(scf.UHF(mol).newton())
        mycc.reset(mol_s2)
        self.assertTrue(mycc.mol is mol_s2)
        self.assertTrue(mycc._scf.mol is mol_s2)

    def test_ao2mo(self):
        mycc = uccsd.UCCSD(mf)
        numpy.random.seed(2)
        nao = mol.nao
        mo = numpy.random.random((2, nao, nao))
        eri_incore = mycc.ao2mo(mo)
        mycc.max_memory = 0
        eri_outcore = mycc.ao2mo(mo)
        self.assertTrue(isinstance(eri_outcore.oovv, h5py.Dataset))
        self.assertAlmostEqual(abs(eri_incore.oooo - eri_outcore.oooo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.oovv - eri_outcore.oovv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovoo - eri_outcore.ovoo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovvo - eri_outcore.ovvo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovov - eri_outcore.ovov).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovvv - eri_outcore.ovvv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.vvvv - eri_outcore.vvvv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OOOO - eri_outcore.OOOO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OOVV - eri_outcore.OOVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVOO - eri_outcore.OVOO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVVO - eri_outcore.OVVO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVOV - eri_outcore.OVOV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVVV - eri_outcore.OVVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.VVVV - eri_outcore.VVVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ooOO - eri_outcore.ooOO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ooVV - eri_outcore.ooVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovOO - eri_outcore.ovOO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovVO - eri_outcore.ovVO).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovOV - eri_outcore.ovOV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovVV - eri_outcore.ovVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.vvVV - eri_outcore.vvVV).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OOvv - eri_outcore.OOvv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVoo - eri_outcore.OVoo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVvo - eri_outcore.OVvo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.OVvv - eri_outcore.OVvv).max(), 0, 12)

    def test_damping(self):
        mol = gto.M(
            atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
            basis = 'ccpvdz',
            spin=1,
            charge=-1,
            symmetry = True,
            verbose = 0
        )
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf)
        mycc.iterative_damping = 0.5
        mycc.run()
        self.assertAlmostEqual(mycc.e_tot, -100.07710261186985, 7)

if __name__ == "__main__":
    print("Full Tests for UCCSD")
    unittest.main()
