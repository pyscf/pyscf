#!/usr/bin/env python
import unittest
import copy
import numpy

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.cc import addons
from pyscf.fci import direct_uhf

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

class KnownValues(unittest.TestCase):
#    def test_with_df(self):
#        mf = scf.UHF(mol).density_fit(auxbasis='weigend').run()
#        mycc = cc.UCCSD(mf).run()
#        self.assertAlmostEqual(mycc.e_tot, -76.118403942938741, 7)

    def test_ERIS(self):
        ucc1 = cc.UCCSD(mf)
        nao,nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2,nao,nmo))
        eris = cc.uccsd._make_eris_outcore(ucc1, mo_coeff)

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.9638849382825754, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3623681896983584, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.81550684442163, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.123681017639598, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)), 133.48083527898248, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 59.421927525288183, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)), 43.556602622204778, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOOO)),-407.05319440524585, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVOO)), 56.284299937160796, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVOV)),-287.72899895597448, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOVV)),-85.484299959144522, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVVO)),-228.18996145476956, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVVV)),-10.715902258877399, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.VVVV)),-89.908425473958303, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooOO)),-336.65979260175226, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovOO)),-16.405125847288176, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovOV)), 231.59042209500075, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooVV)), 20.338077193028354, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovVO)), 206.48662856981386, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovVV)),-71.273249852220516, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvVV)), 172.47130671068496, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVoo)),-19.927660309103977, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOvv)),-27.761433381797019, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVvo)),-140.09648311337384, 11)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVvv)), 40.700983950220547, 11)

        eris0 = cc.uccsd._make_eris_incore(ucc1, mo_coeff)
        self.assertAlmostEqual(abs(numpy.array(eris.oooo)-eris0.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovoo)-eris0.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovov)-eris0.ovov).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.oovv)-eris0.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovvo)-eris0.ovvo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovvv)-eris0.ovvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.vvvv)-eris0.vvvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OOOO)-eris0.OOOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVOO)-eris0.OVOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVOV)-eris0.OVOV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OOVV)-eris0.OOVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVVO)-eris0.OVVO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVVV)-eris0.OVVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.VVVV)-eris0.VVVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ooOO)-eris0.ooOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovOO)-eris0.ovOO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovOV)-eris0.ovOV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ooVV)-eris0.ooVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovVO)-eris0.ovVO).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.ovVV)-eris0.ovVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.vvVV)-eris0.vvVV).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVoo)-eris0.OVoo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OOvv)-eris0.OOvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVvo)-eris0.OVvo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris.OVvv)-eris0.OVvv).max(), 0, 11)

    def test_amplitudes_from_rccsd(self):
        e, t1, t2 = cc.RCCSD(rhf).set(conv_tol=1e-10).kernel()
        t1, t2 = myucc.amplitudes_from_rccsd(t1, t2)
        self.assertAlmostEqual(abs(t1[0]-myucc.t1[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(t1[1]-myucc.t1[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(t2[0]-myucc.t2[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(t2[1]-myucc.t2[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(t2[2]-myucc.t2[2]).max(), 0, 6)

#    def test_uccsd_rdm(self):
#        dm1 = myucc.make_rdm1()
#        dm2 = myucc.make_rdm2()
#        self.assertAlmostEqual(numpy.linalg.norm(dm1), 3.1080942935191711, 6)
#        self.assertAlmostEqual(numpy.linalg.norm(dm2), 13.151382528402792, 6)

    def test_uccsd_frozen(self):
        ucc1 = copy.copy(myucc)
        ucc1.frozen = 1
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
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.basis = '631g'
        mol.spin = 2
        mol.build()
        mf = scf.UHF(mol).run()

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
        e1+= numpy.einsum('ijkl,jilk', eriaa, dm2aa) * .5
        e1+= numpy.einsum('ijkl,jilk', eriab, dm2ab)
        e1+= numpy.einsum('ijkl,jilk', eribb, dm2bb) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 7)

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
        mf = scf.UHF(mol).run(conv_tol=1e-14)
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
        self.assertAlmostEqual(abs(dm1ref[0] - rdm1[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm1ref[1] - rdm1[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[0] - rdm2[0]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[1] - rdm2[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(dm2ref[2] - rdm2[2]).max(), 0, 6)


if __name__ == "__main__":
    print("Full Tests for UCCSD")
    unittest.main()

