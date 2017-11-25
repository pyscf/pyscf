#!/usr/bin/env python
import unittest
import copy
import numpy

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc

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

def test_add_vvvv_full(t2, vvvv):
    nvira, nvirb = t2.shape[2:4]
    vvvv = np.zeros((nvira,nvira,nvirb,nvirb))
    idx, idy = np.tril_indices(nvira)
    vvvv[idx,idy] = lib.unpack_tril(vvvv)
    vvvv[idy,idx] = vvvv[idx,idy]
    Ht2 = lib.einsum('ijef,aebf->ijab', t2, vvvv)
    return Ht2

class KnownValues(unittest.TestCase):
    def test_uccsd_from_dft(self):
        mf = dft.UKS(mol)
        mycc = cc.UCCSD(mf)

#    def test_with_df(self):
#        mf = scf.UHF(mol).density_fit(auxbasis='weigend').run()
#        mycc = cc.UCCSD(mf).run()
#        self.assertAlmostEqual(mycc.e_tot, -76.118403942938741, 7)

    def test_ERIS(self):
        ucc1 = cc.UCCSD(mf)
        nao,nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2,nao,nmo))
        eris = cc.uccsd._ChemistsERIs(ucc1, mo_coeff, method='outcore')

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.9638849382825754, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooov)), 21.35362101033294 , 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3623681896983584, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovo)), 26.411461005316333, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.81550684442163, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.123681017639598, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)), 133.48083527898248, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 59.421927525288183, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)), 43.556602622204778, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOOO)),-407.05319440524585, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOOV)), 48.215114675855148, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVOO)), 56.284299937160796, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOVO)), 41.237969987993381, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVOV)),-287.72899895597448, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOVV)),-85.484299959144522, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVVO)),-228.18996145476956, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVVV)),-10.715902258877399, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.VVVV)),-89.908425473958303, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooOO)),-336.65979260175226, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooOV)), 57.513484419020301, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovOO)),-16.405125847288176, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooVO)), 48.42215717159899 , 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovOV)), 231.59042209500075, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooVV)), 20.338077193028354, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovVO)), 206.48662856981386, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovVV)),-71.273249852220516, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvVV)), 172.47130671068496, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOov)), 19.863401671450099, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVoo)),-19.927660309103977, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOvo)), 15.969375821404906, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OOvv)),-27.761433381797019, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVvo)),-140.09648311337384, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.OVvv)), 40.700983950220547, 14)

    def test_ERISspin(self):
        ucc1 = cc.UCCSD(mf)
        nao,nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2,nao,nmo))
        eris = cc.uccsd._ERISspin(ucc1, mo_coeff, method='outcore')
        self.assertAlmostEqual(lib.finger(numpy.array(eris.orbspin)), -1.3026329974802735, 14)

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)),-295.08461560808291, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooov)), 1095.2476264858244, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)), 2046.6153861594653, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovo)),-958.26213789197777, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)),-916.29883128755637, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)),-86.037304248772571, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)),-204.12413647564222, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 1619.8406027545959, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)), 524.34821561384456, 14)

    def test_amplitudes_from_rccsd(self):
        e, t1, t2 = cc.RCCSD(rhf).set(conv_tol=1e-10).kernel()
        t1, t2 = myucc.amplitudes_from_rccsd(t1, t2)
        self.assertAlmostEqual(abs(t1-myucc.spatial2spin(myucc.t1)).max(), 0, 6)
        self.assertAlmostEqual(abs(t2-myucc.spatial2spin(myucc.t2)).max(), 0, 6)

    def test_uccsd_rdm(self):
        dm1 = myucc.make_rdm1()
        dm2 = myucc.make_rdm2()
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 3.1080942935191711, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 13.151382528402792, 6)

    def test_uccsd_frozen(self):
        ucc1 = copy.copy(myucc)
        ucc1.frozen = 1
        self.assertEqual(ucc1.nmo, 24)
        self.assertEqual(ucc1.nocc, 8)
        ucc1.frozen = [0,1]
        self.assertEqual(ucc1.nmo, 22)
        self.assertEqual(ucc1.nocc, 6)
        ucc1.frozen = [1,9]
        self.assertEqual(ucc1.nmo, 22)
        self.assertEqual(ucc1.nocc, 8)
        ucc1.frozen = [9,10,12]
        self.assertEqual(ucc1.nmo, 20)
        self.assertEqual(ucc1.nocc, 10)
        ucc1.nmo = (13,12)
        ucc1.nocc = (5,4)
        self.assertEqual(ucc1.nmo, 25)
        self.assertEqual(ucc1.nocc, 9)


if __name__ == "__main__":
    print("Full Tests for UCCSD")
    unittest.main()

