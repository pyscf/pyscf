#!/usr/bin/env python
import unittest
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import uccsd

def finger(a):
    return numpy.dot(a.ravel(), numpy.cos(numpy.arange(a.size)))

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.spin = 0
mol.build()
mf = scf.UHF(mol).run()

mol1 = gto.Mole()
mol1.verbose = 0
mol1.atom = [['O', (0.,   0., 0.)],
             ['O', (1.21, 0., 0.)]]
mol1.basis = 'cc-pvdz'
mol1.spin = 2
mol1.build()
mf1 = scf.UHF(mol1).run()

no = mol1.nelectron
n = mol1.nao_nr()
nv = n * 2 - no
mf1.mo_occ = numpy.zeros((2,mol1.nao_nr()))
mf1.mo_occ[:,:no//2] = 1
numpy.random.seed(12)
mf1.mo_coeff = numpy.random.random((2,n,n))
dm = mf1.make_rdm1(mf1.mo_coeff, mf1.mo_occ)
fockao = mf1.get_hcore() + mf1.get_veff(mol1, dm)
mo_energya = numpy.einsum('pi,pq,qi->i', mf1.mo_coeff[0], fockao[0], mf1.mo_coeff[0])
mo_energyb = numpy.einsum('pi,pq,qi->i', mf1.mo_coeff[1], fockao[1], mf1.mo_coeff[1])
idxa = numpy.hstack([mo_energya[:no//2].argsort(), no//2+mo_energya[no//2:].argsort()])
idxb = numpy.hstack([mo_energyb[:no//2].argsort(), no//2+mo_energyb[no//2:].argsort()])
mf1.mo_coeff[0] = mf1.mo_coeff[0][:,idxa]
mf1.mo_coeff[1] = mf1.mo_coeff[1][:,idxb]

ucc1 = cc.UCCSD(mf1)
ucc1.eris = eris = ucc1.ao2mo()

numpy.random.seed(12)
r1 = numpy.random.random((no,nv)) - .9
r2 = numpy.random.random((no,no,nv,nv)) - .9
r1,r2 = ucc1.vector_to_amplitudes_ee(ucc1.amplitudes_to_vector_ee(r1,r2))
r1 = ucc1.spin2spatial(r1, eris.orbspin)
r2 = ucc1.spin2spatial(r2, eris.orbspin)
ucc1.eris = eris
ucc1.t1 = [x*1e-5 for x in r1]
ucc1.t2 = [x*1e-5 for x in r2]

class KnowValues(unittest.TestCase):
    def test_frozen(self):
        mf1 = scf.UHF(mol1).run()
        # Freeze 1s electrons
        frozen = [[0,1], [0,1]]
        ucc = cc.UCCSD(mf1, frozen=frozen)
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.34869875247588372, 8)

    def test_eomee(self):
        ucc = cc.UCCSD(mf)
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.2133432430989155, 6)
        e,v = ucc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_ucc_eris(self):
        self.assertAlmostEqual(finger(numpy.asarray(eris.oooo)), -154.31628911898906, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooov)), -31.066274442849803, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovoo)), -212.0603115718182 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.oovo)), -38.299597545260752, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovov)), 210.41308081802879 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.oovv)), 245.13897568200619 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovvo)), 123.78478694710006 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovvv)), 117.6961223005484  , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.vvvv)), 621.90765168101734 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOOO)), -44.969919923376551, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOOV)), -28.711905693268207, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVOO)), -51.281877592852744, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOVO)), -85.875920039850328, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVOV)), 124.85555105382456 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOVV)), 227.21768155937997 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVVO)), 129.27273096927016 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVVV)), -32.30772293206823 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.VVVV)), 491.89926355710327 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooOO)), -97.313184405746085, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooOV)), 7.3290828950903375 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovOO)), -104.3395245453126 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooVO)), -74.022271485865417, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovOV)), 135.97684392544903 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooVV)), 257.30668364428919 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovVO)), 124.86973015057238 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovVV)), 139.19179627029064 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.vvVV)), 550.46227687665487 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOov)), -70.877846134069088, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVoo)), -82.161516922543868, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOvo)), -54.97442192794999 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OOvv)), 200.69552671113721 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVvo)), 144.41387241954908 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.OVvv)), 21.940358346011749 , 9)

    def test_ucc_update_amps(self):
        t1, t2 = ucc1.update_amps(r1, r2, eris)
        t1 = ucc1.spatial2spin(t1, eris.orbspin)
        t2 = ucc1.spatial2spin(t2, eris.orbspin)
        self.assertAlmostEqual(finger(t1)*1e-6, -0.25406629897519412, 9)
        self.assertAlmostEqual(finger(t2)*1e-6, -369.77282889007887 , 6)
        self.assertAlmostEqual(uccsd.energy(ucc1, r1, r2, eris), 212092.42063102487, 8)
        e0, t1, t2 = ucc1.init_amps(eris)
        self.assertAlmostEqual(finger(ucc1.spatial2spin(t1, eris.orbspin)), -1388.6092444316866, 9)
        self.assertAlmostEqual(finger(ucc1.spatial2spin(t2, eris.orbspin)), -38008.739917327577, 4)
        self.assertAlmostEqual(e0, 5308849.5847222833, 3)
        #t1, t2 = ucc1.update_amps(t1, t2, eris)
        #self.assertAlmostEqual(finger(ucc1.spatial2spin(t1, eris.orbspin)), -163451623851.87241, 2)
        #self.assertAlmostEqual(finger(ucc1.spatial2spin(t2, eris.orbspin)), 186007137548528.5  , 0)

    def test_ucc_eomee_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r1,r2 = ucc1.vector_to_amplitudes_ee(ucc1.amplitudes_to_vector_ee(r1,r2))
        r1 = ucc1.spin2spatial(r1, eris.orbspin)
        r2 = ucc1.spin2spatial(r2, eris.orbspin)
        vec = ucc1.amplitudes_to_vector(r1,r2)
        vec1 = ucc1.eomee_ccsd_matvec(vec)
        r1, r2 = ucc1.vector_to_amplitudes(vec1)
        r1 = ucc1.spatial2spin(r1, eris.orbspin)
        r2 = ucc1.spatial2spin(r2, eris.orbspin)
        vec1 = ucc1.amplitudes_to_vector_ee(r1,r2)
        self.assertAlmostEqual(finger(vec1), 315635.74451279285, 8)

    def test_ucc_eomee_ccsd_diag(self):
        vec1, vec2 = ucc1.eeccsd_diag()
        self.assertAlmostEqual(finger(vec1), 1668.0265772801999, 9)
        self.assertAlmostEqual(finger(vec2), 2293.0574123253973, 9)

    def test_ucc_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r1,r2 = ucc1.vector_to_amplitudes_ee(ucc1.amplitudes_to_vector_ee(r1,r2))
        r1 = ucc1.spin2spatial_eomsf(r1, eris.orbspin)
        r2 = ucc1.spin2spatial_eomsf(r2, eris.orbspin)
        vec = ucc1.amplitudes_to_vector_eomsf(r1,r2)
        vec1 = ucc1.eomsf_ccsd_matvec(vec)
        r1, r2 = ucc1.vector_to_amplitudes_eomsf(vec1)
        r1 = ucc1.spatial2spin_eomsf(r1, eris.orbspin)
        r2 = ucc1.spatial2spin_eomsf(r2, eris.orbspin)
        vec1 = ucc1.amplitudes_to_vector_ee(r1,r2)
        self.assertAlmostEqual(finger(vec1), 94551.862963518928, 8)

#    def test_ucc_eomip_matvec(self):
#        numpy.random.seed(12)
#        r1 = numpy.random.random((no)) - .9
#        r2 = numpy.random.random((no,no,nv)) - .9
#        vec = ucc1.amplitudes_to_vector_ip(r1,r2)
#        r1,r2 = ucc1.vector_to_amplitudes_ip(vec)
#        self.assertAlmostEqual(finger(r1), 1.0885839546838316, 12)
#        self.assertAlmostEqual(finger(r2), 10.141552631959476, 12)
#        vec1 = ucc1.ipccsd_matvec(vec)
#        self.assertAlmostEqual(finger(vec1), -115342.31613492536, 9)
#
#        self.assertAlmostEqual(finger(ucc1.ipccsd_diag()), 2552.1564739826904, 9)
#
#    def test_ucc_eomea_matvec(self):
#        numpy.random.seed(12)
#        r1 = numpy.random.random((nv)) - .9
#        r2 = numpy.random.random((no,nv,nv)) - .9
#        vec = ucc1.amplitudes_to_vector_ea(r1,r2)
#        r1,r2 = ucc1.vector_to_amplitudes_ea(vec)
#        self.assertAlmostEqual(finger(r1), 0.83945192988432771, 12)
#        self.assertAlmostEqual(finger(r2), -0.12857498913331167, 12)
#        vec1 = ucc1.eaccsd_matvec(vec)
#        self.assertAlmostEqual(finger(vec1), -3334.2883930428034, 9)
#
#        self.assertAlmostEqual(finger(ucc1.eaccsd_diag()), 4011.3514490902526, 9)

if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()

