#!/usr/bin/env python
import unittest
import copy
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo

def finger(a):
    return numpy.dot(a.ravel(), numpy.cos(numpy.arange(a.size)))

mol = gto.Mole()
mol.atom = [
[8 , (0. , 0.     , 0.)],
[1 , (0. , -0.757 , 0.587)],
[1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.verbose = 0
mol.spin = 0
mol.build()
mf = scf.RHF(mol).run()

mf1 = copy.copy(mf)
no = mol.nelectron // 2
n = mol.nao_nr()
nv = n - no
mf1.mo_occ = numpy.zeros(mol.nao_nr())
mf1.mo_occ[:no] = 2
numpy.random.seed(12)
mf1.mo_coeff = numpy.random.random((n,n))
dm = mf1.make_rdm1(mf1.mo_coeff, mf1.mo_occ)
fockao = mf1.get_hcore() + mf1.get_veff(mol, dm)
mf1.mo_energy = numpy.einsum('pi,pq,qi->i', mf1.mo_coeff, fockao, mf1.mo_coeff)
idx = numpy.hstack([mf1.mo_energy[:no].argsort(), no+mf1.mo_energy[no:].argsort()])
mf1.mo_coeff = mf1.mo_coeff[:,idx]

mycc1 = cc.RCCSD(mf1)
mycc1.eris = eris = mycc1.ao2mo()
numpy.random.seed(12)
r1 = numpy.random.random((no,nv)) - .9
r2 = numpy.random.random((no,no,nv,nv)) - .9
r2 = r2 + r2.transpose(1,0,3,2)
mycc1.t1 = r1*1e-5
mycc1.t2 = r2*1e-5

class KnowValues(unittest.TestCase):
    def test_ipccsd(self):
        mycc = cc.RCCSD(mf)
        ecc, t1, t2 = mycc.kernel()
        e,v = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

    def test_eaccsd(self):
        mycc = cc.RCCSD(mf)
        ecc, t1, t2 = mycc.kernel()
        e,v = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.51006797826488071, 6)

    def test_eeccsd(self):
        mycc = cc.RCCSD(mf)
        ecc, t1, t2 = mycc.kernel()
        e,v = mycc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_eris(self):
        self.assertAlmostEqual(finger(numpy.asarray(eris.oooo)), -452.80659100759118, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ooov)), -143.2416085514744 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.oovo)), -34.505774311007372, 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovov)), 391.38496487473122 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.oovv)), 604.1880629302932  , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovvo)), 175.13231520748195 , 9)
        self.assertAlmostEqual(finger(numpy.asarray(eris.ovvv)), -211.2821359975481 , 9)
        self.assertAlmostEqual(finger(ao2mo.restore(1,eris.vvvv,nv)), 538.74339637596995, 9)

    def test_vector_to_amplitudes(self):
        t1, t2 = mycc1.vector_to_amplitudes(mycc1.amplitudes_to_vector(r1,r2))
        self.assertAlmostEqual(abs(r1-t1).sum(), 0, 9)
        self.assertAlmostEqual(abs(r2-t2).sum(), 0, 9)

    def test_eomee_ccsd_matvec_singlet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        vec = mycc1.amplitudes_to_vector(r1,r2)
        vec1 = mycc1.eomee_ccsd_matvec_singlet(vec)
        r1, r2 = mycc1.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(finger(r1), -112883.3791497977, 8)
        self.assertAlmostEqual(finger(r2), -268199.3475813322, 8)

    def test_eomee_ccsd_matvec_triplet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        r2[0] = r2[0] - r2[0].transpose(0,1,3,2)
        r2[0] = r2[0] - r2[0].transpose(1,0,2,3)
        r2[1] = r2[1] - r2[1].transpose(1,0,3,2)
        vec = mycc1.amplitudes_to_vector_triplet(r1, r2)
        vec1 = mycc1.eomee_ccsd_matvec_triplet(vec)
        r1, r2 = mycc1.vector_to_amplitudes_triplet(vec1)
        self.assertAlmostEqual(finger(r1   ), 3550.5250670914056, 9)
        self.assertAlmostEqual(finger(r2[0]), -237433.03756895234,8)
        self.assertAlmostEqual(finger(r2[1]), 127680.0182437716 , 8)

    def test_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        vec = mycc1.amplitudes_to_vector_eomsf(r1,r2)
        vec1 = mycc1.eomsf_ccsd_matvec(vec)
        r1, r2 = mycc1.vector_to_amplitudes_eomsf(vec1)
        self.assertAlmostEqual(finger(r1   ), -19368.729268465482, 8)
        self.assertAlmostEqual(finger(r2[0]), 84325.863680611626 , 8)
        self.assertAlmostEqual(finger(r2[1]), 6715.9574457836134 , 8)

    def test_eomee_diag(self):
        vec1S, vec1T, vec2 = mycc1.eeccsd_diag()
        self.assertAlmostEqual(finger(vec1S), 62.028729797614801, 9)
        self.assertAlmostEqual(finger(vec1T), 2221.3155272953709, 9)
        self.assertAlmostEqual(finger(vec2) ,-5486.1611871545592, 9)

if __name__ == "__main__":
    print("Tests for EOM RCCSD")
    unittest.main()

