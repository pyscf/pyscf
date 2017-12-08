import unittest
from pyscf import ao2mo
import numpy
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.cc import uccsd
from pyscf.cc import addons
from pyscf.cc import uccsd_lambda
from pyscf.cc import gccsd, gccsd_lambda

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '631g'
mol.spin = 2
mol.build()

class KnownValues(unittest.TestCase):
    def test_update_amps(self):
        mf = scf.UHF(mol).run()
        numpy.random.seed(21)
        mycc = uccsd.UCCSD(mf)
        eris = mycc.ao2mo()
        gcc1 = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        eri1 = gcc1.ao2mo()
        orbspin = eri1.orbspin

        nocc = mol.nelectron
        nvir = mol.nao_nr()*2 - nocc

        t1r = numpy.random.random((nocc,nvir))*.1
        t2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
        t2r = t2r - t2r.transpose(1,0,2,3)
        t2r = t2r - t2r.transpose(0,1,3,2)
        l1r = numpy.random.random((nocc,nvir))*.1
        l2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
        l2r = l2r - l2r.transpose(1,0,2,3)
        l2r = l2r - l2r.transpose(0,1,3,2)
        t1r = addons.spin2spatial(t1r, orbspin)
        t2r = addons.spin2spatial(t2r, orbspin)
        t1r = addons.spatial2spin(t1r, orbspin)
        t2r = addons.spatial2spin(t2r, orbspin)
        l1r = addons.spin2spatial(l1r, orbspin)
        l2r = addons.spin2spatial(l2r, orbspin)
        l1r = addons.spatial2spin(l1r, orbspin)
        l2r = addons.spatial2spin(l2r, orbspin)
        imds = gccsd_lambda.make_intermediates(gcc1, t1r, t2r, eri1)
        l1ref, l2ref = gccsd_lambda.update_lambda(gcc1, t1r, t2r, l1r, l2r, eri1, imds)

        t1 = addons.spin2spatial(t1r, orbspin)
        t2 = addons.spin2spatial(t2r, orbspin)
        l1 = addons.spin2spatial(l1r, orbspin)
        l2 = addons.spin2spatial(l2r, orbspin)
        imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1, l2 = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(abs(addons.spatial2spin(l1, orbspin)-l1ref).max(), 0, 8)
        self.assertAlmostEqual(abs(addons.spatial2spin(l2, orbspin)-l2ref).max(), 0, 8)

        l1ref = addons.spin2spatial(l1ref, orbspin)
        l2ref = addons.spin2spatial(l2ref, orbspin)
        self.assertAlmostEqual(abs(l1[0]-l1ref[0]).max(), 0, 8)
        self.assertAlmostEqual(abs(l1[1]-l1ref[1]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[0]-l2ref[0]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[1]-l2ref[1]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[2]-l2ref[2]).max(), 0, 8)


if __name__ == "__main__":
    print("Full Tests for UCCSD lambda")
    unittest.main()
