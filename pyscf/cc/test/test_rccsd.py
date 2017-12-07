#!/usr/bin/env python
import unittest
import copy
import numpy

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf.cc import rccsd

mol = gto.Mole()
mol.verbose = 7
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = rccsd.RCCSD(mf).run(conv_tol=1e-10)


class KnownValues(unittest.TestCase):
    def test_roccsd(self):
        mf = scf.ROHF(mol).run()
        mycc = cc.RCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -76.119346385357446, 7)

    def test_dump_chk(self):
        cc1 = copy.copy(mycc)
        cc1.nmo = mf.mo_energy.size
        cc1.nocc = mol.nelectron // 2
        cc1.dump_chk()
        cc1 = cc.CCSD(mf)
        cc1.__dict__.update(lib.chkfile.load(cc1._scf.chkfile, 'ccsd'))
        eris = cc1.ao2mo()
        e = cc1.energy(cc1.t1, cc1.t2, eris)
        self.assertAlmostEqual(e, -0.13539788638119823, 8)

    def test_ccsd_t(self):
        e = mycc.ccsd_t()
        self.assertAlmostEqual(e, -0.0009964234049929792, 10)

    def test_ao_direct(self):
        cc1 = cc.CCSD(mf)
        cc1.direct = True
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13539788638119823, 8)

    def test_diis(self):
        cc1 = cc.CCSD(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13516622806104395, 8)

    def test_ERIS(self):
        cc1 = cc.RCCSD(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.rccsd._make_eris_outcore(cc1, mo_coeff)

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.9638849382825754, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3623681896984081, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)), 133.4808352789826 , 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.123681017639655, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.81550684442149, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 95.756230114113322, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)),-10.450387490987545, 12)

    def test_amplitudes_to_vector(self):
        vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
        #self.assertAlmostEqual(lib.finger(vec), -0.056992042448099592, 6)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        self.assertAlmostEqual(abs(r1-mycc.t1).max(), 0, 14)
        self.assertAlmostEqual(abs(r2-mycc.t2).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        vec1 = mycc.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(vec-vec1).max(), 0, 14)

    def test_rccsd_frozen(self):
        cc1 = copy.copy(mycc)
        cc1.frozen = 1
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [0,1]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 3)
        cc1.frozen = [1,9]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [9,10,12]
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 5)
        cc1.nmo = 10
        cc1.nocc = 6
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 6)


if __name__ == "__main__":
    print("Full Tests for RCCSD")
    unittest.main()

