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
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = cc.RCCSD(mf).run(conv_tol=1e-10)


class KnownValues(unittest.TestCase):
    def test_rccsd_from_dft(self):
        mf = dft.RKS(mol)
        mycc = cc.CCSD(mf)
        mycc = cc.RCCSD(mf)

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
        self.assertAlmostEqual(e, -0.13539788638119823, 10)

    def test_ccsd_t(self):
        e = mycc.ccsd_t()
        self.assertAlmostEqual(e, -0.0009964234049929792, 10)

    def test_ao_direct(self):
        cc1 = cc.CCSD(mf)
        cc1.direct = True
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13539788638119823, 9)

    def test_diis(self):
        cc1 = cc.CCSD(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13516622806104395, 9)

    def test_ERIS(self):
        cc1 = cc.CCSD(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.rccsd._ERIS(cc1, mo_coeff, method='outcore')

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.9638849382825754, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooov)), 21.35362101033294 , 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3623681896983584, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.81550684442163, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovo)), 26.411461005316333, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.123681017639598, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)), 133.48083527898248, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 59.421927525288183, 14)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)), 43.556602622204778, 14)

    def test_amplitudes_to_vector_ee(self):
        vec = mycc.amplitudes_to_vector_ee(mycc.t1, mycc.t2)
        #self.assertAlmostEqual(lib.finger(vec), -0.056992042448099592, 6)
        r1, r2 = mycc.vector_to_amplitudes_ee(vec)
        self.assertAlmostEqual(abs(r1-mycc.t1).max(), 0, 14)
        self.assertAlmostEqual(abs(r2-mycc.t2).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        r1, r2 = mycc.vector_to_amplitudes_ee(vec)
        vec1 = mycc.amplitudes_to_vector_ee(r1, r2)
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

