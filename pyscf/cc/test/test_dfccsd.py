#!/usr/bin/env python
import unittest
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
mf = scf.RHF(mol).density_fit(auxbasis='weigend')
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = cc.RCCSD(mf).run(conv_tol=1e-10)

class KnownValues(unittest.TestCase):
    def test_with_df(self):
        self.assertAlmostEqual(mycc.e_tot, -76.118403942938741, 7)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.ccsd._ERIS(mycc, mo_coeff)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.962033460861587 , 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ooov)), 21.3528747097292  , 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3666078517246127, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.122525571320821, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.80972789115584, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 59.418747028576142, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)), 43.562457227975969, 12)

    def test_df_eaccsd(self):
        cc1 = cc.RCCSD(mf)
        cc1.max_memory = 0
        cc1.run()
        e,v = cc1.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.19040177199076605, 6)
        self.assertAlmostEqual(e[1], 0.28341260743397712, 6)
        self.assertAlmostEqual(e[2], 0.52230183774970684, 6)


if __name__ == "__main__":
    print("Full Tests for DFCCSD")
    unittest.main()

