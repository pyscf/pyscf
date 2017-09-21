#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto, scf, lib, fci
from pyscf.mcscf import newton_casscf

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ['H', ( 5.,-1.    , 1.   )],
    ['H', ( 0.,-5.    ,-2.   )],
    ['H', ( 4.,-0.5   ,-3.   )],
    ['H', ( 0.,-4.5   ,-1.   )],
    ['H', ( 3.,-0.5   ,-0.   )],
    ['H', ( 0.,-3.    ,-1.   )],
    ['H', ( 2.,-2.5   , 0.   )],
    ['H', ( 1., 1.    , 3.   )],
]
mol.basis = 'sto-3g'
mol.build()
mf = scf.RHF(mol).run(conv_tol=1e-14)
numpy.random.seed(1)
mf.mo_coeff = numpy.random.random(mf.mo_coeff.shape)
mc = newton_casscf.CASSCF(mf, 4, 4)
mc.fcisolver = fci.direct_spin1.FCI(mol)
ci0 = numpy.random.random((6,6))
ci0/= numpy.linalg.norm(ci0)


class KnowValues(unittest.TestCase):
    def test_gen_g_hop(self):
        mo = mc.mo_coeff
        gall, gop, hop, hdiag = newton_casscf.gen_g_hop(mc, mo, ci0, mc.ao2mo(mo))
        self.assertAlmostEqual(lib.finger(gall), 21.288022525148595, 8)
        self.assertAlmostEqual(lib.finger(hdiag), -4.6864640132374618, 8)
        x = numpy.random.random(gall.size)
        u, ci1 = newton_casscf.extract_rotation(mc, x, 1, ci0)
        self.assertAlmostEqual(lib.finger(gop(u, ci1)), -412.9441873541524, 8)
        self.assertAlmostEqual(lib.finger(hop(x)), 73.358310983341198, 8)


if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()


