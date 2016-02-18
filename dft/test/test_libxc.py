#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto, scf
from pyscf import dft

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = 'h 0 0 0; h 1 .5 0; h 0 4 1; h 1 0 .2'
mol.basis = 'aug-ccpvdz'
mol.build()
dm = scf.RHF(mol).run().make_rdm1()
mf = dft.RKS(mol)
mf.grids.atom_grid = {"H": (50, 110)}
mf.prune = None
mf.grids.build()
nao = mol.nao_nr()
ao = dft.numint.eval_ao(mol, mf.grids.coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(w, a.ravel())

class KnowValues(unittest.TestCase):
    def test_parse_xc(self):
        hyb, fn_facs = dft.libxc.parse_xc('.5*HF+.5*B3LYP,.5*VWN')
        self.assertAlmostEqual(hyb, .7, 12)
        self.assertEqual([x[0] for x in fn_facs], [1,106,131,7])
        self.assertTrue(numpy.allclose([x[1] for x in fn_facs],
                                       (0.08, 0.72, 0.81, 0.69)))

    def test_lyp(self):
        e,v,f = dft.libxc.eval_xc(',LYP', rho, deriv=2)[:3]
        self.assertAlmostEqual(finger(e), -0.36064427429079671, 3)
        self.assertAlmostEqual(finger(v[0]), -0.75835482781412178, 3)
        self.assertAlmostEqual(finger(v[1]), -201.87936893573291, 3)
        self.assertAlmostEqual(finger(f[0]), 41297.120111016673, 1)
        self.assertAlmostEqual(finger(f[1]), -962737.70242896723, 2)
        self.assertAlmostEqual(finger(f[2]), -3.3540180961110231, 9)

if __name__ == "__main__":
    print("Test libxc")
    unittest.main()


