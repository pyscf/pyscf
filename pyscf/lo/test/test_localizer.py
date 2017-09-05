#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto, scf
from pyscf.lo import boys, edmiston, pipek

mol = gto.Mole()
mol.atom = '''
     H    0.000000000000     2.491406946734     0.000000000000
     C    0.000000000000     1.398696930758     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
  '''
mol.basis = '6-31g'
mol.symmetry = 0
mol.verbose = 0
mol.build()
mf = scf.RHF(mol).run()

# note tests may fail due to initial guess problem

class KnowValues(unittest.TestCase):
    def test_boys(self):
        idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
        loc = boys.Boys(mol)
        loc.max_cycle = 100
        mo = loc.kernel(mf.mo_coeff[:,idx])
        dip = boys.dipole_integral(mol, mo)
        z = numpy.einsum('xii,xii->', dip, dip)
        self.assertAlmostEqual(z, 98.670988758151907, 4)

#        mo = loc.kernel(mf.mo_coeff[:,idx+1])
#        dip = boys.dipole_integral(mol, mo)
#        z = numpy.einsum('xii,xii->', dip, dip)
#        self.assertAlmostEqual(z, 27.481320331665497, 4)

    #def test_edmiston(self):
    #    idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
    #    loc = edmiston.EdmistonRuedenberg(mol)
    #    mo = loc.kernel(mf.mo_coeff[:,idx])
    #    dip = boys.dipole_integral(mol, mo)
    #    z = numpy.einsum('xii,xii->', dip, dip)
    #    self.assertAlmostEqual(z, 79.73132964805923, 9)

    def test_pipek(self):
        idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
        loc = pipek.PipekMezey(mol)
        loc.max_cycle = 100
        mo = loc.kernel(mf.mo_coeff[:,idx])
        pop = pipek.atomic_pops(mol, mo)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 12, 4)


if __name__ == "__main__":
    print("Full Tests for localizer")
    unittest.main()
