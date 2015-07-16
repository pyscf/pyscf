#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fciqmcscf

b = 1.4
mol = gto.Mole()

mol.build(
        verbose = 5,
output = None,
atom = [['He',(  0.000000,  0.000000, -b/2)],
        ['He',(  0.000000,  0.000000,  b)]],
basis = {'He': 'cc-pvdz'},
symmetry = False,
#symmetry_subgroup = 'D2h',
)

m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

class KnowValues(unittest.TestCase):
    def test_mc2step_7o4e_fciqmc_4states(self):
        mc = mcscf.CASSCF(m, 7, 4)
        mc.max_cycle_macro = 10
#        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
#        mc.fcisolver.RDMSamples = 2000
#        mc.fcisolver.maxwalkers = 3000
        mc.fcisolver.state_weights = [1.00]

        emc = mc.mc2step()[0]
        print('Final energy:', emc)

if __name__ == "__main__":
    unittest.main()
