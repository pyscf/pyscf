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
#output = 'casscf.out',
output = None,
atom = [['Li',(  0.000000,  0.000000, 1.005436697)],
        ['H',(  0.000000,  0.000000,  0.0)]],
basis = {'H': 'sto-3g', 'Li': 'sto-3g'},
symmetry = 'C2v'
)

m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

class KnowValues(unittest.TestCase):
    def test_dipoles_hfbasis(self):
        fciqmc_calc = fciqmcscf.FCIQMCCI(mol)
        fciqmc_calc.tau = 0.01
        fciqmc_calc.RDMSamples = 2000

        norb = m.mo_coeff.shape[1]
        e = fciqmcscf.fciqmc.run_standalone(fciqmc_calc, m.mo_coeff)
        dips = fciqmc_calc.dipoles(m.mo_coeff, fcivec=None, norb=norb, nelec=mol.nelectron)

        self.assertAlmostEqual(e,-7.787146064428100, 5)
        self.assertAlmostEqual(dips[0],0.0,7)
        self.assertAlmostEqual(dips[1],0.0,7)
        self.assertAlmostEqual(dips[2],1.85781390006,4)

    def test_dipoles_casscfbasis(self):

        mc = mcscf.CASSCF(m,6,4)    #There are only 6 orbitals, 4 electrons, so this is the full space, giving the exact NO basis
        mc.natorb = True            #Ensures that casscf_mo returns the natural orbital basis in the active space
        emc,e_ci,fcivec,casscf_mo = mc.mc2step(m.mo_coeff)

        fciqmc_calc = fciqmcscf.FCIQMCCI(mol)
        fciqmc_calc.tau = 0.01
        fciqmc_calc.RDMSamples = 2000
        norb = mc.mo_coeff.shape[1]
        e = fciqmcscf.fciqmc.run_standalone(fciqmc_calc, casscf_mo)   #Run from CASSCF natural orbitals
        dips = fciqmc_calc.dipoles(casscf_mo, fcivec=None, norb=norb, nelec=mol.nelectron)

        self.assertAlmostEqual(e,-7.787146064428100, 5)
        self.assertAlmostEqual(dips[0],0.0,7)
        self.assertAlmostEqual(dips[1],0.0,7)
        self.assertAlmostEqual(dips[2],1.85781390006,4)

if __name__ == "__main__":
    print('Tests for dipole moments from standalone FCIQMC calculation in HF and natural orbital basis')
    unittest.main()
