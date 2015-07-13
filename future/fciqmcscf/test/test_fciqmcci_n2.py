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
verbose = 0,
output = None,
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': 'ccpvdz', },
symmetry = 'D2h'
)
m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()


class KnowValues(unittest.TestCase):
    def test_mc2step_4o4e_fci(self):
        mc = mcscf.CASSCF(m, 4, 4)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.91378640707609, 7)
    
    def test_mc2step_6o6e_fci(self):
        mc = mcscf.CASSCF(m, 6, 6)
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.98028859357791, 7)

    def test_mc2step_4o4e_fciqmc_wdipmom(self):
        #nelec is the number of electrons in the active space
        nelec = 4
        norb = 4
        mc = mcscf.CASSCF(m, norb, nelec)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000

        emc, e_ci, fcivec, casscf_mo = mc.mc2step()
        self.assertAlmostEqual(emc,-108.91378666934476, 7)

        # Calculate dipole moment in casscf_mo basis
        # First find the 1RDM in the full space
        one_pdm = fciqmc.find_full_casscf_12rdm(mc.fcisolver, casscf_mo, 
                'spinfree_TwoRDM.1', norb, nelec)[0]
        dipmom = fciqmc.calc_dipole(mol,casscf_mo,one_pdm)
        print('Dipole moments: ',dipmom)
        #self.assertAlmostEqual(dipmom[0],xxx,5)
        #self.assertAlmostEqual(dipmom[1],xxx,5)
        #self.assertAlmostEqual(dipmom[2],xxx,5)

    def test_mc2step_6o6e_fciqmc(self):
        mc = mcscf.CASSCF(m, 6, 6)
        mc.max_cycle_macro = 10
        mc.fcisolver = fciqmcscf.FCIQMCCI(mol)
        mc.fcisolver.RDMSamples = 5000
        emc = mc.mc2step()[0]
        self.assertAlmostEqual(emc,-108.98028859357791, 7)

if __name__ == "__main__":
    print("Full Tests for FCIQMC-CASSCF N2")
    unittest.main()

