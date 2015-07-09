#!/usr/bin/env python
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


class KnowValues(unittest.TestCase):
    def test_dipoles_hfbasis(self):
        m = scf.RHF(mol)
        m.conv_tol = 1e-9
        m.scf()

        fciqmc_calc = fciqmcscf.FCIQMCCI(mol)
        fciqmc_calc.tau = 0.01
        fciqmc_calc.RDMSamples = 2000

        norb = m.mo_coeff.shape[1]
        e = fciqmcscf.fciqmc.run_standalone(fciqmc_calc, m.mo_coeff)
        dips = fciqmc_calc.dipoles(m.mo_coeff, fcivec=None, norb=norb, nelec=mol.nelectron)

        self.assertAlmostEqual(e,-7.787146064428100, 5)
        self.assertAlmostEqual(dips[0],0.0,7)
        self.assertAlmostEqual(dips[1],0.0,7)
        self.assertAlmostEqual(dips[2],1.85781390006,5)

    def test_dipoles_casscfbasis(self):
        m = scf.RHF(mol)
        m.conv_tol = 1e-9
        m.scf()

        mc = mcscf.CASSCF(m,5,4)    #There are only 6 orbitals, 4 electrons, so this is close to the whole space. 
                                    #However, the default behaviour seems to change for the whole space, and returns the mf mos.
        emc,e_ci,fcivec,casscf_mo = mc.mc2step(m.mo_coeff)[0]

        fciqmc_calc = fciqmcscf.FCIQMCCI(mol)
        fciqmc_calc.tau = 0.01
        fciqmc_calc.RDMSamples = 2000
        norb = mc.mo_coeff.shape[1]
        e = fciqmcscf.fciqmc.run_standalone(fciqmc_calc, casscf_mo)   #Run from CASSCF natural orbitals
        dips = fciqmc_calc.dipoles(casscf_mo, fcivec=None, norb=norb, nelec=mol.nelectron)


if __name__ == "__main__":
    print('Tests for dipole moments from standalone FCIQMC calculation in HF and natural orbital basis')
    unittest.main()
