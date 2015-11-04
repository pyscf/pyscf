#!/usr/bin/env python

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf.fciqmcscf.fciqmc import *

b = 1.4
mol = gto.Mole()

mol.build(
        verbose = 5,
#output = 'casscf.out',
output = None,
atom = [['Li',(  0.000000,  0.000000, 1.005436697)],
        ['H',(  0.000000,  0.000000,  0.0)]],
basis = {'H': 'sto-3g', 'Li': 'sto-3g'},
symmetry = True,
symmetry_subgroup = 'C2v',
)

m = scf.RHF(mol)
m.conv_tol = 1e-9
m.scf()

class KnowValues(unittest.TestCase):
    def test_dipoles_hfbasis(self):
        fciqmcci = FCIQMCCI(mol)
        fciqmcci.tau = 0.01
        fciqmcci.RDMSamples = 2000

        norb = m.mo_coeff.shape[1]
        energy = run_standalone(fciqmcci, m.mo_coeff)

        s = m.get_ovlp()

#We need to create a function to read in the spin-2RDMs (see eg read_neci_two_pdm and related functions)

#To convert one-electron matrix into AO basis (see calc_dipole for other way around)
        reduce(numpy.dot, (m.mo_coeff, matrix, m.mo_coeff.T))
#To do this for a two-electron matrix, then do this one at a time for each index...
        

#Other things we might find useful:
        eriref = numpy.einsum('pjkl,pi->ijkl', eriref, mo)
        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
        eriref = numpy.einsum('ijpl,pk->ijkl', eriref, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)
#Though later on we will want to use the ao2mo.incore routines

        numpy.zeros((mol.natm, 9), dtype=int)   (Create numpy array of length the number of atoms)
        mol.bas_atom(ib)    (Atom number)       
        mol.bas_nctr(ib)    

    def atom_nshells(self, atm_id):
        r'''Number of basis/shells of the given atom

        Args:
            atm_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1')
        >>> mol.atom_nshells(1)
        5
        '''
        symb = self.atom_symbol(atm_id)
        return len(self._basis[symb])

    def bas_atom(self, bas_id):
        r'''The atom (0-based id) that the given basis sits on

        Args:
            bas_id : int
                0-based

        Examples:

        >>> mol.build(atom='H 0 0 0; Cl 0 0 1.1', basis='cc-pvdz')
        >>> mol.bas_atom(7)
        1
        '''
        return self._bas[bas_id,ATOM_OF]




        two_pdm = read_neci_two_pdm('spinfree_TwoRDM.1', norb)
        one_pdm = one_from_two_pdm(two_pdm, mol.nelectron)
        dips, elec, nuc = calc_dipole(mol, m.mo_coeff, one_pdm)

        self.assertAlmostEqual(energy, -7.787146064428100, 5)
        self.assertAlmostEqual(dips[0], 0.0, 7)
        self.assertAlmostEqual(dips[1], 0.0, 7)
        self.assertAlmostEqual(dips[2], 1.85781390006, 4)

    def test_dipoles_casscfbasis(self):

        # There are only 6 orbitals and 4 electrons, so this is the full
        # space, giving the exact NO basis.
        mc = mcscf.CASSCF(m,6,4)
        # Ensures that casscf_mo returns the natural orbital basis in the
        # active space.
        mc.natorb = True
        emc, e_ci, fcivec, casscf_mo = mc.mc2step(m.mo_coeff)

        fciqmcci = FCIQMCCI(mol)
        fciqmcci.tau = 0.01
        fciqmcci.RDMSamples = 2000
        norb = mc.mo_coeff.shape[1]
        # Run from CASSCF natural orbitals
        energy = run_standalone(fciqmcci, casscf_mo)
        two_pdm = read_neci_two_pdm('spinfree_TwoRDM.1', norb)
        one_pdm = one_from_two_pdm(two_pdm, mol.nelectron)
        dips, elec, nuc = calc_dipole(mol, mc.mo_coeff, one_pdm)

        self.assertAlmostEqual(energy, -7.787146064428100, 5)
        self.assertAlmostEqual(dips[0], 0.0, 7)
        self.assertAlmostEqual(dips[1], 0.0, 7)
        self.assertAlmostEqual(dips[2], 1.85781390006, 4)

if __name__ == "__main__":
    print('Tests for dipole moments from standalone FCIQMC calculation in HF '
          'and natural orbital basis sets.')
    unittest.main()
