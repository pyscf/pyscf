#!/usr/bin/env python

import unittest
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import fci

mol = gto.M(atom='Be 0 0 0; H .23 -1.1 0; H .23 1.1 0',
            symmetry=1, verbose=0)
m = scf.RHF(mol)
m.kernel()
norb = m.mo_energy.size
nelec = mol.nelectron

class KnowValues(unittest.TestCase):
    def test_symm_spin0(self):
        fs = fci.FCI(mol, m.mo_coeff)
        fs.wfnsym = 'B1'
        fs.nroots = 3
        e, c = fs.kernel()
        self.assertAlmostEqual(e[0], -19.286003160337, 9)
        self.assertAlmostEqual(e[1], -18.812177419921, 9)
        self.assertAlmostEqual(e[2], -18.786684534678, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[0], norb, nelec)[0], 0, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[1], norb, nelec)[0], 6, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[2], norb, nelec)[0], 0, 9)

    def test_symm_spin1(self):
        fs = fci.FCI(mol, m.mo_coeff, singlet=False)
        fs.wfnsym = 'B1'
        fs.nroots = 3
        e, c = fs.kernel()
        self.assertAlmostEqual(e[0], -19.303845373762, 9)
        self.assertAlmostEqual(e[1], -19.286003160337, 9)
        self.assertAlmostEqual(e[2], -19.014838825548, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[0], norb, nelec)[0], 2, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[1], norb, nelec)[0], 0, 9)
        self.assertAlmostEqual(fci.spin_op.spin_square0(c[2], norb, nelec)[0], 2, 9)


if __name__ == "__main__":
    print("Full Tests for init_guess")
    unittest.main()

