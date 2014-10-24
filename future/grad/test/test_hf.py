#!/usr/bin/env python

import unittest
from pyscf import scf
from pyscf import gto
from pyscf.future import grad

mol = gto.Mole()
mol.verbose = 0
mol.output = None#"out_hf"

mol.atom.extend([
    [1   , (0. , 0.1, .817)],
    ["F" , (0. , 0. , 0.)], ])
mol.basis = {"H": '6-31g',
             "F": '6-31g',}
mol.build()

def finger(mat):
    return abs(mat).sum()

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_e(mol, rhf)), 7.9210392362911595, 9)

    def test_r_uhf(self):
        uhf = scf.dhf.UHF(mol)
        g = grad.dhf.UHF(uhf)
        self.assertAlmostEqual(finger(g.grad_e(mol, uhf)), 7.9216825870803245, 9)

    def test_nuclear_repulsion(self):
        rhf = scf.RHF(mol)
        g = grad.hf.RHF(rhf)
        self.assertAlmostEqual(finger(g.grad_nuc(mol)), 8.2887823210941249, 9)


if __name__ == "__main__":
    print("Full Tests for HF")
    unittest.main()
