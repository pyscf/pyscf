#!/usr/bin/env python

from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
import pyscf.symm

mol = gto.Mole()
mol.verbose = 0
mol.atom = '''
    O    0.  0.      0.
    H    0.  -0.757  0.587
    H    0.  0.757   0.587'''
mol.basis = 'sto-3g'
mol.symmetry = 1
mol.build()
m = scf.RHF(mol)
m.conv_tol = 1e-15
ehf = m.scf()
norb = m.mo_coeff.shape[1]
nelec = mol.nelectron
h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
g2e = ao2mo.incore.full(m._eri, m.mo_coeff)
orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
cis = fci.direct_spin1_symm.FCISolver(mol)
cis.orbsym = orbsym

numpy.random.seed(15)
na = fci.cistring.num_strings(norb, nelec//2)
ci0 = numpy.random.random((na,na))
ci0 = ci0
ci0 /= numpy.linalg.norm(ci0)

class KnowValues(unittest.TestCase):
    def test_contract(self):
        ci1 = cis.contract_2e(g2e, ci0, norb, nelec)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.3111226275705418, 10)

    def test_kernel(self):
        e, c = fci.direct_spin1_symm.kernel(h1e, g2e, norb, nelec, orbsym=orbsym)
        self.assertAlmostEqual(e, -84.200905534209554, 8)
        e = fci.direct_spin1_symm.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -84.200905534209554, 8)


if __name__ == "__main__":
    print("Full Tests for spin1-symm")
    unittest.main()



