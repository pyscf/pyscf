#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf.symm import addons

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ],
    basis = 'cc-pvdz',
    symmetry = 1,
)

mf = scf.RHF(mol)
mf.scf()


class KnowValues(unittest.TestCase):
    def test_label_orb_symm(self):
        l = addons.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
        lab0 = ['A1', 'A1', 'B1', 'A1', 'B2', 'A1', 'B1', 'B1',
                'A1', 'A1', 'B2', 'B1', 'A1', 'A2', 'B2', 'A1',
                'B1', 'B1', 'A1', 'B2', 'A2', 'A1', 'A1', 'B1']
        self.assertEqual(l, lab0)

    def test_symmetrize_orb(self):
        c = addons.symmetrize_orb(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
        self.assertTrue(numpy.allclose(c, mf.mo_coeff))
        numpy.random.seed(1)
        c = addons.symmetrize_orb(mol, mol.irrep_name, mol.symm_orb,
                                  numpy.random.random((mf.mo_coeff.shape)))
        self.assertAlmostEqual(numpy.linalg.norm(c), 14.054399033261175)


if __name__ == "__main__":
    print("Full Tests for symm.addons")
    unittest.main()

