#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import lib

# for cgto
mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom.extend([[2, (0.,0.,0.)], ])
mol.basis = {"He": 'cc-pvdz'}
mol.build()

class KnowValues_NR(unittest.TestCase):
    """non-relativistic"""
    def test_fock_1e(self):
        rhf = scf.RHF(mol)
        h1e = rhf.get_hcore(mol)
        s1e = rhf.get_ovlp(mol)
        e, c = rhf.eig(h1e, s1e)
        self.assertAlmostEqual(e[0], -1.9936233377269388, 12)

    def test_init_guess(self):
        """NR HF"""
        rhf = scf.RHF(mol)
        with lib.quite_run():
            e = rhf._init_guess_by_atom(mol)[0]
        self.assertAlmostEqual(e * .5, -1.4275802386213701, 12)

    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.conv_threshold = 1e-10
        self.assertAlmostEqual(rhf.scf(), -2.8551604772427379, 10)

    def test_nr_uhf(self):
        uhf = scf.UHF(mol)
        uhf.conv_threshold = 1e-10
        self.assertAlmostEqual(uhf.scf(), -2.8551604772427379, 10)

#    def test_gaussian_nucmod(self):
#        gnuc = hf.gto.molinf.MoleInfo()
#        gnuc.verbose = 0
#        gnuc.output = "out_he"
#        gnuc.atom.extend([[2, (0.,0.,0.)], ])
#        gnuc.etb = {"He": { "max_l": 1, "s": (4, .4, 3.8), "p": (2, 1, 3.4)}}
#        gnuc.nucmod = {1:2}
#        gnuc.build()
#        rhf = scf.RHF(gnuc)
#        rhf.conv_threshold = 1e-10
#        rhf.potential("coulomb")
#        self.assertAlmostEqual(rhf.scf(), -2.8447211759894566, 10)
#        # restore nucmod
#        mol.nucmod = {1:1}
#        mol.build()


if __name__ == "__main__":
    print("Full Tests for He")
    unittest.main()

