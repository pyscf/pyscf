import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import ase.lattice

LATTICE_CONST = 3.5668

class KnowValues(unittest.TestCase):
    def test_klda8_cubic_kpt_222(self):
        from ase.lattice.cubic import Diamond
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.h = ase_atom.cell
        cell.gs = np.array([8]*3)
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        kpts = pyscf_ase.make_kpts(cell, (2,2,2))
        mf = pbcdft.KUKS(cell, kpts)
        mf.xc = 'lda,vwn'
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.4260923810061, 8)
        self.assertAlmostEqual(mf._ecoul, 3.2519841629129687, 8)
        self.assertAlmostEqual(mf._exc, -13.937917984479245, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kuks")
    unittest.main()

