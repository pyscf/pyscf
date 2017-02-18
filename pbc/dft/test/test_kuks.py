#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import ase.lattice
from ase.lattice.cubic import Diamond

LATTICE_CONST = 3.5668

class KnowValues(unittest.TestCase):
    def test_klda8_cubic_kpt_222(self):
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.a = ase_atom.cell
        cell.gs = np.array([8]*3)
        cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        kpts = cell.make_kpts((2,2,2), with_gamma_point=False)
        mf = pbcdft.KUKS(cell, kpts)
        mf.xc = 'lda,vwn'
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.42583489512954, 8)
        self.assertAlmostEqual(mf._ecoul, 3.2519161200384685, 8)
        self.assertAlmostEqual(mf._exc, -13.937886385300949, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.kuks")
    unittest.main()

