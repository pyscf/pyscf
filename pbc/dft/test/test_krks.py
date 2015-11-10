import unittest 
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import ase
import ase.lattice
import ase.dft.kpoints

LATTICE_CONST = 3.5668

def build_cell(ase_atom, ngs):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.h = ase_atom.cell
    cell.gs = np.array([ngs,ngs,ngs])

    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    #cell.verbose = 4
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_klda8_cubic_gamma(self):
        from ase.lattice.cubic import Diamond
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        cell = build_cell(ase_atom, 8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -44.8952124954005, 8)

    def test_klda8_cubic_kpt_222(self):
        from ase.lattice.cubic import Diamond
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
        cell = build_cell(ase_atom, 8)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.4292039673842, 8)

    def test_klda8_primitive_gamma(self):
        from ase.lattice import bulk
        ase_atom = bulk('C', 'diamond', a=LATTICE_CONST)
        cell = build_cell(ase_atom, 8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -10.2214263103746, 8)

    def test_klda8_primitive_kpt_222(self):
        from ase.lattice import bulk
        ase_atom = bulk('C', 'diamond', a=LATTICE_CONST)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
        cell = build_cell(ase_atom, 8)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -11.3536435234899, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.krks")
    unittest.main()
