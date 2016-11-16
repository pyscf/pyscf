#!/usr/bin/env python
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import ase
import ase.lattice
import ase.dft.kpoints
from ase.lattice.cubic import Diamond
from ase.lattice import bulk

LATTICE_CONST = 3.5668

def build_cell(ase_atom, ngs):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.a = ase_atom.cell
    cell.gs = np.array([ngs,ngs,ngs])

    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 0
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_klda8_cubic_gamma(self):
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        cell = build_cell(ase_atom, 8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        #print "mf._ecoul =", mf._ecoul
        #print "mf._exc =", mf._exc
        self.assertAlmostEqual(e1, -44.892502703975893, 8)

    def test_klda8_cubic_kpt_222(self):
        ase_atom = Diamond(symbol='C', latticeconstant=LATTICE_CONST)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
        cell = build_cell(ase_atom, 8)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        #print "mf._ecoul =", mf._ecoul
        #print "mf._exc =", mf._exc
        self.assertAlmostEqual(e1, -45.425834895129569, 8)

    def test_klda8_primitive_gamma(self):
        ase_atom = bulk('C', 'diamond', a=LATTICE_CONST)
        cell = build_cell(ase_atom, 8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        #print "mf._ecoul =", mf._ecoul
        #print "mf._exc =", mf._exc
        self.assertAlmostEqual(e1, -10.221426938778345, 8)

    def test_klda8_primitive_kpt_222(self):
        ase_atom = bulk('C', 'diamond', a=LATTICE_CONST)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
        cell = build_cell(ase_atom, 8)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        #print "mf._ecoul =", mf._ecoul
        #print "mf._exc =", mf._exc
        self.assertAlmostEqual(e1, -11.353643738291005, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.krks")
    unittest.main()
