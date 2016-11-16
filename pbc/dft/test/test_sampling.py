#!/usr/bin/env python
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.tools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import ase
import ase.dft.kpoints
from ase.lattice import bulk

def make_primitive_cell(ngs):
    ase_atom = bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell.T

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    #cell.nimgs = np.array([7,7,7])
    cell.verbose = 0
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def xtest_gamma(self):
        cell = make_primitive_cell(8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -10.2214263103747, 8)

    def xtest_kpt_222(self):
        cell = make_primitive_cell(8)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        kmf = pbcdft.KRKS(cell, abs_kpts)
        kmf.xc = 'lda,vwn'
        #kmf.analytic_int = False
        #kmf.verbose = 7
        e1 = kmf.scf()
        self.assertAlmostEqual(e1, -11.3536435234900, 8)

    def test_kpt_vs_supercell(self):
        ngs = 5
        nk = (3, 1, 1)
        # Comparison is only perfect for odd-numbered supercells and kpt sampling
        assert all(np.array(nk) % 2 == np.array([1,1,1]))
        cell = make_primitive_cell(ngs)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack(nk)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        kmf = pbcdft.KRKS(cell, abs_kpts)
        kmf.xc = 'lda,vwn'
        #kmf.analytic_int = False
        #kmf.verbose = 7
        ekpt = kmf.scf()

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.gs = np.array([nk[0]*ngs + (nk[0]-1)//2,
                               nk[1]*ngs + (nk[1]-1)//2,
                               nk[2]*ngs + (nk[2]-1)//2])
        #supcell.verbose = 7
        supcell.build()

        mf = pbcdft.RKS(supcell)
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        esup = mf.scf()/np.prod(nk)

        #print("kpt sampling energy =", ekpt)
        #print("supercell energy    =", esup)
        self.assertAlmostEqual(ekpt, esup, 5)


if __name__ == '__main__':
    print("Full Tests for k-point sampling")
    unittest.main()

