#!/usr/bin/env python
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import kuhf
import pyscf.pbc.tools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import ase
import ase.lattice
import ase.dft.kpoints

def make_primitive_cell(ngs):
    from ase.lattice import bulk
    ase_atom = ase.build.bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()
    return cell

class KnowValues(unittest.TestCase):
    def test_kpt_vs_supercell(self):
        # For large ngs, agreement is always achieved
        # ngs = 8
        # For small ngs, agreement only achieved if "wrapping" k-k'+G in get_coulG
        ngs = 4
        nk = (3, 1, 1)
        cell = make_primitive_cell(ngs)

        scaled_kpts = ase.dft.kpoints.monkhorst_pack(nk)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -11.221426249047617, 8)

        nk = (5, 1, 1)
        scaled_kpts = ase.dft.kpoints.monkhorst_pack(nk)
        abs_kpts = cell.get_abs_kpts(scaled_kpts)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -12.337299038604856, 8)

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.gs = np.array([nk[0]*ngs + (nk[0]-1)//2,
                               nk[1]*ngs + (nk[1]-1)//2,
                               nk[2]*ngs + (nk[2]-1)//2])
        #print "supcell gs =", supcell.gs
        supcell.build()

        scaled_gamma = ase.dft.kpoints.monkhorst_pack((1,1,1))
        gamma = supcell.get_abs_kpts(scaled_gamma)
        mf = khf.KRHF(supcell, gamma, exxdiv='vcut_sph')
        esup = mf.scf()/np.prod(nk)

        #print "kpt sampling energy =", ekpt
        #print "supercell energy    =", esup
        #print "difference          =", ekpt-esup
        self.assertAlmostEqual(ekpt, esup, 6)

    def test_init_guess_by_chkfile(self):
        ngs = 4
        nk = (1, 1, 1)
        cell = make_primitive_cell(ngs)

        kpts = cell.make_kpts(nk)
        kmf = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        dm1 = kmf.make_rdm1()
        dm2 = kmf.from_chk(kmf.chkfile)
        self.assertTrue(dm2.dtype == np.double)
        self.assertTrue(np.allclose(dm1, dm2))

        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf.chkfile = kmf.chkfile
        mf.init_guess = 'chkfile'
        dm1 = mf.from_chk(kmf.chkfile)
        mf.max_cycle = 1
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, ekpt, 9)

        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        dm = kmf1.from_chk(mf.chkfile)
        kmf1.max_cycle = 1
        ekpt = kmf1.scf(dm)
        self.assertAlmostEqual(ekpt, -11.17814699669376, 8)

    def test_kuhf(self):
        ngs = 4
        cell = make_primitive_cell(ngs)
        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = kuhf.KUHF(cell, kpts, exxdiv='vcut_sph')
        ekpt = kmf1.scf()
        self.assertAlmostEqual(ekpt, -11.218735269838586, 8)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf")
    unittest.main()
