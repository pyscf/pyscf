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

def finger(a):
    return np.dot(np.cos(np.arange(a.size)), a.ravel())

def make_primitive_cell(ngs):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    cell.verbose = 7
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

        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -11.221426249047617, 8)

        nk = (5, 1, 1)
        abs_kpts = cell.make_kpts(nk, wrap_around=True)
        kmf = khf.KRHF(cell, abs_kpts, exxdiv='vcut_sph')
        ekpt = kmf.scf()
        self.assertAlmostEqual(ekpt, -12.337299166550796, 8)

        supcell = pyscf.pbc.tools.super_cell(cell, nk)
        supcell.gs = np.array([nk[0]*ngs + (nk[0]-1)//2,
                               nk[1]*ngs + (nk[1]-1)//2,
                               nk[2]*ngs + (nk[2]-1)//2])
        #print "supcell gs =", supcell.gs
        supcell.build()

        gamma = [0,0,0]
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
        e1 = mf.kernel(dm1)
        mf.conv_check = False
        self.assertAlmostEqual(e1, ekpt, 9)

        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = khf.KRHF(cell, kpts, exxdiv='vcut_sph')
        dm = kmf1.from_chk(mf.chkfile)
        kmf1.max_cycle = 1
        ekpt = kmf1.scf(dm)
        kmf1.conv_check = False
        self.assertAlmostEqual(ekpt, -11.17814699669376, 8)

    def test_kuhf(self):
        ngs = 4
        cell = make_primitive_cell(ngs)
        nk = (3, 1, 1)
        kpts = cell.make_kpts(nk)
        kmf1 = kuhf.KUHF(cell, kpts, exxdiv='vcut_sph')
        ekpt = kmf1.scf()
        self.assertAlmostEqual(ekpt, -11.218735269838586, 8)
        np.random.seed(1)
        kpts_bands = np.random.random((2,3))
        e = kmf1.get_bands(kpts_bands)[0]
        self.assertAlmostEqual(finger(np.array(e)), -0.045547555445877741, 6)

if __name__ == '__main__':
    print("Full Tests for pbc.scf.khf")
    unittest.main()
