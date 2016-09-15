#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

#import unittest
import unittest
import numpy as np

from pyscf import gto, scf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.ao2mo import eris
import pyscf.pbc.tools

import pyscf.pbc.cc
import pyscf.pbc.cc.kccsd_rhf

import ase
import ase.lattice
import ase.dft.kpoints
import test_make_cell

def test_kcell(cell, ngs, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    scaled_kpts = ase.dft.kpoints.monkhorst_pack(nk)
    print "scaled kpts"
    print scaled_kpts
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
    print "abs kpts"
    print abs_kpts

    #############################################
    # Running HF                                #
    #############################################
    print ""
    print "*********************************"
    print "STARTING HF                      "
    print "*********************************"
    print ""
    kmf = pbchf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-15
    #kmf.verbose = 7
    ekpt = kmf.scf()
    print "scf energy (per unit cell) = %.17g" % ekpt

    #############################################
    # Running CCSD                              #
    #############################################
    print ""
    print "*********************************"
    print "STARTING CCSD                    "
    print "*********************************"
    print ""

    cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf,abs_kpts)
    cc.conv_tol=1e-15
    cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
    print "cc energy (per unit cell) = %.17g" % ecc
    return ekpt, ecc

class KnowValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        ngs = 5
        cell = test_make_cell.test_cell_n0(L,ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.70044359250909261
        cc_111 = -0.00010678573554735512
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc,cc_111,9)

    def test_111_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.70854822118080063
        cc_111 = -0.024826472252138673
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc,cc_111,9)

    def test_111_n3(self):
        ngs = 5
        cell = test_make_cell.test_cell_n3(ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -7.4057541401768008
        cc_111 = -0.19455094965791886
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc,cc_111,9)

    def test_311_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
        print "cell gs =", cell.gs
        nk = (3, 1, 1)
        hf_311 = -0.89797412589365655
        cc_311 = -0.045952606749078105
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 9)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

