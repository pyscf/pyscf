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
#    print "scaled kpts"
#    print scaled_kpts
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
#    print "abs kpts"
#    print abs_kpts

    #############################################
    # Running HF                                #
    #############################################
#    print ""
#    print "*********************************"
#    print "STARTING HF                      "
#    print "*********************************"
#    print ""
    kmf = pbchf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    #kmf.verbose = 7
    ekpt = kmf.scf()
#    print "scf energy (per unit cell) = %.17g" % ekpt

    #############################################
    # Running CCSD                              #
    #############################################
#    print ""
#    print "*********************************"
#    print "STARTING CCSD                    "
#    print "*********************************"
#    print ""

    cc = pyscf.pbc.cc.kccsd_rhf.RCCSD(kmf)
    cc.conv_tol=1e-8
    cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
#    print "cc energy (per unit cell) = %.17g" % ecc
    return ekpt, ecc

class KnowValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        ngs = 5
        cell = test_make_cell.test_cell_n0(L,ngs)
#        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.73491491306419987
        cc_111 = -1.1580008204825658e-05
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
#        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.73506011036963814
        cc_111 = -0.023265431169472835
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n3(self):
        ngs = 5
        cell = test_make_cell.test_cell_n3(ngs)
#        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -7.4117951240232118
        cc_111 = -0.19468901057053406
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_311_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
        print "cell gs =", cell.gs
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = test_kcell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

