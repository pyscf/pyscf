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
import pyscf.pbc.cc.kccsd

import make_test_cell

def run_kcell(cell, n, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    abs_kpts = cell.make_kpts(nk, wrap_around=True)

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

    cc = pyscf.pbc.cc.kccsd.CCSD(pbchf.addons.convert_to_ghf(kmf))
    cc.conv_tol=1e-8
    #cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
#    print "cc energy (per unit cell) = %.17g" % ecc
    return ekpt, ecc

class KnowValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        n = 11
        cell = make_test_cell.test_cell_n0(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73491491306419987
        cc_111 = -1.1580008204825658e-05
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n1(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73506011036963814
        cc_111 = -0.023265431169472835
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n3(self):
        n = 11
        cell = make_test_cell.test_cell_n3([n]*3)
        nk = (1, 1, 1)
        hf_111 = -7.4117951240232118
        cc_111 = -0.19468901057053406
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,9)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_311_n1(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

