#!/usr/bin/env python
#
# Authors: James D. McClain <jmcclain@princeton.edu>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

#import unittest
import numpy as np
import unittest

from pyscf import gto, scf

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.ao2mo import eris
import pyscf.pbc.tools

import pyscf.pbc.cc

import ase
import ase.lattice
import ase.dft.kpoints
import make_test_cell

def run_cell(cell, ngs, nk):
    #############################################
    # Do a supercell Gamma-pt calculation       #
    #############################################
    supcell = pyscf.pbc.tools.super_cell(cell, nk)
    supcell.gs = np.array([nk[0]*ngs + 1*(nk[0]-1)//2,
                           nk[1]*ngs + 1*(nk[1]-1)//2,
                           nk[2]*ngs + 1*(nk[2]-1)//2])
#    print "supcell gs =", supcell.gs
    supcell.build()


    scaled_gamma = ase.dft.kpoints.monkhorst_pack((1,1,1))
    gamma = supcell.get_abs_kpts(scaled_gamma)

    #############################################
    # Running HF                                #
    #############################################
#    print ""
#    print "*********************************"
#    print "STARTING HF                      "
#    print "*********************************"
#    print ""

    mf = pbchf.RHF(supcell, exxdiv=None)
    mf.conv_tol = 1e-14
    #mf.verbose = 7
    escf = mf.scf()
    escf_per_cell = escf/np.prod(nk)
#    print "scf energy (per unit cell) = %.17g" % escf_per_cell

    #############################################
    # Running CCSD                              #
    #############################################
#    print ""
#    print "*********************************"
#    print "STARTING CCSD                    "
#    print "*********************************"
#    print ""
    cc = pyscf.pbc.cc.CCSD(mf)
    cc.conv_tol=1e-8
    #cc.verbose = 7
    ecc, t1, it2 = cc.kernel()
    ecc_per_cell = ecc/np.prod(nk)
#    print "cc energy (per unit cell) = %.17g" % ecc_per_cell
    return escf_per_cell, ecc_per_cell

class KnowValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        ngs = 5
        cell = make_test_cell.test_cell_n0(L,ngs)
        #print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.73491491306419987
        cc_111 = -1.1580008204825658e-05
        escf,ecc=run_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_111_n1(self):
        L = 7.0
        ngs = 4
        cell = make_test_cell.test_cell_n1(L,ngs)
        #print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.73506011036963814
        cc_111 = -0.023265431169472835
        escf,ecc=run_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_111_n3(self):
        L = 10.0
        ngs = 5
        cell = make_test_cell.test_cell_n3(ngs)
        #print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -7.4117951240232118
        cc_111 = -0.19468901057053406
        escf,ecc=run_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 6)

    def test_311_n1(self):
        L = 7.0
        ngs = 4
        cell = make_test_cell.test_cell_n1(L,ngs)
        #print "cell gs =", cell.gs
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf,ecc=run_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

