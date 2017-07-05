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
import test_make_cell

def test_cell(cell, ngs, nk):
    #############################################
    # Do a supercell Gamma-pt calculation       #
    #############################################
    supcell = pyscf.pbc.tools.super_cell(cell, nk)
    supcell.gs = np.array([nk[0]*ngs + 1*(nk[0]-1)//2,
                           nk[1]*ngs + 1*(nk[1]-1)//2,
                           nk[2]*ngs + 1*(nk[2]-1)//2])
    print "supcell gs =", supcell.gs
    supcell.build()


    scaled_gamma = ase.dft.kpoints.monkhorst_pack((1,1,1))
    gamma = supcell.get_abs_kpts(scaled_gamma)

    #############################################
    # Running HF                                #
    #############################################
    print ""
    print "*********************************"
    print "STARTING HF                      "
    print "*********************************"
    print ""

    mf = pbchf.RHF(supcell, exxdiv=None)
    mf.conv_tol = 1e-15
    #mf.verbose = 7
    escf = mf.scf()
    escf_per_cell = escf/np.prod(nk)
    print "scf energy (per unit cell) = %.17g" % escf_per_cell

    #############################################
    # Running CCSD                              #
    #############################################
    print ""
    print "*********************************"
    print "STARTING CCSD                    "
    print "*********************************"
    print ""
    cc = pyscf.pbc.cc.CCSD(mf)
    cc.conv_tol=1e-15
    #cc.verbose = 7
    ecc, t1, it2 = cc.kernel()
    ecc_per_cell = ecc/np.prod(nk)
    print "cc energy (per unit cell) = %.17g" % ecc_per_cell
    return escf_per_cell, ecc_per_cell

class KnowValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        ngs = 5
        cell = test_make_cell.test_cell_n0(L,ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.70044359250909261
        cc_111 = -0.00010678573554735347
        escf,ecc=test_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 9)

    def test_111_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -0.70854822118080063
        cc_111 = -0.024826472252138659
        escf,ecc=test_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 9)

    def test_111_n3(self):
        L = 10.0
        ngs = 5
        cell = test_make_cell.test_cell_n3(ngs)
        print "cell gs =", cell.gs
        nk = (1, 1, 1)
        hf_111 = -7.4057541401768008
        cc_111 = -0.1945509496579045
        escf,ecc=test_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_111, 9)
        self.assertAlmostEqual(ecc, cc_111, 9)


    def test_311_n1(self):
        L = 7.0
        ngs = 4
        cell = test_make_cell.test_cell_n1(L,ngs)
        print "cell gs =", cell.gs
        nk = (3, 1, 1)
        hf_311 =-0.89797412589365733
        cc_311 =-0.045952606748993201
        test_cell(cell,ngs,nk)
        escf,ecc=test_cell(cell,ngs,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 9)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()

