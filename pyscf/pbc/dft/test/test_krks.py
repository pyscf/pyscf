#!/usr/bin/env python
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft

def build_cell(ngs):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.a = '''3.5668  0.      0.
                0.      3.5668  0.
                0.      0.      3.5668'''
    cell.gs = np.array([ngs,ngs,ngs])
    cell.atom ='''
C, 0.,  0.,  0.
C, 0.8917,  0.8917,  0.8917
C, 1.7834,  1.7834,  0.
C, 2.6751,  2.6751,  0.8917
C, 1.7834,  0.    ,  1.7834
C, 2.6751,  0.8917,  2.6751
C, 0.    ,  1.7834,  1.7834
C, 0.8917,  2.6751,  2.6751'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell

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
    def test_klda8_cubic_gamma(self):
        cell = build_cell(8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -44.892502703975893, 8)

    def test_klda8_cubic_kpt_222(self):
        cell = build_cell(8)
        abs_kpts = cell.make_kpts([2]*3, with_gamma_point=False)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.425834895129569, 8)

    def test_klda8_primitive_gamma(self):
        cell = make_primitive_cell(8)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -10.221426445656439, 8)

    def test_klda8_primitive_kpt_222(self):
        cell = make_primitive_cell(8)
        abs_kpts = cell.make_kpts([2]*3, with_gamma_point=False)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -11.353643583707452, 8)


if __name__ == '__main__':
    print("Full Tests for pbc.dft.krks")
    unittest.main()
