#!/usr/bin/env python

'''
k-point spin-restricted periodic MP2 calculation using the staggered mesh method
Author: Xin Xing (xxing@berkeley.edu)
Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids: Second-Order Møller–Plesset Perturbation Theory
           J. Chem. Theory Comput. 2021, 17, 8, 4733-4745
'''

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df
from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger




def run_kcell_fftdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    emf = kmf.scf()
    emp2_sub = KMP2_stagger(kmf, flag_submesh=True).run()
    emp2_ext = KMP2_stagger(kmf, flag_submesh=False).run()
    return emf, emp2_sub.e_corr, emp2_ext.e_corr

def run_kcell_gdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    gdf = df.GDF(cell, abs_kpts).build()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    emf = kmf.scf()
    emp2_sub = KMP2_stagger(kmf, flag_submesh=True).run()
    emp2_ext = KMP2_stagger(kmf, flag_submesh=False).run()
    return emf, emp2_sub.e_corr, emp2_ext.e_corr


def run_kcell_complex_fftdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    kmf.conv_tol = 1e-12
    emf = kmf.scf()
    kmf.mo_coeff = [kmf.mo_coeff[i].astype(np.complex128) for i in range(np.prod(nk))]
    emp2_sub = KMP2_stagger(kmf, flag_submesh=True).run()
    emp2_ext = KMP2_stagger(kmf, flag_submesh=False).run()
    return emf, emp2_sub.e_corr, emp2_ext.e_corr

def run_kcell_complex_gdf(cell, nk):
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    kmf = pbcscf.KRHF(cell, abs_kpts)
    gdf = df.GDF(cell, abs_kpts).build()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    emf = kmf.scf()
    kmf.mo_coeff = [kmf.mo_coeff[i].astype(np.complex128) for i in range(np.prod(nk))]
    emp2_sub = KMP2_stagger(kmf, flag_submesh=True).run()
    emp2_ext = KMP2_stagger(kmf, flag_submesh=False).run()
    return emf, emp2_sub.e_corr, emp2_ext.e_corr

class KnownValues(unittest.TestCase):
    def test_222_h2_fftdf(self):
        cell = pbcgto.Cell()
        cell.atom='''
            H 3.00   3.00   2.10
            H 3.00   3.00   3.90
            '''
        cell.a = '''
            6.0   0.0   0.0
            0.0   6.0   0.0
            0.0   0.0   6.0
            '''
        cell.unit = 'B'
        cell.pseudo = 'gth-pade'
        cell.basis = 'gth-szv'
        cell.verbose = 4
        cell.build()

        nk = [2,2,2]
        emf, emp2_sub, emp2_ext = run_kcell_fftdf(cell,nk)
        self.assertAlmostEqual(emf, -1.10046681450171, 9)
        self.assertAlmostEqual(emp2_sub, -0.0160900371069261, 9)
        self.assertAlmostEqual(emp2_ext, -0.0140288251933276, 9)

        emf, emp2_sub, emp2_ext = run_kcell_complex_fftdf(cell,nk)
        self.assertAlmostEqual(emp2_sub, -0.0160900371069261, 9)
        self.assertAlmostEqual(emp2_ext, -0.0140288251933276, 9)

    def test_222_h2_gdf(self):
        cell = pbcgto.Cell()
        cell.atom='''
            H 3.00   3.00   2.10
            H 3.00   3.00   3.90
            '''
        cell.a = '''
            6.0   0.0   0.0
            0.0   6.0   0.0
            0.0   0.0   6.0
            '''
        cell.unit = 'B'
        cell.pseudo = 'gth-pade'
        cell.basis = 'gth-szv'
        cell.verbose = 4
        cell.build()

        nk = [2,2,2]
        emf, emp2_sub, emp2_ext = run_kcell_gdf(cell,nk)
        self.assertAlmostEqual(emf, -1.10186079943922, 9)
        self.assertAlmostEqual(emp2_sub, -0.0158364523431077, 9)
        self.assertAlmostEqual(emp2_ext, -0.0140278627430396, 9)

        emf, emp2_sub, emp2_ext = run_kcell_complex_gdf(cell,nk)
        self.assertAlmostEqual(emp2_sub, -0.0158364523431077, 9)
        self.assertAlmostEqual(emp2_ext, -0.0140278627430396, 9)


    def test_222_diamond_frozen(self):
        cell = pbcgto.Cell()
        cell.pseudo = 'gth-pade'
        cell.basis = 'gth-szv'
        cell.ke_cutoff=100
        cell.atom='''
            C     0.      0.      0.
            C     1.26349729, 0.7294805 , 0.51582061
            '''
        cell.a = '''
            2.52699457, 0.        , 0.
            1.26349729, 2.18844149, 0.
            1.26349729, 0.7294805 , 2.06328243
            '''
        cell.unit = 'angstrom'
        cell.verbose = 4
        cell.build()

        nk = [2,2,2]
        abs_kpts = cell.make_kpts(nk, wrap_around=True)

        #   FFTDF-based calculation
        kmf = pbcscf.KRHF(cell, abs_kpts)
        kmf.conv_tol = 1e-12
        emf = kmf.scf()
        emp2_sub = KMP2_stagger(kmf, flag_submesh=True, frozen=[0,1,2]).run()
        emp2_ext = KMP2_stagger(kmf, flag_submesh=False, frozen=[0,1,2]).run()

        self.assertAlmostEqual(emp2_sub.e_corr, -0.0254955913664726, 9)
        self.assertAlmostEqual(emp2_ext.e_corr, -0.0126977970896905, 9)

        #   GDF-based calculation
        kmf = pbcscf.KRHF(cell, abs_kpts)
        gdf = df.GDF(cell, abs_kpts).build()
        kmf.with_df = gdf       
        kmf.conv_tol = 1e-12
        emf = kmf.scf()
        emp2_sub = KMP2_stagger(kmf, flag_submesh=True, frozen=[0,1,2]).run()
        emp2_ext = KMP2_stagger(kmf, flag_submesh=False, frozen=[0,1,2]).run()

        self.assertAlmostEqual(emp2_sub.e_corr, -0.0252835750365586, 9)
        self.assertAlmostEqual(emp2_ext.e_corr, -0.0126846178079962, 9)


if __name__ == '__main__':
    print("Staggered KMP2 energy calculation test")
    unittest.main()

