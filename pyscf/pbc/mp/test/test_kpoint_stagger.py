#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Test code for
k-point spin-restricted periodic MP2 calculation using the staggered mesh method
Author: Xin Xing (xxing@berkeley.edu)
Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids: Second-Order
        Møller–Plesset Perturbation Theory, J. Chem. Theory Comput. 2021, 17, 8, 4733-4745
'''

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df
from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger

def build_h2_fftdf_cell():
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
    cell.mesh = [12]*3
    cell.verbose = 4
    cell.output = '/dev/null'
    cell.build()
    return cell

def build_h2_gdf_cell():
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
    cell.basis = [[0, [1.0, 1]],]
    cell.pseudo = 'gth-pade'
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()
    return cell

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
    def test_222_h2_fftdf_high_cost(self):
        cell = build_h2_fftdf_cell()
        nk = [2,2,2]
        emf, emp2_sub, emp2_ext = run_kcell_fftdf(cell,nk)
        self.assertAlmostEqual(emf, -1.1097111706856706, 7)
        self.assertAlmostEqual(emp2_sub, -0.016715452146525603, 7)
        self.assertAlmostEqual(emp2_ext, -0.01449245927191392, 7)

        emf, emp2_sub, emp2_ext = run_kcell_complex_fftdf(cell,nk)
        self.assertAlmostEqual(emp2_sub, -0.016715452146525596, 7)
        self.assertAlmostEqual(emp2_ext, -0.014492459271913925, 7)

    def test_222_h2_gdf(self):
        cell = build_h2_gdf_cell()
        nk = [2,2,2]
        emf, emp2_sub, emp2_ext = run_kcell_gdf(cell,nk)
        self.assertAlmostEqual(emf, -0.40854731697431584, 7)
        self.assertAlmostEqual(emp2_sub, -0.04014371773827328, 7)
        self.assertAlmostEqual(emp2_ext, -0.04043697990545155, 7)

        emf, emp2_sub, emp2_ext = run_kcell_complex_gdf(cell,nk)
        self.assertAlmostEqual(emp2_sub, -0.04014371773827323, 7)
        self.assertAlmostEqual(emp2_ext, -0.040436979905451524, 7)

    def test_222_diamond_frozen_high_cost(self):
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
        kmf.scf()
        emp2_sub = KMP2_stagger(kmf, flag_submesh=True, frozen=[0,1,2]).run()
        emp2_ext = KMP2_stagger(kmf, flag_submesh=False, frozen=[0,1,2]).run()

        self.assertAlmostEqual(emp2_sub.e_corr, -0.0254955913664726, 7)
        self.assertAlmostEqual(emp2_ext.e_corr, -0.0126977970896905, 7)

        #   GDF-based calculation
        kmf = pbcscf.KRHF(cell, abs_kpts).density_fit()
        kmf.with_df._prefer_ccdf = True
        kmf.conv_tol = 1e-12
        kmf.scf()
        emp2_sub = KMP2_stagger(kmf, flag_submesh=True, frozen=[0,1,2]).run()
        emp2_ext = KMP2_stagger(kmf, flag_submesh=False, frozen=[0,1,2]).run()

        self.assertAlmostEqual(emp2_sub.e_corr, -0.0252835750365586, 7)
        self.assertAlmostEqual(emp2_ext.e_corr, -0.0126846178079962, 7)

if __name__ == '__main__':
    print("Staggered KMP2 energy calculation test")
    unittest.main()
