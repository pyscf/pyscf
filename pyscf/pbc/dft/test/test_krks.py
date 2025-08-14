#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import scf as pbcscf


def build_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.a = '''3.5668  0.      0.
                0.      3.5668  0.
                0.      0.      3.5668'''
    cell.mesh = mesh
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

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    return cell


def setUpModule():
    global cell
    L = 4.
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.a = np.eye(3)*L
    cell.atom =[['He' , ( L/2+0., L/2+0. ,   L/2+1.)],]
    cell.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    cell.build()

def tearDownModule():
    global cell
    del cell


class KnownValues(unittest.TestCase):
    def test_klda(self):
        cell = pbcgto.M(atom='H 0 0 0; H 1 0 0', a=np.eye(3)*2, basis=[[0, [1, 1]]])
        cell.build()
        mf = cell.KRKS(kpts=cell.make_kpts([2,2,1]))
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -0.3846075202893169, 7)

    def test_klda8_cubic_gamma(self):
        cell = build_cell([17]*3)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        mf.conv_tol = 1e-8
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -44.892502703975893, 7)

    def test_klda8_cubic_kpt_222(self):
        cell = build_cell([17]*3)
        abs_kpts = cell.make_kpts([2]*3, with_gamma_point=False)
        mf = pbcdft.KRKS(cell, abs_kpts)
        mf.conv_tol=1e-9
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-8
        #mf.verbose = 7
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -45.425834895129569, 7)

    def test_klda8_primitive_gamma(self):
        cell = make_primitive_cell([17]*3)
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
        #kmf.verbose = 7
        mf.conv_tol = 1e-8
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -10.221426445656439, 7)

    def test_klda8_primitive_kpt_222(self):
        cell = make_primitive_cell([17]*3)
        abs_kpts = cell.make_kpts([2]*3, with_gamma_point=False)
        mf = pbcdft.KRKS(cell, abs_kpts)
        #mf.analytic_int = False
        mf.xc = 'lda,vwn'
        #mf.verbose = 7
        mf.conv_tol = 1e-8
        e1 = mf.scf()
        self.assertAlmostEqual(e1, -11.353643583707452, 7)

    def test_rsh_fft(self):
        mf = pbcdft.KRKS(cell)
        mf.xc = 'hse06'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.482418296326724, 7)

        mf.xc = 'camb3lyp'
        mf.conv_tol = 1e-8
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4745140703871877, 7)

    def test_rsh_df(self):
        mf = pbcdft.KRKS(cell).density_fit()
        mf.xc = 'wb97'
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4916945546399165, 6)

        mf.xc = 'camb3lyp'
        mf.omega = .15
        mf.conv_tol = 1e-8
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.4766238116030683, 5)

    def test_to_hf(self):
        mf = pbcdft.KRKS(cell).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.khf.KRHF))

        mf = pbcdft.KRKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(not a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.khf.KRHF))

        mf = pbcdft.KROKS(cell).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.krohf.KROHF))

        mf = pbcdft.KROKS(cell, kpts=cell.make_kpts([2,1,1])).density_fit()
        mf.with_df._j_only = True
        a_hf = mf.to_hf()
        self.assertTrue(not a_hf.with_df._j_only)
        self.assertTrue(isinstance(a_hf, pbcscf.krohf.KROHF))

    def test_reset(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-pade'
        cell.verbose = 7
        cell.output = '/dev/null'
        cell.build()
        kpts0 = cell.make_kpts([3,1,1])
        mf = cell.KRKS(kpts=kpts0)

        cell1 = pbcgto.Cell()
        cell1.atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95'
        cell1.a = '''0.   1.9  1.9
                     1.9  0.   1.9
                     1.9  1.9  0.    '''
        cell1.basis = 'gth-dzvp'
        cell1.pseudo = 'gth-pade'
        cell1.verbose = 7
        cell1.output = '/dev/null'
        cell1.build()
        mf.reset(cell1)
        assert abs(mf.kpts - kpts0).sum() > 0.1
        ref = cell1.make_kpts([3,1,1])
        assert abs(mf.kpts - ref).max() < 1e-9

        cell1.set_geom_(a='''0.   2.0  2.0
                             2.0  0.   2.0
                             2.0  2.0  0.    ''')
        ref = cell1.make_kpts([3,1,1])
        mf.reset(cell1)
        assert abs(mf.kpts - kpts0).sum() > 0.1
        assert abs(mf.kpts - ref).max() < 1e-9

    def test_reset_ksym(self):
        cell = pbcgto.Cell()
        cell.unit = 'A'
        cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
        cell.a = '''0.      1.7834  1.7834
                    1.7834  0.      1.7834
                    1.7834  1.7834  0.    '''
        cell.basis = 'gth-dzvp'
        cell.space_group_symmetry = True
        cell.pseudo = 'gth-pade'
        cell.verbose = 7
        cell.output = '/dev/null'
        cell.build()
        kpts0 = cell.make_kpts([3,1,1], space_group_symmetry=True)
        mf = pbcdft.KRKS(cell, kpts=kpts0)

        ref = pbcgto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
            a = '''0.   1.9  1.9
                   1.9  0.   1.9
                   1.9  1.9  0. ''',
            basis = 'gth-dzvp',
            space_group_symmetry = True,
            pseudo = 'gth-pade',
            verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
        cell.set_geom_(
            'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
            a = '''0.   1.9  1.9
                   1.9  0.   1.9
                   1.9  1.9  0. ''')
        mf.reset(cell)
        assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9

        ref = pbcgto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; C 0.95,  0.95,  0.95',
            a = '''0.   2.0  2.0
                   2.0  0.   2.0
                   2.0  2.0  0. ''',
            basis = 'gth-dzvp',
            space_group_symmetry = True,
            pseudo = 'gth-pade',
            verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
        cell.set_geom_(a='''0.   2.0  2.0
                            2.0  0.   2.0
                            2.0  2.0  0. ''')
        ref = cell.make_kpts([3,1,1], space_group_symmetry=True)
        mf.reset(cell)
        assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9

        ref = pbcgto.M(
            unit = 'A',
            atom = 'C 0.,  0.,  0.; C 1.,  1.,  1.',
            a = '''0.   2.0  2.0
                   2.0  0.   2.0
                   2.0  2.0  0. ''',
            basis = 'gth-dzvp',
            space_group_symmetry = True,
            pseudo = 'gth-pade',
            verbose = 0).make_kpts([3,1,1], space_group_symmetry=True)
        cell.set_geom_(
            'C 0.,  0.,  0.; C 1.,  1.,  1.')
        mf.reset(cell)
        assert abs(mf.kpts.kpts_ibz - ref.kpts_ibz).max() < 1e-9


if __name__ == '__main__':
    print("Full Tests for pbc.dft.krks")
    unittest.main()
