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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import dft

def setUpModule():
    global cell
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [19]*3
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.build()


def tearDownModule():
    global cell
    cell.stdout.close()
    del cell

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(cell)
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.137043711032916, 8)

    def test_nr_rhf_k1(self):
        kpts = cell.make_kpts([2,1,1,])
        mf = scf.RHF(cell)
        mf.kpt = kpts[1]
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -11.336879498930173, 8)

    def test_nr_uhf(self):
        mf = scf.UHF(cell)
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.137043711032916, 8)

    def test_nr_rohf(self):
        mf = scf.ROHF(cell).newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.137043711032916, 8)

    def test_nr_rks_lda(self):
        mf = dft.RKS(cell)
        mf.xc = 'lda,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -9.7670882971475663, 8)

    def test_nr_uks_lda(self):
        mf = dft.UKS(cell)
        mf.xc = 'lda,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -9.7670882971475663, 8)

    def test_nr_rks_gga(self):
        mf = dft.RKS(cell)
        mf.xc = 'b88,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -9.9355341416893559, 8)

    def test_nr_uks_gga(self):
        mf = dft.UKS(cell)
        mf.xc = 'b88,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -9.9355341416893559, 8)

    def test_nr_uks_rsh(self):
        mf = dft.UKS(cell)
        mf.xc = 'camb3lyp'
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.346325245581905, 8)

    def test_nr_krhf(self):
        mf = scf.KRHF(cell, cell.make_kpts([2,1,1]))
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.5309059210831, 7)

    def test_nr_kuhf(self):
        mf = scf.KUHF(cell, cell.make_kpts([2,1,1]))
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.5309059210831, 7)

    def test_nr_krohf(self):
        mf = scf.KROHF(cell, cell.make_kpts([2,1,1])).newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.5309059210831, 7)

    def test_nr_krks_lda(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'lda,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.307756038726733, 8)

    def test_nr_kuks_lda(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'lda,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.307756038726733, 8)

    def test_nr_krks_gga(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b88,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.446717855794008, 8)

    def test_nr_kuks_gga(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b88,'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.446717855794008, 8)

    def test_nr_krks_rsh(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'camb3lyp'
        mf = mf.newton()
        mf.conv_tol_grad = 1e-4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -10.810601144955786, 8)

    def test_rks_gen_g_hop(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp5'
        nao = cell.nao_nr()
        numpy.random.seed(1)
        mo = numpy.random.random((2,nao,nao)) + 0j
        mo_occ = numpy.zeros((2,nao))
        mo_occ[:,:5] = 2
        nocc, nvir = 5, nao-5
        dm1 = numpy.random.random(2*nvir*nocc) + .1j
        mf = scf.newton(mf)
        mf.grids.build()
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, mf.get_hcore())
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 37.967972033738519, 6)

    def test_uks_gen_g_hop(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp5'
        nao = cell.nao_nr()
        numpy.random.seed(1)
        mo = numpy.random.random((2,2,nao,nao)) + 0j
        mo_occ = numpy.zeros((2,2,nao))
        mo_occ[:,:,:5] = 1
        nocc, nvir = 5, nao-5
        dm1 = numpy.random.random(4*nvir*nocc) + .1j
        mf = scf.newton(mf)
        mf.grids.build()
        g, hop, hdiag = mf.gen_g_hop(mo, mo_occ, [mf.get_hcore()]*2)
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 28.01954683540594, 6)


if __name__ == "__main__":
    print("Full Tests for PBC Newton solver")
    unittest.main()
