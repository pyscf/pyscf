#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import dft

cell = gto.Cell()
cell.unit = 'B'
cell.atom = '''
C  0.          0.          0.        
C  1.68506879  1.68506879  1.68506879
'''
cell.a = '''
0.      1.7834  1.7834
1.7834  0.      1.7834
1.7834  1.7834  0.    
'''

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.gs = [9]*3
cell.verbose = 5
cell.output = '/dev/null'
cell.build()

class KnowValues(unittest.TestCase):
    def test_nr_rhf(self):
        mf = scf.RHF(cell)
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.40296163946444, 9)

    def test_nr_rhf(self):
        kpts = cell.make_kpts([2,1,1,])
        mf = scf.RHF(cell)
        mf.kpt = kpts[1]
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -5.2608719097095538, 9)

    def test_nr_uhf(self):
        mf = scf.UHF(cell)
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.40296163946444, 9)

    def test_nr_rks_lda(self):
        mf = dft.RKS(cell)
        mf.xc = 'lda'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.5287158033530446, 9)

    def test_nr_uks_lda(self):
        mf = dft.UKS(cell)
        mf.xc = 'lda'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.5287158033530446, 9)

    def test_nr_rks_gga(self):
        mf = dft.RKS(cell)
        mf.xc = 'b88'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.7335981841364507, 9)

    def test_nr_uks_gga(self):
        mf = dft.UKS(cell)
        mf.xc = 'b88'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -1.7335981841364507, 9)

    def test_nr_krhf(self):
        mf = scf.KRHF(cell, cell.make_kpts([2,1,1]))
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -3.5815167460242954, 9)

    def test_nr_kuhf(self):
        mf = scf.KUHF(cell, cell.make_kpts([2,1,1]))
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -3.5815167460242954, 9)

    def test_nr_krks_lda(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'lda'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.9241968889649641, 9)

    def test_nr_kuks_lda(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'lda'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -2.9241968889649641, 9)

    def test_nr_krks_gga(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b88'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -3.0578675603376091, 9)

    def test_nr_kuks_gga(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b88'
        mf = scf.newton(mf)
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -3.0578675603376091, 9)

    def test_rks_gen_g_hop(self):
        mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp'
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
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 3335.9975942652204, 7)

    def test_uks_gen_g_hop(self):
        mf = dft.KUKS(cell, cell.make_kpts([2,1,1]))
        mf.xc = 'b3lyp'
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
        self.assertAlmostEqual(numpy.linalg.norm(hop(dm1)), 1253.8512425091715, 7)


if __name__ == "__main__":
    print("Full Tests for PBC Newton solver")
    unittest.main()

