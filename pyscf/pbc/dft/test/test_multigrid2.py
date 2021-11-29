import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad
from pyscf.pbc.grad import krks as krks_grad

cell = gto.Cell()
boxlen = 5.0
cell.a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom = """
O          1.84560        1.21649        1.10372
H          2.30941        1.30070        1.92953
H          0.91429        1.26674        1.28886
"""
cell.basis = 'gth-szv'
cell.ke_cutoff = 140  # in a.u.
cell.precision = 1e-6
cell.pseudo = 'gth-pade'
cell.verbose = 0
cell.build()

mf = dft.RKS(cell)
mf.with_df = multigrid.MultiGridFFTDF(cell)

mf1 = dft.RKS(cell)
mf1.with_df = multigrid.MultiGridFFTDF2(cell)
mf1.with_df.ngrids = 4
mf1.with_df.ke_ratio = 3.
mf1.with_df.rel_cutoff = 20.0


cell1 = gto.Cell()
cell1.atom = '''O  0 0 0
                O  0 0 1.2'''
cell1.a = [[8, 0, 0], [0, 8, 0], [0, 0, 8]]
cell1.basis = 'gth-szv'
cell1.pseudo = 'gth-pade'
cell1.ke_cutoff = 200
cell1.verbose = 0
cell1.build()


def tearDownModule():
    global cell, mf, mf1, cell1
    del cell, mf, mf1, cell1

class KnownValues(unittest.TestCase):
    def test_orth_rks_lda(self):
        mf.xc = "lda,vwn"
        e_ref = mf.kernel()
        mf1.xc = mf.xc
        e1 = mf1.kernel()
        self.assertAlmostEqual(abs(e_ref-e1).max(), 0, 6)

    def test_orth_rks_gga(self):
        mf.xc = "pbe,pbe"
        e_ref = mf.kernel()
        mf1.xc = mf.xc
        e1 = mf1.kernel()
        self.assertAlmostEqual(abs(e_ref-e1).max(), 0, 6)

    def test_orth_rks_grad(self):
        mf1 = dft.RKS(cell1)
        mf1.xc = 'lda,vwn'
        mf1.with_df = multigrid.MultiGridFFTDF2(cell1)
        mf1.with_df.ngrids = 4
        mf1.with_df.ke_ratio = 3.
        mf1.with_df.rel_cutoff = 20.0
        mf1.kernel()
        grad = rks_grad.Gradients(mf1)
        g1 = grad.kernel()
        g0 = numpy.array([[0, 0,  0.3681403965],
                          [0, 0, -0.3690848791]])
        self.assertAlmostEqual(abs(g1-g0).max(), 0, 6)

        mf1.xc = 'pbe,pbe'
        mf1.kernel()
        grad = rks_grad.Gradients(mf1)
        g1 = grad.kernel()
        g0 = numpy.array([[0, 0,  0.3813440274],
                          [0, 0, -0.3850131015]])
        self.assertAlmostEqual(abs(g1-g0).max(), 0, 6)

if __name__ == '__main__':
    print("Full Tests for multigrid2")
    unittest.main()
