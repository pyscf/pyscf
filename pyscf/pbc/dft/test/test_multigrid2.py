import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad

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

def tearDownModule():
    global cell, mf, mf1
    del cell, mf, mf1

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

    def test_orth_rks_grad_lda(self):
        mf1.xc = 'lda, vwn'
        mf1.kernel()
        grad = rks_grad.Gradients(mf1)
        g1 = grad.kernel()
        g0 = numpy.array([[-0.06563255,  0.02118291,  0.14407768],
                          [-0.06847428, -0.01214106, -0.11872132],
                          [ 0.13385968, -0.00708309, -0.0244801 ]])
        self.assertAlmostEqual(abs(g1-g0).max(), 0, 6)

    def test_orth_rks_grad_gga(self):
        mf1.xc = 'pbe,pbe'
        mf1.kernel()
        grad = rks_grad.Gradients(mf1)
        g1 = grad.kernel()
        g0 = numpy.array([[-0.06486347,  0.02139685,  0.14225973],
                          [-0.06763985, -0.01208953, -0.11806326],
                          [ 0.13279761, -0.00709116, -0.02470343]])
        self.assertAlmostEqual(abs(g1-g0).max(), 0, 6)

if __name__ == '__main__':
    print("Full Tests for multigrid2")
    unittest.main()
