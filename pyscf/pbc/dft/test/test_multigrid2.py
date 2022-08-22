import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad

def setUpModule():
    global cell, mf1, mf2
    cell = gto.Cell()
    boxlen = 5.0
    cell.a = numpy.array([[boxlen,0.0,0.0],
                          [0.0,boxlen,0.0],
                          [0.0,0.0,boxlen]])
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
    cell.rcut_by_shell_radius = True
    cell.build()

    mf1 = dft.RKS(cell)
    mf1.with_df = multigrid.MultiGridFFTDF(cell)

    mf2 = dft.RKS(cell)
    mf2.with_df = multigrid.MultiGridFFTDF2(cell)

def tearDownModule():
    global cell, mf1, mf2
    del cell, mf1, mf2

def _test_energy_grad(mytest, xc, gref, eprec=6, gprec=6):
    mf1.xc = xc
    e1 = mf1.kernel()
    mf2.xc = xc
    e2 = mf2.kernel()
    mytest.assertAlmostEqual(abs(e1-e2), 0, eprec)
    
    g2 = rks_grad.Gradients(mf2).kernel()
    mytest.assertAlmostEqual(abs(g2-gref).max(), 0, gprec)

class KnownValues(unittest.TestCase):
    def test_orth_rks_lda(self):
        g0 = numpy.array([[-0.06562532,  0.02117777,  0.1440714 ],
                          [-0.06847938, -0.01213882, -0.11871832],
                          [ 0.13385757, -0.00708016, -0.02447684]])

        _test_energy_grad(self, "lda,vwn", g0, 6, 6)

    def test_orth_rks_gga(self):
        g0 = numpy.array([[-0.06485603,  0.02139166,  0.14225288],
                          [-0.06764492, -0.01208725, -0.11805988],
                          [ 0.13279518, -0.0070882,  -0.02469993]])
        _test_energy_grad(self, "pbe,pbe", g0, 6, 6)

if __name__ == '__main__':
    print("Full Tests for multigrid2")
    unittest.main()
