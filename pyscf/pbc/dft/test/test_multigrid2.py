import unittest
import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid

cell = gto.Cell()
boxlen = 5.0
cell.a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom = """
O          1.84560        1.21649        1.10372
H          2.30941        1.30070        1.92953
H          0.91429        1.26674        1.28886
"""
cell.basis = 'gth-szv'
cell.ke_cutoff = 350  # in a.u.
cell.precision = 1e-6
cell.pseudo = 'gth-pade'
cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_orth_rks(self):
        mf = dft.RKS(cell)
        mf.xc = "pbe,pbe"
        mf.with_df = multigrid.MultiGridFFTDF(cell)
        e_ref = mf.kernel()

        mf.with_df = multigrid.MultiGridFFTDF2(cell)
        mf.with_df.ngrids = 4
        mf.with_df.ke_ratio = 3.
        mf.with_df.rel_cutoff = 20.0
        e1 = mf.kernel()
        self.assertAlmostEqual(abs(e_ref-e1).max(), 0, 6)

if __name__ == '__main__':
    print("Full Tests for multigrid2")
    unittest.main()
