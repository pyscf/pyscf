import unittest
import numpy
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import hf as pbchf
import pyscf.pbc.scf as pscf

L = 4
n = 10
cell = pbcgto.Cell()
cell.build(unit = 'B',
           verbose = 5,
           output = '/dev/null',
           h = ((L,0,0),(0,L,0),(0,0,L)),
           gs = [n,n,n],
           atom = [['He', (L/2.-.5,L/2.,L/2.-.5)],
                   ['He', (L/2.   ,L/2.,L/2.+.5)]],
           basis = { 'He': [[0, (0.8, 1.0)],
                            [0, (1.0, 1.0)],
                            [0, (1.2, 1.0)]]})

class KnowValues(unittest.TestCase):
    def test_rhf_vcut_sph(self):
        mf = pbchf.RHF(cell, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.1379172088570595, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='vcut_sph')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_ewald(self):
        mf = pbchf.RHF(cell, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.3511582284698633, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv='ewald')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='ewald')
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -4.2048655827967139, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv='ewald')
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_rhf_exx_None(self):
        mf = pbchf.RHF(cell, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.9325094887283196, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.double)

        mf = pscf.KRHF(cell, [[0,0,0]], exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv=None)
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, -2.7862168430230341, 9)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)

        mf = pscf.KRHF(cell, k, exxdiv=None)
        e0 = mf.kernel()
        self.assertTrue(numpy.allclose(e0,e1))

    def test_init_guess_by_chkfile(self):
        numpy.random.seed(1)
        k = numpy.random.random(3)
        mf = pbchf.RHF(cell, k, exxdiv='vcut_sph')
        e1 = mf.kernel()
        dm1 = mf.make_rdm1()

        mf1 = pbchf.RHF(cell, exxdiv='vcut_sph')
        mf1.chkfile = mf.chkfile
        mf1.init_guess = 'chkfile'
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -4.29190260870812, 9)
        self.assertTrue(mf1.mo_coeff.dtype == numpy.double)



if __name__ == '__main__':
    print("Full Tests for pbc.scf.hf")
    unittest.main()
