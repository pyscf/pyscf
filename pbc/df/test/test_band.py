import unittest
import numpy
from pyscf.pbc import gto, scf, df

cell = gto.M(atom='H 1 2 1; H 1 1 1', a=numpy.eye(3)*4, verbose=0, gs=[5]*3)
numpy.random.seed(1)
kband = numpy.random.random((2,3))

def finger(a):
    return numpy.cos(numpy.arange(a.size)).dot(a.ravel())

class KnowValues(unittest.TestCase):
    def test_fft_band(self):
        mf = scf.RHF(cell)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.037522993818908168, 9)

    def test_aft_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093770552709347754, 9)

    def test_df_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.DF(cell)
        mf.with_df.kpts_band = kband[0]
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093621142380270361, 9)

    def test_mdf_band(self):
        mf = scf.RHF(cell)
        mf.with_df = df.MDF(cell)
        mf.with_df.kpts_band = kband[0]
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.093705179440648712, 9)

    def test_fft_bands(self):
        mf = scf.KRHF(cell)
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.26538705237640059, 9)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.65622644106336381, 9)

    def test_aft_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.AFTDF(cell)
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32214958122356513, 9)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.64207984898611348, 9)

    def test_df_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.DF(cell)
        mf.with_df.kpts_band = kband
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32200476157045782, 9)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.64216552168395347, 9)

    def test_mdf_bands(self):
        mf = scf.KRHF(cell)
        mf.with_df = df.MDF(cell)
        mf.with_df.kpts_band = kband
        mf.kpts = cell.make_kpts([2]*3)
        mf.kernel()
        self.assertAlmostEqual(finger(mf.get_bands(kband[0])[0]), -0.32205954835271078, 9)
        self.assertAlmostEqual(finger(mf.get_bands(kband)[0]), -0.64209375063268592, 9)


if __name__ == '__main__':
    print("Full Tests for bands")
    unittest.main()

