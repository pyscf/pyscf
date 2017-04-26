import unittest
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import mdf
#from mpi4pyscf.pbc.df import mdf
pyscf.pbc.DEBUG = False

L = 5.
n = 5
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.rcut = 17
cell.build(0,0)

mf0 = pscf.RHF(cell)
mf0.exxdiv = 'vcut_sph'


numpy.random.seed(1)
kpts = numpy.random.random((5,3))
kpts[0] = 0
kpts[3] = kpts[0]-kpts[1]+kpts[2]
kpts[4] *= 1e-5

kmdf = mdf.MDF(cell)
kmdf.auxbasis = 'weigend'
kmdf.kpts = kpts
kmdf.gs = (5,)*3


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_vbar(self):
        auxcell = mdf.make_modrho_basis(cell, 'ccpvdz', 1.)
        vbar = mdf.MDF(cell).auxbar(auxcell)
        self.assertAlmostEqual(finger(vbar), -0.00438699039629, 9)

    def test_get_eri_gamma(self):
        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.52573746398795, 6)
        print(finger(eri0000), -1.2233754212058559, 7)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        print(eri1111.real.sum(), 140.52578834053651, 7)
        print(eri1111.imag.sum(), 0, 7)
        print(finger(eri1111), -1.2233754212485191, 7)
        self.assertTrue(numpy.allclose(eri1111, eri0000))

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        print(eri4444.real.sum(), 259.46584063870654, 7)
        print(abs(eri4444.imag).sum(), 0.00044187049454704288, 7)
        print(finger(eri4444), 1.97047850762612-3.6095445388135437e-07j, 7)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertTrue(numpy.allclose(eri0000, eri4444, atol=1e-7))

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 258.81892727996598, 6)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 16.275873507775319, 6)
        self.assertAlmostEqual(finger(eri1111), 2.2339187156692413+0.10954686831990143j, 7)
        check2 = kmdf.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 259.13093909555658, 6)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 8.4042469209022546, 6)
        self.assertAlmostEqual(finger(eri0011), 2.1375088961307176+0.12350225552066603j, 6)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 411.86008546054956, 6)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 136.58654649276673, 6)
        self.assertAlmostEqual(finger(eri0110), 1.3767170503977997+0.12379057924079918j, 6)
        check2 = kmdf.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 410.37542464581406, 6)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.18510527268199378, 6)
        self.assertAlmostEqual(finger(eri0123), 1.7644500565943559+0.30677193151572507j, 6)


if __name__ == '__main__':
    print("Full Tests for mdf")
    unittest.main()

