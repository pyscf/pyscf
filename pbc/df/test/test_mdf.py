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
cell.h = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 1000
cell.nimgs = [2,2,2]
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

    def test_get_nuc(self):
        dm = mf0.get_init_guess()

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.61096857467683, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.approx_sr_level = 2
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.61096857563984, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.61096857973422, 9)

        numpy.random.seed(1)
        kpts = numpy.random.random((2,3))

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[0]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        vnuc = odf.get_nuc(kpts=kpt)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', vnuc, dm),
                               -150.6106338105896, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[0]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        odf.approx_sr_level = 1
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.61096857467683, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[0]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=kpt)
        ek0 = numpy.einsum('ij,ji->', vnuc, dm)
        self.assertAlmostEqual(ek0, -150.61063381611541, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[1]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=kpt)
        ek1 = numpy.einsum('ij,ji->', vnuc, dm)
        self.assertAlmostEqual(ek1, -150.609556567277, 9)

        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.kpts = kpts
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=odf.kpts)
        self.assertAlmostEqual(numpy.einsum('xij,ji->', vnuc, dm), ek0+ek1, 9)

    def test_get_eri_gamma(self):
        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.52119757983661, 9)
        self.assertAlmostEqual(finger(eri0000), -1.2233310290052046, 9)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 140.52119757983661, 9)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(finger(eri1111), -1.2233310290052046, 9)
        self.assertTrue(numpy.allclose(eri1111, eri0000))

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 259.45414408787667, 9)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0.00044186526697869387, 9)
        self.assertAlmostEqual(finger(eri4444), 1.968178030116118-3.6128363791369754e-07j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertTrue(numpy.allclose(eri0000, eri4444, atol=1e-7))

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 258.80750136730046, 9)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 16.275644241636385, 9)
        self.assertAlmostEqual(finger(eri1111), 2.2316054396417342+0.10954248829900334j, 9)
        check2 = kmdf.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 259.11949819154603, 9)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 8.4041296068314857, 9)
        self.assertAlmostEqual(finger(eri0011), 2.1351781088791886+0.1235017871924925j, 9)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 411.848224952762, 9)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 136.58658276376337, 9)
        self.assertAlmostEqual(finger(eri0110), 1.3743766723562483+0.12351558863744456j, 9)
        check2 = kmdf.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 410.38796097327645, 9)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.18511135642499094, 9)
        self.assertAlmostEqual(finger(eri0123), 1.7616933071834859+0.30808257556235996j, 9)

    def test_get_mo_eri(self):
        nao = cell.nao_nr()
        eri = ao2mo.restore(1, kmdf.get_ao_eri((0,0,0)), nao)
        numpy.random.seed(5)
        mo = numpy.random.random((nao,nao))
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo)
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo)
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo)
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo)
        eri0 = ao2mo.restore(4, eri0, nao)
        eri1 = kmdf.get_mo_eri(mo, kpts[0])
        self.assertTrue(numpy.allclose(eri0,eri1))

        eri = kmdf.get_ao_eri(kpts[:4]).reshape((nao,)*4)
        numpy.random.seed(5)
        mo = numpy.random.random((nao,nao))
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       ).reshape(nao**2,-1)
        eri1 = kmdf.get_mo_eri(mo, kpts[:4])
        self.assertTrue(numpy.allclose(eri0,eri1))



if __name__ == '__main__':
    print("Full Tests for mdf")
    unittest.main()

