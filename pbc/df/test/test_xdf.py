import unittest
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import xdf
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

kxdf = xdf.XDF(cell)
kxdf.auxbasis = 'weigend'
kxdf.approx_sr_level = 3
kxdf.kpts = kpts
kxdf.gs = (5,)*3


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_vbar(self):
        auxcell = xdf.make_modrho_basis(cell, 'ccpvdz', 1.)
        vbar = xdf.XDF(cell).auxbar(auxcell)
        self.assertAlmostEqual(finger(vbar), -0.00438699039629, 9)

    def test_get_nuc(self):
        dm = mf0.get_init_guess()

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.60957644775101, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.approx_sr_level = 2
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.60957682112786, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        self.assertAlmostEqual(numpy.einsum('ij,ji->', odf.get_nuc(), dm),
                               -150.60957390729678, 9)

        numpy.random.seed(1)
        kpts = numpy.random.random((2,3))

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[0]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        vnuc = odf.get_nuc(kpts=kpt)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', vnuc, dm),
                               -150.60924201533331, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[0]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=kpt)
        ek0 = numpy.einsum('ij,ji->', vnuc, dm)
        self.assertAlmostEqual(ek0, -150.60923914072549, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        kpt = kpts[1]
        odf.kpts = [kpt]
        odf.gs = (5,)*3
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=kpt)
        ek1 = numpy.einsum('ij,ji->', vnuc, dm)
        self.assertAlmostEqual(ek1, -150.60816189187889, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        odf.kpts = kpts
        odf.approx_sr_level = 3
        vnuc = odf.get_nuc(kpts=odf.kpts)
        self.assertAlmostEqual(numpy.einsum('xij,ji->', vnuc, dm), ek0+ek1, 9)

    def test_get_eri_gamma(self):
        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.26255441220636, 9)
        self.assertAlmostEqual(finger(eri0000), -1.2394797964725166, 9)

        odf = xdf.XDF(cell)
        odf.auxbasis = 'weigend'
        odf.approx_sr_level = 3
        odf.gs = (5,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.25105329856345, 9)
        self.assertAlmostEqual(finger(eri0000), -1.2395110178359614, 9)

        eri1111 = kxdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 140.25105329856345, 9)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(finger(eri1111), -1.2395110178359614, 9)
        self.assertTrue(numpy.allclose(eri1111, eri0000))

        eri4444 = kxdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 258.39992902506515, 9)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0.00044176286086290175, 9)
        self.assertAlmostEqual(finger(eri4444), 1.9708351959686539-3.9022542925924344e-07j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertTrue(numpy.allclose(eri0000, eri4444, atol=1e-7))

    def test_get_eri_1111(self):
        eri1111 = kxdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 257.76941191498139, 9)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 16.274561967292094, 9)
        self.assertAlmostEqual(finger(eri1111), 2.2335275741087415+0.10851191298241072j, 9)
        check2 = kxdf.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011(self):
        eri0011 = kxdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 258.07334571586989, 9)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 8.4035872280957342, 9)
        self.assertAlmostEqual(finger(eri0011), 2.1388240207255667+0.12333861053161892j, 9)

    def test_get_eri_0110(self):
        eri0110 = kxdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 411.74271228002362, 9)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 136.37680387763899, 9)
        self.assertAlmostEqual(finger(eri0110), 1.3786746950186308+0.12073694181289334j, 9)
        check2 = kxdf.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kxdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 410.28237390383345, 9)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.1846104945658219, 9)
        self.assertAlmostEqual(finger(eri0123), 1.7647356643840468+0.30596849894410644j, 9)

    def test_get_mo_eri(self):
        nao = cell.nao_nr()
        eri = ao2mo.restore(1, kxdf.get_ao_eri((0,0,0)), nao)
        numpy.random.seed(5)
        mo = numpy.random.random((nao,nao))
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo)
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo)
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo)
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo)
        eri0 = ao2mo.restore(4, eri0, nao)
        eri1 = kxdf.get_mo_eri(mo, kpts[0])
        self.assertTrue(numpy.allclose(eri0,eri1))

        eri = kxdf.get_ao_eri(kpts[:4]).reshape((nao,)*4)
        numpy.random.seed(5)
        mo = numpy.random.random((nao,nao))
        eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
        eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
        eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
        eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       ).reshape(nao**2,-1)
        eri1 = kxdf.get_mo_eri(mo, kpts[:4])
        self.assertTrue(numpy.allclose(eri0,eri1))



if __name__ == '__main__':
    print("Full Tests for xdf")
    unittest.main()



