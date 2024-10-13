import unittest
import numpy
from pyscf import lib
einsum = lib.einsum

def setUpModule():
    global bak
    lib.numpy_helper.EINSUM_MAX_SIZE, bak = 0, lib.numpy_helper.EINSUM_MAX_SIZE

def tearDownModule():
    global bak
    lib.numpy_helper.EINSUM_MAX_SIZE = bak

class KnownValues(unittest.TestCase):
    def test_d_d(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_c_c(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32)
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-5)

    def test_c_d(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-5)

    def test_d_z(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7)) + 0j
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_z_z(self):
        a = numpy.random.random((7,1,3,4)) + 0j
        b = numpy.random.random((2,4,5,7)) + 0j
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice1(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        c1 = einsum('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_d(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:,:,1:3,:], b)
        c1 = einsum('abcd,fdea->cebf', a[:,:,1:3,:], b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        c1 = einsum('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice1(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        c1 = einsum('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_cslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_z_cslice(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_cslice_dslice(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[2:6], b[:,:,1:3,2:6])
        c1 = einsum('abcd,fdea->cebf', a[2:6], b[:,:,1:3,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_3operands(self):
        a = numpy.random.random((7,1,3,4)) + 1j
        b = numpy.random.random((2,4,5,7))
        c = numpy.random.random((2,8,3,6))
        c0 = numpy.einsum('abcd,fdea,ficj->iebj', a, b, c)
        c1 = einsum('abcd,fdea,ficj->iebj', a, b, c)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_3operands1(self):
        a = numpy.random.random((2,2,2,2)) + 1j
        b = numpy.random.random((2,2,2,2))
        c = numpy.random.random((2,2,2,2))
        c0 = numpy.einsum('abcd,acde,adef->ebf', a, b, c)
        c1 = einsum('abcd,acde,adef->ebf', a, b, c)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_3operands2(self):
        a = numpy.random.random(4)
        ab = numpy.random.rand(4, 5)
        b = numpy.random.random(5)
        c0 = numpy.einsum('i,ij,j->ij', a, ab, b)
        c1 = einsum('i,ij,j->ij', a, ab, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_1operand(self):
        a = numpy.random.random((4,1,3,4)) + 1j
        c0 = numpy.einsum('abca->bc', a)
        c1 = einsum('abca->bc', a)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_wrong_dimension(self):
        a = numpy.random.random((5,1,3,4))
        b = numpy.random.random((2,4,5,7))
        self.assertRaises(ValueError, einsum, 'abcd,fdea->cebf', a, b)

    def test_contraction1(self):
        a = numpy.random.random((5,2,3,2))
        c0 = numpy.einsum('ijkj->k', a)
        c1 = einsum('ijkj->k', a)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction2(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))+1j
        c0 = numpy.einsum('ijkj,jlp->pk', a, b)
        c1 = einsum('ijkj,jlp->pk', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction3(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))+1j
        c0 = numpy.einsum('ijkj,jlp->jpk', a, b)
        c1 = einsum('ijkj,jlp->jpk', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction4(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))
        c0 = numpy.einsum('...jkj,jlp->...jp', a, b)
        c1 = einsum('...jkj,jlp->...jp', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction5(self):
        x = numpy.random.random((8,6))
        y = numpy.random.random((8,6,6))
        c0 = numpy.einsum("in,ijj->n", x, y)
        c1 = einsum("in,ijj->n", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = numpy.random.random((6,6))
        y = numpy.random.random((8,8))
        c0 = numpy.einsum("ii,jj->", x, y)
        c1 = einsum("ii,jj->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = numpy.random.random((6))
        y = numpy.random.random((8))
        c0 = numpy.einsum("i,j->", x, y)
        c1 = einsum("i,j->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        c0 = numpy.einsum("i,i->", x, x)
        c1 = einsum("i,i->", x, x)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

        x = numpy.random.random((6,8))
        y = numpy.random.random((8,6))
        c0 = numpy.einsum("ij,ji->", x, y)
        c1 = einsum("ij,ji->", x, y)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contract(self):
        try:
            from pyscf.lib import tblis_einsum
            tblis_available = True
        except (ImportError, OSError):
            tblis_available = False

        if tblis_available:
            a = numpy.random.random((5,4,6))
            b = numpy.random.random((4,9,6))

            c1 = numpy.ones((9,5), dtype=numpy.complex128)
            c0 = tblis_einsum.contract('ijk,jlk->li', a, b, out=c1, alpha=.5j, beta=.2)
            c1 = numpy.ones((9,5), dtype=numpy.complex128)
            c1 = c1*.2 + numpy.einsum('ijk,jlk->li', a, b)*.5j
            self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_4operands(self):
        a = numpy.random.random((30,40,5,10))
        b = numpy.random.random((10,30,5,20))
        c = numpy.random.random((10,20,20))
        d = numpy.random.random((20,10))
        f = lib.einsum('ijkl,xiky,ayp,px->ajl', a,b,c,d, optimize=True)
        ref = lib.einsum('ijkl,xiky->jlxy', a,b)
        ref = lib.einsum('jlxy,ayp->jlxap', ref,c)
        ref = lib.einsum('jlxap,px->ajl', ref,d)
        self.assertAlmostEqual(abs(ref-f).max(), 0, 9)

        f = lib.einsum('ijkl,xiky,lyp,px->jl', a,b,c,d, optimize=True)
        ref = lib.einsum('ijkl,xiky->jlxy', a, b)
        ref = lib.einsum('jlxy,lyp->jlxp', ref, c)
        ref = lib.einsum('jlxp,px->jl', ref, d)
        self.assertAlmostEqual(abs(ref-f).max(), 0, 9)


if __name__ == '__main__':
    unittest.main()
