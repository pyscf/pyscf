#!/usr/bin/env python
import ctypes
import unittest
import numpy
from pyscf import lib, gto

libcgto = lib.load_library('libdft')
BLKSIZE = 64

mol = gto.M(atom='''
            O 0.5 0.5 0.
            H  1.  1.2 0.
            H  0.  0.  1.3
            ''',
            basis='ccpvqz')

def eval_gto(mol, eval_name, coords,
             comp=1, shls_slice=None, non0tab=None, ao_loc=None, out=None):
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    coords = numpy.asarray(coords, dtype=numpy.double, order='F')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]
    if ao_loc is None:
        ao_loc = gto.moleintor.make_loc(bas, eval_name)

    if shls_slice is None:
        shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0];
    if 'spinor' in eval_name:
        ao = numpy.ndarray((2,comp,nao,ngrids), dtype=numpy.complex128, buffer=out)
    else:
        ao = numpy.ndarray((comp,nao,ngrids), buffer=out)

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas),
                             dtype=numpy.int8)

    drv = getattr(libcgto, eval_name)
    drv((ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ngrids),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    ao = numpy.swapaxes(ao, -1, -2)
    if comp == 1:
        if 'spinor' in eval_name:
            ao = ao[:,0]
        else:
            ao = ao[0]
    return ao

numpy.random.seed(1)
ngrids = 2000
coords = numpy.random.random((ngrids,3))
coords = (coords-.5)**2 * 80

def finger(a):
    return numpy.dot(numpy.cos(numpy.arange(a.size)), a.ravel())

class KnowValues(unittest.TestCase):
    def test_sph(self):
        ao = eval_gto(mol, 'GTOval_sph', coords)
        self.assertAlmostEqual(finger(ao), -6.8109234394857712, 9)

    def test_cart(self):
        ao = eval_gto(mol, 'GTOval_cart', coords)
        self.assertAlmostEqual(finger(ao), -16.384888666900274, 9)

    def test_ip_cart(self):
        ao = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3)
        self.assertAlmostEqual(finger(ao), 94.04527465181198, 9)

    def test_sph_deriv1(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv1', coords, comp=4)
        self.assertAlmostEqual(finger(ao), -45.129633361047482, 9)

    def test_sph_deriv2(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv2', coords, comp=10)
        self.assertAlmostEqual(finger(ao), -88.126901222477954, 9)

    def test_sph_deriv3(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv3', coords, comp=20)
        self.assertAlmostEqual(finger(ao), -402.84257273073263, 9)

    def test_sph_deriv4(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv4', coords, comp=35)
        self.assertAlmostEqual(finger(ao), 4933.0635429300246, 9)

    def test_shls_slice(self):
        ao0 = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3)
        ao1 = ao0[:,:,14:77]
        ao = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3, shls_slice=(7, 19))
        self.assertAlmostEqual(abs(ao-ao1).sum(), 0, 9)

    def test_ig_sph(self):
        ao = eval_gto(mol, 'GTOval_ig_sph', coords, comp=3)
        self.assertAlmostEqual(finger(ao), 8.6601301646129052, 9)

    def test_ipig_sph(self):
        ao = eval_gto(mol, 'GTOval_ipig_sph', coords, comp=9)
        self.assertAlmostEqual(finger(ao), -53.56608497643537, 9)

    def test_spinor(self):
        ao = eval_gto(mol, 'GTOval_spinor', coords)
        self.assertAlmostEqual(finger(ao), -4.5941099464020079+5.9444339000526707j, 9)

    def test_ip_spinor(self):
        ao = eval_gto(mol, 'GTOval_ip_spinor', coords, comp=3)
        self.assertAlmostEqual(finger(ao), -52.516545034166775+24.765350351395604j, 9)

    def test_sp_spinor(self):
        ao = eval_gto(mol, 'GTOval_sp_spinor', coords)
        self.assertAlmostEqual(finger(ao), 14.570414046478941-64.303103699665527j, 9)

    def test_ipsp_spinor(self):
        ao = eval_gto(mol, 'GTOval_ipsp_spinor', coords, comp=3)
        self.assertAlmostEqual(finger(ao), -129.67878568476067+393.58768218886212j, 9)


if __name__ == '__main__':
    print('Full Tests for grid_ao')
    unittest.main()
