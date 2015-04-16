#!/usr/bin/env python

import os
import ctypes
import _ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

_loaderpath = os.path.join(os.path.dirname(lib.__file__), 'vhf')
libcvhf2 = numpy.ctypeslib.load_library('libcvhf', _loaderpath)


numpy.random.seed(15)
nao = 100
i0, j0, k0, l0 = 40,30,20,10
dm = numpy.random.random((nao,nao))
def run(fname):
    vj = numpy.zeros((nao,nao))
    di, dj, dk, dl = range(1,5)
    eri = numpy.asarray(numpy.random.random((di,dj,dk,dl)), order='F')
    fn = getattr(libcvhf2, fname)
    fn(eri.ctypes.data_as(ctypes.c_void_p),
       dm.ctypes.data_as(ctypes.c_void_p),
       vj.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(i0), ctypes.c_int(i0+di),
       ctypes.c_int(j0), ctypes.c_int(j0+dj),
       ctypes.c_int(k0), ctypes.c_int(k0+dk),
       ctypes.c_int(l0), ctypes.c_int(l0+dl),
       ctypes.c_int(nao))
    return eri, vj

class KnowValues(unittest.TestCase):
    def test_nrs1_ji_s1kl(self):
        eri, vj = run('CVHFnrs1_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))

    def test_nrs1_lk_s1ij(self):
        eri, vj = run('CVHFnrs1_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))

    def test_nrs1_jk_s1il(self):
        eri, vj = run('CVHFnrs1_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))

    def test_nrs1_li_s1kj(self):
        eri, vj = run('CVHFnrs1_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))

    def test_nrs2ij_lk_s1ij(self):
        eri, vj = run('CVHFnrs2ij_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,2,3), dm[l0:l0+dl,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,i0:i0+di], ref))

    def test_nrs2ij_ji_s1kl(self):
        eri, vj = run('CVHFnrs2ij_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref =(numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
            + numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,2,3), dm[i0:i0+di,j0:j0+dj]))
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))

    def test_nrs2ij_jk_s1il(self):
        eri, vj = run('CVHFnrs2ij_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(1,0,2,3), dm[i0:i0+di,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,l0:l0+dl], ref))

    def test_nrs2ij_li_s1kj(self):
        eri, vj = run('CVHFnrs2ij_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(1,0,2,3), dm[l0:l0+dl,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,i0:i0+di], ref))

    def test_nrs2kl_lk_s1ij(self):
        eri, vj = run('CVHFnrs2kl_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref =(numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
            + numpy.einsum('ijkl,lk->ij', eri.transpose(0,1,3,2), dm[k0:k0+dk,l0:l0+dl]))
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))

    def test_nrs2kl_ji_s1kl(self):
        eri, vj = run('CVHFnrs2kl_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,ji->kl', eri.transpose(0,1,3,2), dm[j0:j0+dj,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,k0:k0+dk], ref))

    def test_nrs2kl_jk_s1il(self):
        eri, vj = run('CVHFnrs2kl_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(0,1,3,2), dm[j0:j0+dj,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,k0:k0+dk], ref))

    def test_nrs2kl_li_s1kj(self):
        eri, vj = run('CVHFnrs2kl_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(0,1,3,2), dm[k0:k0+dk,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,j0:j0+dj], ref))

    def test_nrs4_ji_s1kl(self):
        eri, vj = run('CVHFnrs4_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        ref+= numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,2,3), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,ji->kl', eri.transpose(0,1,3,2), dm[j0:j0+dj,i0:i0+di])
        ref+= numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,3,2), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,k0:k0+dk], ref))

    def test_nrs4_lk_s1ij(self):
        eri, vj = run('CVHFnrs4_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        ref+= numpy.einsum('ijkl,lk->ij', eri.transpose(0,1,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,2,3), dm[l0:l0+dl,k0:k0+dk])
        ref+= numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,i0:i0+di], ref))

    def test_nrs4_jk_s1il(self):
        eri, vj = run('CVHFnrs4_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(0,1,3,2), dm[j0:j0+dj,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,k0:k0+dk], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(1,0,2,3), dm[i0:i0+di,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(1,0,3,2), dm[i0:i0+di,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,k0:k0+dk], ref))

    def test_nrs4_li_s1kj(self):
        eri, vj = run('CVHFnrs4_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(0,1,3,2), dm[k0:k0+dk,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(1,0,2,3), dm[l0:l0+dl,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,i0:i0+di], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(1,0,3,2), dm[k0:k0+dk,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,i0:i0+di], ref))


    def test_nra2ij_lk_s1ij(self):
        eri, vj = run('CVHFnra2ij_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,2,3), dm[l0:l0+dl,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,i0:i0+di], ref))

    def test_nra2ij_ji_s1kl(self):
        eri, vj = run('CVHFnra2ij_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref =(numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
            - numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,2,3), dm[i0:i0+di,j0:j0+dj]))
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))

    def test_nra2ij_jk_s1il(self):
        eri, vj = run('CVHFnra2ij_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(1,0,2,3), dm[i0:i0+di,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,l0:l0+dl], ref))

    def test_nra2ij_li_s1kj(self):
        eri, vj = run('CVHFnra2ij_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(1,0,2,3), dm[l0:l0+dl,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,i0:i0+di], ref))

    def test_nra2kl_lk_s1ij(self):
        eri, vj = run('CVHFnra2kl_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref =(numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
            - numpy.einsum('ijkl,lk->ij', eri.transpose(0,1,3,2), dm[k0:k0+dk,l0:l0+dl]))
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))

    def test_nra2kl_ji_s1kl(self):
        eri, vj = run('CVHFnra2kl_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,ji->kl', eri.transpose(0,1,3,2), dm[j0:j0+dj,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,k0:k0+dk], ref))

    def test_nra2kl_jk_s1il(self):
        eri, vj = run('CVHFnra2kl_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(0,1,3,2), dm[j0:j0+dj,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,k0:k0+dk], ref))

    def test_nra2kl_li_s1kj(self):
        eri, vj = run('CVHFnra2kl_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(0,1,3,2), dm[k0:k0+dk,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,j0:j0+dj], ref))

    def test_nra4ij_ji_s1kl(self):
        eri, vj = run('CVHFnra4ij_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        ref-= numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,2,3), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,ji->kl', eri.transpose(0,1,3,2), dm[j0:j0+dj,i0:i0+di])
        ref-= numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,3,2), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,k0:k0+dk], ref))

    def test_nra4ij_lk_s1ij(self):
        eri, vj = run('CVHFnra4ij_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        ref+= numpy.einsum('ijkl,lk->ij', eri.transpose(0,1,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,2,3), dm[l0:l0+dl,k0:k0+dk])
        ref+=-numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,i0:i0+di], ref))

    def test_nra4ij_jk_s1il(self):
        eri, vj = run('CVHFnra4ij_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(0,1,3,2), dm[j0:j0+dj,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,k0:k0+dk], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(1,0,2,3), dm[i0:i0+di,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(1,0,3,2), dm[i0:i0+di,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,k0:k0+dk], ref))

    def test_nra4ij_li_s1kj(self):
        eri, vj = run('CVHFnra4ij_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(0,1,3,2), dm[k0:k0+dk,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(1,0,2,3), dm[l0:l0+dl,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,i0:i0+di], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(1,0,3,2), dm[k0:k0+dk,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,i0:i0+di], ref))

    def test_nra4kl_ji_s1kl(self):
        eri, vj = run('CVHFnra4kl_ji_s1kl')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,ji->kl', eri, dm[j0:j0+dj,i0:i0+di])
        ref+= numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,2,3), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,ji->kl', eri.transpose(0,1,3,2), dm[j0:j0+dj,i0:i0+di])
        ref+=-numpy.einsum('ijkl,ji->kl', eri.transpose(1,0,3,2), dm[i0:i0+di,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,k0:k0+dk], ref))

    def test_nra4kl_lk_s1ij(self):
        eri, vj = run('CVHFnra4kl_lk_s1ij')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,lk->ij', eri, dm[l0:l0+dl,k0:k0+dk])
        ref-= numpy.einsum('ijkl,lk->ij', eri.transpose(0,1,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,2,3), dm[l0:l0+dl,k0:k0+dk])
        ref-= numpy.einsum('ijkl,lk->ij', eri.transpose(1,0,3,2), dm[k0:k0+dk,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,i0:i0+di], ref))

    def test_nra4kl_jk_s1il(self):
        eri, vj = run('CVHFnra4kl_jk_s1il')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,jk->il', eri, dm[j0:j0+dj,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(0,1,3,2), dm[j0:j0+dj,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[i0:i0+di,k0:k0+dk], ref))
        ref = numpy.einsum('ijkl,jk->il', eri.transpose(1,0,2,3), dm[i0:i0+di,k0:k0+dk])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,l0:l0+dl], ref))
        ref =-numpy.einsum('ijkl,jk->il', eri.transpose(1,0,3,2), dm[i0:i0+di,l0:l0+dl])
        self.assertTrue(numpy.allclose(vj[j0:j0+dj,k0:k0+dk], ref))

    def test_nra4kl_li_s1kj(self):
        eri, vj = run('CVHFnra4kl_li_s1kj')
        di, dj, dk, dl = eri.shape
        ref = numpy.einsum('ijkl,li->kj', eri, dm[l0:l0+dl,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,j0:j0+dj], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(0,1,3,2), dm[k0:k0+dk,i0:i0+di])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,j0:j0+dj], ref))
        ref = numpy.einsum('ijkl,li->kj', eri.transpose(1,0,2,3), dm[l0:l0+dl,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[k0:k0+dk,i0:i0+di], ref))
        ref =-numpy.einsum('ijkl,li->kj', eri.transpose(1,0,3,2), dm[k0:k0+dk,j0:j0+dj])
        self.assertTrue(numpy.allclose(vj[l0:l0+dl,i0:i0+di], ref))

if __name__ == '__main__':
    print('Full Tests for nrdot')
    unittest.main()


