#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto import ft_ao

libpbc = lib.load_library('libpbc')

def setUpModule():
    global mol, b, Gvbase, Gv, gxyz
    mol = gto.Mole()
    mol.atom = '''
    C    1.3    .2       .3
    C     .1   -.1      1.1 '''
    mol.basis = 'ccpvdz'
    mol.build()
    mesh = (7,9,11)

    numpy.random.seed(12)
    invh = numpy.diag(numpy.random.random(3))
    b = 2*numpy.pi * invh
    Gvbase = (numpy.fft.fftfreq(mesh[0], 1./mesh[0]),
              numpy.fft.fftfreq(mesh[1], 1./mesh[1]),
              numpy.fft.fftfreq(mesh[2], 1./mesh[2]))
    Gv = numpy.dot(lib.cartesian_prod(Gvbase), b)
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])

def tearDownModule():
    global mol, Gvbase, Gv, gxyz
    del mol, Gvbase, Gv, gxyz


def ft_ao_o0(mol, Gv):
    nao = mol.nao_nr()
    ngrids = Gv.shape[0]
    aoG = numpy.zeros((nao,ngrids), dtype=numpy.complex128)
    gx = numpy.empty((12,ngrids), dtype=numpy.complex128)
    gy = numpy.empty((12,ngrids), dtype=numpy.complex128)
    gz = numpy.empty((12,ngrids), dtype=numpy.complex128)
    buf = numpy.empty((64,ngrids), dtype=numpy.complex128)
    kk = numpy.einsum('ki,ki->k', Gv, Gv)

    i0 = 0
    for ib in range(mol.nbas):
        ci = mol._libcint_ctr_coeff(ib)
        ei = mol.bas_exp(ib)
        li = mol.bas_angular(ib)
        ri = mol.bas_coord(ib)
        ni = ci.shape[1]
        di = (li*2+1) * ni
        nfi = (li+1)*(li+2)//2
        kr = numpy.dot(Gv,ri)
        cs = numpy.exp(-1j*kr)

        buf[:nfi*ni] = 0
        for ip in range(ci.shape[0]):
            ai = ei[ip]
            fac = (numpy.pi/ai)**1.5 * numpy.exp(-.25/ai*kk)
            gx[0] = 1
            gy[0] = 1
            gz[0] = cs * fac
            if li > 0:
                gx[1] = -1j*Gv[:,0]/(2*ai) * gx[0]
                gy[1] = -1j*Gv[:,1]/(2*ai) * gy[0]
                gz[1] = -1j*Gv[:,2]/(2*ai) * gz[0]
                for m in range(1, li):
                    gx[m+1] = m/(2*ai) * gx[m-1] - 1j*Gv[:,0]/(2*ai) * gx[m]
                    gy[m+1] = m/(2*ai) * gy[m-1] - 1j*Gv[:,1]/(2*ai) * gy[m]
                    gz[m+1] = m/(2*ai) * gz[m-1] - 1j*Gv[:,2]/(2*ai) * gz[m]

            for m,(ix,iy,iz) in enumerate(loop_cart(li)):
                val = gx[ix] * gy[iy] * gz[iz]
                for i, cip in enumerate(ci[ip]):
                    buf[i*nfi+m] += cip*val

        ti = c2s_bra(li, numpy.eye(nfi)).T
        tmp1 = numpy.empty((di,ngrids), dtype=numpy.complex128)
        for i in range(ni):
            tmp1[i*(li*2+1):(i+1)*(li*2+1)] = \
                    numpy.einsum('pi,px->ix', ti, buf[i*nfi:(i+1)*nfi])
        aoG[i0:i0+di] += tmp1
        i0 += di
    return aoG.T

def loop_cart(l):
    for ix in reversed(range(l+1)):
        for iy in reversed(range(l-ix+1)):
            iz = l - ix - iy
            yield ix, iy, iz

def c2s_bra(l, gcart):
    if l == 0:
        return gcart * 0.282094791773878143
    elif l == 1:
        return gcart * 0.488602511902919921
    else:
        m = gcart.shape[1]
        gsph = numpy.empty((l*2+1,m))
        fc2s = gto.moleintor.libcgto.CINTc2s_ket_sph
        fc2s(gsph.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(m),
             gcart.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(l))
        return gsph

class KnownValues(unittest.TestCase):
    def test_ft_ao1(self):
        ref = ft_ao_o0(mol, Gv)
        dat = ft_ao.ft_ao(mol, Gv)
        self.assertTrue(numpy.allclose(ref, dat))

        dat = ft_ao.ft_ao(mol, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase)
        self.assertTrue(numpy.allclose(ref, dat))

    def test_ft_ao2(self):
        numpy.random.seed(12)
        invh = numpy.random.random(3) + numpy.eye(3) * 2.5
        b = 2*numpy.pi * invh
        Gv = numpy.dot(lib.cartesian_prod(Gvbase), b)
        ref = ft_ao_o0(mol, Gv)
        dat = ft_ao.ft_ao(mol, Gv)
        self.assertTrue(numpy.allclose(ref, dat))

        mol1 = mol.copy()
        mol1.cart = True
        ref = ft_ao.ft_ao(mol1, Gv)
        dat = ft_ao.ft_ao(mol1, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase)
        self.assertTrue(numpy.allclose(ref, dat))

    def test_ft_aopair1(self):
        dat = ft_ao.ft_aopair(mol, Gv)
        self.assertAlmostEqual(lib.fp(dat), (-5.9794759129252348+8.07254562525371j), 9)

        dat_s2 = ft_ao.ft_aopair(mol, Gv, aosym='s2')
        nao = dat.shape[-1]
        for i in range(nao):
            for j in range(i+1):
                dat[:,i,j] = dat[:,j,i] = dat_s2[:,i*(i+1)//2+j]
        self.assertAlmostEqual(lib.fp(dat), (-5.9794759129252348+8.07254562525371j), 9)

        dat1 = ft_ao.ft_aopair(mol, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(lib.fp(dat1), (-5.9794759129252348+8.07254562525371j), 9)

    def test_ft_aopair2(self):
        numpy.random.seed(12)
        invh = numpy.random.random(3) + numpy.eye(3) * 2.5
        b = 2*numpy.pi * invh
        Gv = numpy.dot(lib.cartesian_prod(Gvbase), b)
        dat = ft_ao.ft_aopair(mol, Gv)
        self.assertAlmostEqual(lib.fp(dat), (-3.1468496579780125-0.019209667673850885j), 9)

        dat1 = ft_ao.ft_aopair(mol, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase)
        self.assertAlmostEqual(lib.fp(dat1), (-3.1468496579780125-0.019209667673850885j), 9)

    def test_ft_aopair_pdotp(self):
        dat = ft_ao.ft_aopair(mol, Gv, intor='GTO_ft_pdotp_sph')
        self.assertAlmostEqual(lib.fp(dat), (-80.69687735727976+69.239798150854909j), 9)

    def test_ft_aopair_pxp(self):
        dat = ft_ao.ft_aopair(mol, Gv, intor='GTO_ft_pxp_sph', comp=3)
        self.assertAlmostEqual(lib.fp(dat), (3.7490985032017079+43.665863070814687j), 8)

    def test_ft_aopair_overlap0(self):
        G = numpy.asarray([[-1.679872,  1.679872,  2.937055],
                           [-1.425679,  1.425679 , 2.492629],
                           [-1.187609 , 1.187609 , 2.076392]])
        mol = gto.M(atom='Ne 7 0.0 0.0; Ne 7 0.0 0.0', basis='3-21g')
        dat = ft_ao.ft_aopair(mol, G)
        self.assertAlmostEqual(lib.fp(dat), (-1.4150713647161861-0.8020058716859948j), 12)

if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
