#!/usr/bin/env python

import ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto import ft_ao

libpbc = lib.load_library('libpbc')
mol = gto.Mole()
mol.atom = '''
C    1.3    .2       .3
C     .1   -.1      1.1 '''
mol.basis = 'ccpvdz'
mol.build()
gs = (3,4,5)
gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
nGv = gxyz.shape[0]


def ft_ao_o0(mol, Gv):
    nao = mol.nao_nr()
    ngrids = Gv.shape[0]
    aoG = numpy.zeros((nao,ngrids), dtype=numpy.complex)
    gx = numpy.empty((12,ngrids), dtype=numpy.complex)
    gy = numpy.empty((12,ngrids), dtype=numpy.complex)
    gz = numpy.empty((12,ngrids), dtype=numpy.complex)
    buf = numpy.empty((64,ngrids), dtype=numpy.complex)
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
        tmp1 = numpy.empty((di,ngrids), dtype=numpy.complex)
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

def finger(a):
    return numpy.dot(a.ravel(), numpy.cos(numpy.arange(a.size)))

class KnowValues(unittest.TestCase):
    def test_ft_ao(self):
        numpy.random.seed(12)
        invh = numpy.diag(numpy.random.random(3))
        Gv = 2*numpy.pi* numpy.dot(gxyz, invh)
        ref = ft_ao_o0(mol, Gv)
        dat = ft_ao.ft_ao(mol, Gv)
        self.assertTrue(numpy.allclose(ref, dat))

    def test_ft_aopair(self):
        numpy.random.seed(12)
        invh = numpy.diag(numpy.random.random(3))
        Gv = 2*numpy.pi* numpy.dot(gxyz, invh)
        dat = ft_ao.ft_aopair(mol, Gv)
        self.assertAlmostEqual(finger(dat), (-5.9794759129252348+8.07254562525371j), 9)

        dat_s2 = ft_ao.ft_aopair(mol, Gv, aosym='s2')
        nao = dat.shape[-1]
        for i in range(nao):
            for j in range(i+1):
                dat[:,i,j] = dat[:,j,i] = dat_s2[:,i*(i+1)//2+j]
        self.assertAlmostEqual(finger(dat), (-5.9794759129252348+8.07254562525371j), 9)


if __name__ == '__main__':
    print('Full Tests for ft_ao')
    unittest.main()
