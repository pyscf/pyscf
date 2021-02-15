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

import unittest
import copy
import ctypes
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import _pbcintor
from pyscf.scf import _vhf

libpbc = lib.load_library('libpbc')

cell = gto.M(atom='C 1 2 1; C 1 1 1', a=numpy.eye(3)*4, mesh = [5]*3,
             basis = {'C':[[0, (1, 1)],
                           [1, (.5, 1)],
                           [2, (1.5, 1)]
                           ]}
            )
numpy.random.seed(1)
kband = numpy.random.random((2,3))
Ls = numpy.random.random((27,3)) * 10


def run3c(fill, kpts, shls_slice=None):
    intor = 'int3c2e_sph'
    nao = cell.nao_nr()
    nkpts = len(kpts)
    if fill == 'PBCnr3c_fill_gs2':
        out = numpy.empty((nao*(nao+1)//2,nao))
        kptij_idx = numpy.arange(nkpts).astype(numpy.int32)
    elif fill == 'PBCnr3c_fill_gs1':
        out = numpy.empty((nao,nao,nao))
        kptij_idx = numpy.arange(nkpts).astype(numpy.int32)
    elif fill in ('PBCnr3c_fill_kks1', 'PBCnr3c_fill_kks2'):
        kptij_idx = numpy.asarray([i*nkpts+j for i in range(nkpts) for j in range(i+1)], dtype=numpy.int32)
        out = numpy.empty((len(kptij_idx),nao,nao,nao), dtype=numpy.complex128)
    elif fill == 'PBCnr3c_fill_ks1':
        kptij_idx = numpy.arange(nkpts).astype(numpy.int32)
        out = numpy.empty((nkpts,nao,nao,nao), dtype=numpy.complex128)
    elif fill == 'PBCnr3c_fill_ks2':
        out = numpy.empty((nkpts,nao*(nao+1)//2,nao), dtype=numpy.complex128)
        kptij_idx = numpy.arange(nkpts).astype(numpy.int32)
    else:
        raise RuntimeError
    nkpts_ij = len(kptij_idx)
    nimgs = len(Ls)
    expkL = numpy.exp(1j*numpy.dot(kpts, Ls.T))
    comp = 1
    if shls_slice is None:
        shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2,
                      cell.nbas*2, cell.nbas*3)

    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, 'int3c2e_sph')
    cintopt = lib.c_null_ptr()
    pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell, 1e-9)

    libpbc.PBCnr3c_drv(getattr(libpbc, intor), getattr(libpbc, fill),
                       out.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(nkpts_ij), ctypes.c_int(nkpts),
                       ctypes.c_int(comp), ctypes.c_int(len(Ls)),
                       Ls.ctypes.data_as(ctypes.c_void_p),
                       expkL.ctypes.data_as(ctypes.c_void_p),
                       kptij_idx.ctypes.data_as(ctypes.c_void_p),
                       (ctypes.c_int*6)(*shls_slice),
                       ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, pbcopt._this,
                       atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
                       env.ctypes.data_as(ctypes.c_void_p))
    return out

def run2c(intor, fill, kpts, shls_slice=None):
    nkpts = len(kpts)

    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    if shls_slice is None:
        shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    comp = 1
    out = numpy.empty((nkpts,comp,ni,nj), dtype=numpy.complex128)

    fintor = getattr(gto.moleintor.libcgto, intor)
    fill = getattr(libpbc, fill)
    intopt = lib.c_null_ptr()
    pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell, 1e-9)

    expkL = numpy.asarray(numpy.exp(1j*numpy.dot(kpts, Ls.T)), order='C')
    drv = libpbc.PBCnr2c_drv
    drv(fintor, fill, out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts), ctypes.c_int(comp), ctypes.c_int(len(Ls)),
        Ls.ctypes.data_as(ctypes.c_void_p),
        expkL.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*4)(*(shls_slice[:4])),
        ao_loc.ctypes.data_as(ctypes.c_void_p), intopt, pbcopt._this,
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
        env.ctypes.data_as(ctypes.c_void_p))
    return out

def finger(a):
    return numpy.dot(numpy.cos(numpy.arange(a.size)), a.ravel())

class KnowValues(unittest.TestCase):
    def test_fill_kk(self):
        fill = 'PBCnr3c_fill_kks1'
        out = out0 = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), 1.0182971799139509+2.3791697509775567j, 9)

        fill = 'PBCnr3c_fill_kks2'
        out = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), 1.0182971799138514+2.3791697509775656j, 9)
        self.assertTrue(numpy.allclose(out, out0))

    def test_fill_k(self):
        fill = 'PBCnr3c_fill_ks2'
        out = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), 6.6227481557912071-0.80306690266835279j, 9)

        out0 = run3c('PBCnr3c_fill_kks2', kband)
        idx = numpy.tril_indices(cell.nao_nr())
        self.assertTrue(numpy.allclose(out0[0][idx], out[0]))
        self.assertTrue(numpy.allclose(out0[2][idx], out[1]))

        fill = 'PBCnr3c_fill_ks1'
        out = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), -14.883220617173009-3.9101277614911112j, 9)
        self.assertTrue(numpy.allclose(out0[0], out[0]))
        self.assertTrue(numpy.allclose(out0[2], out[1]))

        fill = 'PBCnr3c_fill_ks2'
        out = run3c(fill, numpy.zeros((1,3)))
        self.assertAlmostEqual(finger(out), -1.3258082818877739, 9)

    def test_fill_g(self):
        fill = 'PBCnr3c_fill_gs2'
        out = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), -1.3258082818877739, 9)

        fill = 'PBCnr3c_fill_gs1'
        out = run3c(fill, kband)
        self.assertAlmostEqual(finger(out), -14.162823165066602, 9)

        mat1 = run3c(fill, kband,
                     shls_slice=(1,4,cell.nbas+2,cell.nbas+4,cell.nbas*2+2,cell.nbas*2+3))
        mat1 = mat1.ravel()[:9*6*5].reshape(9,6,5)
        self.assertTrue(numpy.allclose(out[1:10,4:10,4:9],mat1))

    def test_fill_2c(self):
        mat = cell.pbc_intor('int1e_ovlp_sph')
        self.assertAlmostEqual(finger(mat), 2.2144557629971247, 9)

        mat = run2c('int1e_ovlp_sph', 'PBCnr2c_fill_ks1', kpts=numpy.zeros((1,3)))
        self.assertAlmostEqual(finger(mat), -0.27252856478433407, 9)

        mat1 = run2c('int1e_ovlp_sph', 'PBCnr2c_fill_ks1',
                     kpts=numpy.zeros((1,3)),
                     shls_slice=(1,4,cell.nbas+2,cell.nbas+4))
        self.assertTrue(numpy.allclose(mat[0,0,1:10,4:10],mat1[0,0]))

        mat = cell.pbc_intor('int1e_ovlp_sph', kpts=kband)
        self.assertAlmostEqual(finger(mat[0]), 2.2137492396285916-0.004739404845627319j, 9)
        self.assertAlmostEqual(finger(mat[1]), 2.2132325548987253+0.0056984781658280699j, 9)


if __name__ == '__main__':
    print('Full Tests for pbc_fill_int')
    unittest.main()
