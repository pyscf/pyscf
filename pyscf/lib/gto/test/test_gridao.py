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
from pyscf import lib, gto, df
from pyscf.gto.eval_gto import BLKSIZE

libcgto = lib.load_library('libdft')

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
    nao = ao_loc[sh1] - ao_loc[sh0]
    if 'spinor' in eval_name:
        ao = numpy.ndarray((2,comp,nao,ngrids), dtype=numpy.complex128, buffer=out)
    else:
        ao = numpy.ndarray((comp,nao,ngrids), buffer=out)

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas),
                             dtype=numpy.uint8)

    drv = getattr(libcgto, eval_name)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
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

PTR_ENV_START = 20
CHARGE_OF  = 0
PTR_COORD  = 1
NUC_MOD_OF = 2
PTR_ZETA   = 3
RAD_GRIDS  = 4
ANG_GRIDS  = 5
ATM_SLOTS  = 6
ATOM_OF   = 0
ANG_OF    = 1
NPRIM_OF  = 2
NCTR_OF   = 3
KAPPA_OF  = 4
PTR_EXP   = 5
PTR_COEFF = 6
BAS_SLOTS = 8
natm = 4
nbas = 0
atm = numpy.zeros((natm,ATM_SLOTS), dtype=numpy.int32)
bas = numpy.zeros((1000,BAS_SLOTS), dtype=numpy.int32)
env = numpy.zeros(10000)
off = PTR_ENV_START
for i in range(natm):
    atm[i, CHARGE_OF] = (i+1)*2
    atm[i, PTR_COORD] = off
    env[off+0] = .2 * (i+1)
    env[off+1] = .3 + (i+1) * .5
    env[off+2] = .1 - (i+1) * .5
    off += 3
off0 = off
# basis with kappa > 0
nh = 0
bas[nh,ATOM_OF ]  = 0
bas[nh,ANG_OF  ]  = 1
bas[nh,KAPPA_OF]  = 1
bas[nh,NPRIM_OF]  = 1
bas[nh,NCTR_OF ]  = 1
bas[nh,PTR_EXP]   = off
env[off+0] = 1
bas[nh,PTR_COEFF] = off + 1
env[off+1] = 1
off += 2
nh += 1
bas[nh,ATOM_OF ]  = 1
bas[nh,ANG_OF  ]  = 2
bas[nh,KAPPA_OF]  = 2
bas[nh,NPRIM_OF]  = 2
bas[nh,NCTR_OF ]  = 2
bas[nh,PTR_EXP]   = off
env[off+0] = 5
env[off+1] = 3
bas[nh,PTR_COEFF] = off + 2
env[off+2] = 1
env[off+3] = 2
env[off+4] = 4
env[off+5] = 1
off += 6
nh += 1
bas[nh,ATOM_OF ]  = 2
bas[nh,ANG_OF  ]  = 3
bas[nh,KAPPA_OF]  = 3
bas[nh,NPRIM_OF]  = 1
bas[nh,NCTR_OF ]  = 1
bas[nh,PTR_EXP ]  = off
env[off+0] = 1
bas[nh,PTR_COEFF] = off + 1
env[off+1] = 1
off += 2
nh += 1
bas[nh,ATOM_OF ]  = 3
bas[nh,ANG_OF  ]  = 4
bas[nh,KAPPA_OF]  = 4
bas[nh,NPRIM_OF]  = 1
bas[nh,NCTR_OF ]  = 1
bas[nh,PTR_EXP ]  = off
env[off+0] = .5
bas[nh,PTR_COEFF] = off + 1
env[off+1] = 1.
off = off + 2
nh += 1

nbas = nh
# basis with kappa < 0
n = off - off0
for i in range(n):
    env[off+i] = env[off0+i]
for i in range(nh):
        bas[i+nh,ATOM_OF ] = bas[i,ATOM_OF ]
        bas[i+nh,ANG_OF  ] = bas[i,ANG_OF  ] - 1
        bas[i+nh,KAPPA_OF] =-bas[i,KAPPA_OF]
        bas[i+nh,NPRIM_OF] = bas[i,NPRIM_OF]
        bas[i+nh,NCTR_OF ] = bas[i,NCTR_OF ]
        bas[i+nh,PTR_EXP ] = bas[i,PTR_EXP ]  + n
        bas[i+nh,PTR_COEFF]= bas[i,PTR_COEFF] + n
        env[bas[i+nh,PTR_COEFF]] /= 2 * env[bas[i,PTR_EXP]]
env[bas[5,PTR_COEFF]+0] = env[bas[1,PTR_COEFF]+0] / (2 * env[bas[1,PTR_EXP]+0])
env[bas[5,PTR_COEFF]+1] = env[bas[1,PTR_COEFF]+1] / (2 * env[bas[1,PTR_EXP]+1])
env[bas[5,PTR_COEFF]+2] = env[bas[1,PTR_COEFF]+2] / (2 * env[bas[1,PTR_EXP]+0])
env[bas[5,PTR_COEFF]+3] = env[bas[1,PTR_COEFF]+3] / (2 * env[bas[1,PTR_EXP]+1])

mol1 = gto.Mole()
mol1._atm = atm
mol1._bas = bas[:nh*2]
mol1._env = env
mol1._built = True


numpy.random.seed(1)
ngrids = 2000
coords = numpy.random.random((ngrids,3))
coords = (coords-.5)**2 * 80

def tearDownModule():
    global mol, mol1, coords
    del mol, mol1, coords


class KnownValues(unittest.TestCase):
    def test_sph(self):
        ao = eval_gto(mol, 'GTOval_sph', coords)
        self.assertAlmostEqual(lib.fp(ao), -6.8109234394857712, 9)

    def test_cart(self):
        ao = eval_gto(mol, 'GTOval_cart', coords)
        self.assertAlmostEqual(lib.fp(ao), -16.384888666900274, 9)

    def test_ip_cart(self):
        ao = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3)
        self.assertAlmostEqual(lib.fp(ao), 94.04527465181198, 9)

    def test_sph_deriv1(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv1', coords, comp=4)
        self.assertAlmostEqual(lib.fp(ao), -45.129633361047482, 9)

    def test_sph_deriv2(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv2', coords, comp=10)
        self.assertAlmostEqual(lib.fp(ao), -88.126901222477954, 9)

    def test_sph_deriv3(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv3', coords, comp=20)
        self.assertAlmostEqual(lib.fp(ao), -402.84257273073263, 9)

    def test_sph_deriv4(self):
        ao = eval_gto(mol, 'GTOval_sph_deriv4', coords, comp=35)
        self.assertAlmostEqual(lib.fp(ao), 4933.0635429300246, 9)

    def test_shls_slice(self):
        ao0 = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3)
        ao1 = ao0[:,:,14:77]
        ao = eval_gto(mol, 'GTOval_ip_cart', coords, comp=3, shls_slice=(7, 19))
        self.assertAlmostEqual(abs(ao-ao1).sum(), 0, 9)

    def test_ig_sph(self):
        ao = eval_gto(mol, 'GTOval_ig_sph', coords, comp=3)
        self.assertAlmostEqual(lib.fp(ao), 8.6601301646129052, 9)

    def test_ipig_sph(self):
        ao = eval_gto(mol, 'GTOval_ipig_sph', coords, comp=9)
        self.assertAlmostEqual(lib.fp(ao), -53.56608497643537, 9)

    def test_spinor(self):
        ao = eval_gto(mol, 'GTOval_spinor', coords)
        self.assertAlmostEqual(lib.fp(ao), -4.5941099464020079+5.9444339000526707j, 9)

    def test_sp_spinor(self):
        ao = eval_gto(mol, 'GTOval_sp_spinor', coords)
        self.assertAlmostEqual(lib.fp(ao), 26.212937567473656-68.970076521029782j, 9)

        numpy.random.seed(1)
        rs = numpy.random.random((213,3))
        rs = (rs-.5)**2 * 30
        ao1 = eval_gto(mol1, 'GTOval_spinor', rs, shls_slice=(0,mol1.nbas//2))
        ao2 = eval_gto(mol1, 'GTOval_sp_spinor', rs, shls_slice=(mol1.nbas//2,mol1.nbas))
        self.assertAlmostEqual(abs(ao1-ao2*1j).sum(), 0, 9)
#
#        t = gto.cart2j_kappa(0, 2)
#        aonr = eval_gto(mol1, 'GTOval_cart', rs, shls_slice=(1,2))
#        aa = numpy.zeros((2,12))
#        aa[:1,:6] = aonr[:,:6]
#        aa[1:,6:] = aonr[:,:6]
#        print aa.dot(t[:,:4])
#
#        t = gto.cart2j_kappa(0, 1)/0.488602511902919921
#        aonr = eval_gto(mol1, 'GTOval_ip_cart', rs, comp=3, shls_slice=(mol1.nbas//2+1,mol1.nbas//2+2))
#        print 'x', aonr[0,0,:3]
#        print 'y', aonr[1,0,:3]
#        print 'z', aonr[2,0,:3]
#        aa = numpy.zeros((3,2,6),dtype=numpy.complex128)
#        aa[0,:1,3:] = aonr[0,0,:3]
#        aa[0,1:,:3] = aonr[0,0,:3]
#        aa[1,:1,3:] =-aonr[1,0,:3]*1j
#        aa[1,1:,:3] = aonr[1,0,:3]*1j
#        aa[2,:1,:3] = aonr[2,0,:3]
#        aa[2,1:,3:] =-aonr[2,0,:3]
#        aa = (aa[0]*-1j + aa[1]*-1j + aa[2]*-1j)
#        print 'p',aa.dot(t[:,2:])*1j

    def test_ip_spinor(self):
        ao = eval_gto(mol, 'GTOval_ip_spinor', coords, comp=3)
        self.assertAlmostEqual(lib.fp(ao), -52.516545034166775+24.765350351395604j, 9)

    def test_ipsp_spinor(self):
        ao = eval_gto(mol, 'GTOval_ipsp_spinor', coords, comp=3)
        self.assertAlmostEqual(lib.fp(ao), -159.94403505490939+400.80148912086418j, 9)

        numpy.random.seed(1)
        rs = numpy.random.random((213,3))
        rs = (rs-.5)**2 * 30
        ao1 = eval_gto(mol1, 'GTOval_ip_spinor', rs, comp=3, shls_slice=(0,mol1.nbas//2))
        ao2 = eval_gto(mol1, 'GTOval_ipsp_spinor', rs, comp=3, shls_slice=(mol1.nbas//2,mol1.nbas))
        self.assertAlmostEqual(abs(ao1-ao2*1j).sum(), 0, 9)

    def test_int1e_grids(self):
        mol1 = gto.M(atom='''
O 0.5 0.5 0.
H  1.  1.2 0.
H  0.  0.  1.3''', basis='ccpvtz')
        ngrids = 201
        grids = numpy.random.random((ngrids, 3)) * 12 - 5
        fmol = gto.fakemol_for_charges(grids)
        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e').transpose(2,0,1)
        j3c = mol1.intor('int1e_grids', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_cart').transpose(2,0,1)
        j3c = mol1.intor('int1e_grids_cart', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_spinor').transpose(2,0,1)
        j3c = mol1.intor('int1e_grids_spinor', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

    def test_range_separated_coulomb_int1e_grids(self):
        mol1 = gto.M(atom='''
O 0.5 0.5 0.
H  1.  1.2 0.
H  0.  0.  1.3''', basis='ccpvtz')
        ngrids = 201
        grids = numpy.random.random((ngrids, 3)) * 12 - 5
        fmol = gto.fakemol_for_charges(grids)

        with mol1.with_range_coulomb(.8):
            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_cart').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids_cart', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_spinor').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids_spinor', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

        with mol1.with_range_coulomb(-.8):
            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_cart').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids_cart', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

            ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_spinor').transpose(2,0,1)
            j3c = mol1.intor('int1e_grids_spinor', grids=grids)
            self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

    def test_int1e_grids_ip(self):
        ngrids = 201
        grids = numpy.random.random((ngrids, 3)) * 12 - 5
        fmol = gto.fakemol_for_charges(grids)
        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_ip1').transpose(0,3,1,2)
        j3c = mol1.intor('int1e_grids_ip', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_ip1_cart').transpose(0,3,1,2)
        j3c = mol1.intor('int1e_grids_ip_cart', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

        ref = df.incore.aux_e2(mol1, fmol, intor='int3c2e_ip1_spinor').transpose(0,3,1,2)
        j3c = mol1.intor('int1e_grids_ip_spinor', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 12)

    def test_int1e_grids_spvsp(self):
        ngrids = 201
        grids = numpy.random.random((ngrids, 3)) * 12 - 5
        fmol = gto.fakemol_for_charges(grids)
        ref = df.r_incore.aux_e2(mol, fmol, intor='int3c2e_spsp1_spinor').transpose(2,0,1)
        j3c = mol.intor('int1e_grids_spvsp_spinor', grids=grids)
        self.assertAlmostEqual(abs(j3c - ref).max(), 0, 11)


if __name__ == '__main__':
    print('Full Tests for grid_ao and grids_ints')
    unittest.main()
