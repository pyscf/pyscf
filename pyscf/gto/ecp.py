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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Effective core potential (ECP)

This module exposes some ecp integration functions from the C implementation.

Reference for ecp integral computation
* Analytical integration
J. Chem. Phys. 65, 3826
J. Chem. Phys. 111, 8778
J. Comput. Phys. 44, 289

* Numerical integration
J. Comput. Chem. 27, 1009
Chem. Phys. Lett. 296, 445
'''


import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.data.elements import ELEMENTS

libecp = moleintor.libcgto
libecp.ECPscalar_cache_size.restype = ctypes.c_int

def type1_by_shell(mol, shls, cart=False):
    li = mol.bas_angular(shls[0])
    lj = mol.bas_angular(shls[1])
    if cart:
        fn = libecp.ECPtype1_cart
        di = (li+1)*(li+2)//2 * mol.bas_nctr(shls[0])
        dj = (lj+1)*(lj+2)//2 * mol.bas_nctr(shls[1])
    else:
        fn = libecp.ECPtype1_sph
        di = (li*2+1) * mol.bas_nctr(shls[0])
        dj = (lj*2+1) * mol.bas_nctr(shls[1])
    cache_size = libecp.ECPscalar_cache_size(
        ctypes.c_int(1), (ctypes.c_int*2)(*shls),
        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        mol._env.ctypes.data_as(ctypes.c_void_p))
    cache = numpy.empty(cache_size)
    buf = numpy.zeros((di,dj), order='F')

    fn(buf.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls),
       mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(len(mol._ecpbas)),
       mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
       mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
       mol._env.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
       cache.ctypes.data_as(ctypes.c_void_p))
    return buf

def type2_by_shell(mol, shls, cart=False):
    li = mol.bas_angular(shls[0])
    lj = mol.bas_angular(shls[1])
    if cart:
        fn = libecp.ECPtype2_cart
        di = (li+1)*(li+2)//2 * mol.bas_nctr(shls[0])
        dj = (lj+1)*(lj+2)//2 * mol.bas_nctr(shls[1])
    else:
        fn = libecp.ECPtype2_sph
        di = (li*2+1) * mol.bas_nctr(shls[0])
        dj = (lj*2+1) * mol.bas_nctr(shls[1])
    cache_size = libecp.ECPscalar_cache_size(
        ctypes.c_int(1), (ctypes.c_int*2)(*shls),
        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        mol._env.ctypes.data_as(ctypes.c_void_p))
    cache = numpy.empty(cache_size)
    buf = numpy.zeros((di,dj), order='F')

    fn(buf.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls),
       mol._ecpbas.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(len(mol._ecpbas)),
       mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
       mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
       mol._env.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
       cache.ctypes.data_as(ctypes.c_void_p))
    return buf

AS_ECPBAS_OFFSET= 18
AS_NECPBAS      = 19
def so_by_shell(mol, shls):
    '''Spin-orbit coupling ECP in spinor basis
    i/2 <Pauli_matrix dot l U(r)>
    '''
    li = mol.bas_angular(shls[0])
    lj = mol.bas_angular(shls[1])
    di = (li*4+2) * mol.bas_nctr(shls[0])
    dj = (lj*4+2) * mol.bas_nctr(shls[1])
    bas = numpy.vstack((mol._bas, mol._ecpbas))
    mol._env[AS_ECPBAS_OFFSET] = len(mol._bas)
    mol._env[AS_NECPBAS] = len(mol._ecpbas)
    buf = numpy.zeros((di,dj), order='F', dtype=numpy.complex128)
    cache = numpy.empty(buf.size*48+100000)
    fn = libecp.ECPso_spinor
    fn(buf.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(di, dj),
       (ctypes.c_int*2)(*shls),
       mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
       mol._env.ctypes.data_as(ctypes.c_void_p), lib.c_null_ptr(),
       cache.ctypes.data_as(ctypes.c_void_p))
    return buf

def core_configuration(nelec_core, atom_symbol=None):
    conf_dic = {
        0 : '0s0p0d0f',
        2 : '1s0p0d0f',
        10: '2s1p0d0f',
        18: '3s2p0d0f',
        28: '3s2p1d0f',
        36: '4s3p1d0f',
        46: '4s3p2d0f',
        54: '5s4p2d0f',
        60: '4s3p2d1f',
        68: '5s4p2d1f',
        78: '5s4p3d1f',
        92: '5s4p3d2f',
    }
    # Core configurations for f-in-core ECPs defined in the following references
    # 10.1007/BF00528565 , 10.1007/s00214-005-0629-0 , 10.1007/s00214-009-0584-2
    elements_4f = ELEMENTS[57:71]
    elements_5f = ELEMENTS[89:103]
    if atom_symbol in elements_4f:
        for i in range(46, 60):
            conf_dic[i] = '4s3p2d1f'
    if atom_symbol in elements_5f:
        for i in range(78, 92):
            conf_dic[i] = '5s4p3d2f'
    if nelec_core not in conf_dic:
        raise RuntimeError('Core configuration for %d core electrons is not available.' % nelec_core)
    coreshell = [int(x) for x in conf_dic[nelec_core][::2]]
    return coreshell


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M(atom='''
 Cu 0. 0. 0.
 H  0.  0. -1.56
 H  0.  0.  1.56
''',
                basis={'Cu':'lanl2dz', 'H':'sto3g'},
                ecp = {'cu':'lanl2dz'},
                #basis={'Cu':'crenbs', 'H':'sto3g'},
                #ecp = {'cu':'crenbs'},
                charge=-1,
                verbose=4)
    mf = scf.RHF(mol)
    print(mf.kernel(), -196.09477546034623)

    mol = gto.M(atom='''
 Na 0. 0. 0.
 H  0.  0.  1.
''',
                basis={'Na':'lanl2dz', 'H':'sto3g'},
                ecp = {'Na':'lanl2dz'},
                verbose=0)
    mf = scf.RHF(mol)
    print(mf.kernel(), -0.45002315562861461)

