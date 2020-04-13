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

import warnings
import ctypes
import numpy
from pyscf import lib
from pyscf.gto.moleintor import make_loc

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c

libcgto = lib.load_library('libcgto')

def write_gto(mol):
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]

    eval_name, comp = _get_intor_and_comp(mol, "GTOval")
    ao_loc = make_loc(bas, eval_name)

    shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]

    ngrids = 1
    non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas),
                         dtype=numpy.int8)

    basisInfo = open("basisInfo.txt", "w")
    if ("cart" in eval_name): 
        basisInfo.write("cart\n")
    else:
        basisInfo.write("sph\n")
    basisInfo.write("%i\n"%(nao))
    basisInfo.write("%i\n%i \n"%(shls_slice[0], shls_slice[1]))
    basisInfo.write("%i\n"%(len(ao_loc)))
    for a in ao_loc:
        basisInfo.write("%i  "%(a))
    basisInfo.write("\n%i  %i\n"%(atm.shape[0], atm.shape[1]))
    for i in range(atm.shape[0]):
        for j in range(atm.shape[1]):
            basisInfo.write("%i  "%(atm[i][j]))
    basisInfo.write("\n%i  %i\n"%(bas.shape[0], bas.shape[1]))
    for i in range(bas.shape[0]):
        for j in range(bas.shape[1]):
            basisInfo.write("%i  "%(bas[i][j]))
    basisInfo.write("\n%i\n"%(len(env)))
    for e in env:
        basisInfo.write("%35.18e\n"%(e))

    basisInfo.write("%i\n"%(len(non0tab[0])))
    for e in non0tab[0]:
        basisInfo.write("%i  "%(e))

    basisInfo.close()

def _get_intor_and_comp(mol, eval_name, comp=None):
    if not ('_sph' in eval_name or '_cart' in eval_name or
            '_spinor' in eval_name):
        if mol.cart:
            eval_name = eval_name + '_cart'
        else:
            eval_name = eval_name + '_sph'

    if comp is None:
        if '_spinor' in eval_name:
            fname = eval_name.replace('_spinor', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[1]
        else:
            fname = eval_name.replace('_sph', '').replace('_cart', '')
            comp = _GTO_EVAL_FUNCTIONS.get(fname, (None,None))[0]
        if comp is None:
            warnings.warn('Function %s not found.  Set its comp to 1' % eval_name)
            comp = 1
    return eval_name, comp

_GTO_EVAL_FUNCTIONS = {
#   Functiona name          : (comp-for-scalar, comp-for-spinor)
    'GTOval'                : (1, 1 ),
    'GTOval_ip'             : (3, 3 ),
    'GTOval_ig'             : (3, 3 ),
    'GTOval_ipig'           : (3, 3 ),
    'GTOval_deriv0'         : (1, 1 ),
    'GTOval_deriv1'         : (4, 4 ),
    'GTOval_deriv2'         : (10,10),
    'GTOval_deriv3'         : (20,20),
    'GTOval_deriv4'         : (35,35),
    'GTOval_sp'             : (4, 1 ),
    'GTOval_ipsp'           : (12,3 ),
}
