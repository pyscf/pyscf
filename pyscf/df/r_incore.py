#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import numpy
from pyscf import gto
from pyscf.df import incore

# (ij|L)
def aux_e2(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=None, hermi=0):
    intor, comp = gto.moleintor._get_intor_and_comp(mol._add_suffix(intor), comp)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    ao_loc1 = mol.ao_loc_2c()
    ao_loc2 = auxmol.ao_loc_nr('ssc' in intor)
    nao = ao_loc1[-1]
    ao_loc = numpy.append(ao_loc1, ao_loc2[1:]+nao)
    out = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice,
                                  comp, aosym, ao_loc=ao_loc)
    return out

# (L|ij)
def aux_e1(mol, auxmol, intor='int3c2e_spinor', aosym='s1', comp=1, hermi=0):
    raise NotImplementedError


def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e_spinor', aosym='s1', int2c='int2c2e_sph', comp=1,
                 verbose=0):
    return incore.cholesky_eri_debug(mol, auxbasis, auxmol, int3c, aosym, int2c,
                                     comp, verbose, aux_e2)
