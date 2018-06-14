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

import os
import ctypes
import unittest
import numpy
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

# FIXME

libri1 = lib.load_library('libri')

mol = gto.Mole()
mol.verbose = 0
mol.output = None#'out_h2o'
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = {'H': 'cc-pvdz',
             'O': 'cc-pvdz',}
mol.build()
rhf = scf.RHF(mol)
rhf.scf()


nao = mol.nao_nr()
naopair = nao*(nao+1)/2
c_atm = numpy.array(mol._atm, dtype=numpy.int32)
c_bas = numpy.array(mol._bas, dtype=numpy.int32)
c_env = numpy.array(mol._env)
natm = ctypes.c_int(c_atm.shape[0])
nbas = ctypes.c_int(c_bas.shape[0])
cintopt = ctypes.c_void_p()

class KnowValues(unittest.TestCase):
    def test_fill_auxe2(self):
        eriref = numpy.empty((nao,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    buf = gto.moleintor.getints_by_shell('cint3c2e_sph', (i,j,k),
                                                         c_atm, c_bas, c_env, 1)
                    di,dj,dk = buf.shape
                    eriref[ip:ip+di,jp:jp+dj,kp:kp+dk] = buf
                    kp += dk
                jp += dj
            ip += di

        intor = getattr(libri1, 'cint3c2e_sph')
        r_atm = numpy.vstack((c_atm, c_atm))
        r_bas = numpy.vstack((c_bas, c_bas))
        fdrv = getattr(libri1, 'RInr_3c2e_auxe2_drv')

        fill = getattr(libri1, 'RIfill_s1_auxe2')
        eri1 = numpy.empty((nao,nao,nao))
        fdrv(intor, fill,
             eri1.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(0), nbas, nbas, nbas, ctypes.c_int(1), cintopt,
             r_atm.ctypes.data_as(ctypes.c_void_p), natm,
             r_bas.ctypes.data_as(ctypes.c_void_p), nbas,
             c_env.ctypes.data_as(ctypes.c_void_p))
        self.assertTrue(numpy.allclose(eriref, eri1))

        fill = getattr(libri1, 'RIfill_s2ij_auxe2')
        eri1 = numpy.empty((naopair,nao))
        fdrv(intor, fill,
             eri1.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(0), nbas, nbas, nbas, ctypes.c_int(1), cintopt,
             r_atm.ctypes.data_as(ctypes.c_void_p), natm,
             r_bas.ctypes.data_as(ctypes.c_void_p), nbas,
             c_env.ctypes.data_as(ctypes.c_void_p))
        idx = numpy.tril_indices(nao)
        self.assertTrue(numpy.allclose(eriref[idx[0],idx[1]], eri1))



if __name__ == '__main__':
    print('Full Tests for df')
    unittest.main()


