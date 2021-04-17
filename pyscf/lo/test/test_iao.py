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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import pbc
from pyscf.lo import iao

mol = gto.Mole()
mol.atom = '''
     O    0.   0.       0
     H    0.   -0.757   0.587
     H    0.   0.757    0.587'''
mol.basis = 'unc-sto3g'
mol.verbose = 5
mol.output = '/dev/null'
mol.build()

mol_lindep = gto.Mole()
mol_lindep.atom = '''
    O              0.   0.       0
    H              0.   -0.757   0.587
    H              0.   0.757    0.587
    ghost-O     1e-8    0.       0
    ghost-H     1e-8    -0.757   0.587
    ghost-H     1e-8    0.757    0.587
    '''
mol_lindep.basis = 'unc-sto3g'
mol_lindep.verbose = 5
mol_lindep.output = '/dev/null'
mol_lindep.build()

cell = pbc.gto.Cell()
cell.atom = "C 0 0 0"
cell.a = 2.0*numpy.eye(3)
cell.basis = "gth-dzv"
cell.pseudo = "gth-pade"
cell.verbose = 0
cell.output = '/dev/null'
cell.build()

cell_lindep = pbc.gto.Cell()
cell_lindep.atom = "C 0 0 0 ; ghost-C 0 0 1e-7"
cell_lindep.a = 2.0*numpy.eye(3)
cell_lindep.basis = "gth-dzv"
cell_lindep.pseudo = "gth-pade"
cell_lindep.verbose = 0
cell_lindep.output = '/dev/null'
cell_lindep.build()

class KnownValues(unittest.TestCase):

    def test_fast_iao_mulliken_pop(self):
        mf = scf.RHF(mol).run()
        a = iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p), 0.56812564587009806, 5)

        mf = scf.UHF(mol).run()
        p,chg = iao.fast_iao_mullikan_pop(mol, mf.make_rdm1(), a)
        self.assertAlmostEqual(lib.finger(p[0]+p[1]), 0.56812564587009806, 5)

    def test_iao_lindep(self):
        # No linear dependency
        mf = scf.RHF(mol).run()
        a = iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])

        mf2 = scf.RHF(mol_lindep)
        mf2 = scf.remove_linear_dep(mf2)
        mf2.kernel()
        a2 = iao.iao(mol_lindep, mf2.mo_coeff[:,mf2.mo_occ>0])

        # The underlying AO bases are different; compare projectors instead
        p = numpy.linalg.multi_dot((a.T, mf.get_ovlp(), a))
        p2 = numpy.linalg.multi_dot((a2.T, mf2.get_ovlp(), a2))
        self.assertTrue(numpy.allclose(p, p2, atol=1e-7, rtol=0))

    def test_iao_lindep_pbc(self):
        kpts = cell.make_kpts([1,1,2])
        mf = pbc.scf.KRHF(cell, kpts)
        mf.kernel()
        orbocc = [mf.mo_coeff[k][:,mf.mo_occ[k]>0] for k in range(len(kpts))]
        a = iao.iao(cell, orbocc, minao="gth-szv", kpts=kpts)

        mf2= pbc.scf.KRHF(cell_lindep, kpts)
        mf2 = scf.remove_linear_dep(mf2)
        mf2.kernel()
        orbocc = [mf2.mo_coeff[k][:,mf2.mo_occ[k]>0] for k in range(len(kpts))]
        a2 = iao.iao(cell_lindep, orbocc, minao="gth-szv", kpts=kpts)

        # The underlying AO bases are different; compare projectors instead
        for k in range(len(kpts)):
            p = numpy.linalg.multi_dot((a[k].T, mf.get_ovlp()[k], a[k]))
            p2 = numpy.linalg.multi_dot((a2[k].T, mf2.get_ovlp()[k], a2[k]))
            self.assertTrue(numpy.allclose(p, p2, atol=1e-7, rtol=0))


if __name__ == "__main__":
    print("TODO: Test iao")
    unittest.main()



