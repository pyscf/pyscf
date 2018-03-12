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
from pyscf import lo
from pyscf.mp import mp2f12_slow as mp2f12

mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = 'ccpvdz'
mol.build()


class KnowValues(unittest.TestCase):
    def test_find_cabs(self):
        auxmol = mol.copy()
        auxmol.basis = 'def2-tzvp'
        auxmol.build(False, False)
        cabs_mol, cabs_coeff = mp2f12.find_cabs(mol, auxmol)
        nao = mol.nao_nr()
        nca = cabs_coeff.shape[0]
        c1 = numpy.zeros((nca,nao))
        c1[:nao,:nao] = lo.orth.lowdin(mol.intor('int1e_ovlp_sph'))
        c = numpy.hstack((c1,cabs_coeff))
        s = reduce(numpy.dot, (c.T, cabs_mol.intor('int1e_ovlp_sph'), c))
        self.assertAlmostEqual(numpy.linalg.norm(s-numpy.eye(c.shape[1])), 0, 8)


if __name__ == "__main__":
    print("Full Tests for mp2-f12")
    unittest.main()

