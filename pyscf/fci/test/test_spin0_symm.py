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

from functools import reduce
import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
import pyscf.symm

mol = gto.Mole()
mol.verbose = 0
mol.atom = '''
    O    0.  0.      0.
    H    0.  -0.757  0.587
    H    0.  0.757   0.587'''
mol.basis = 'sto-3g'
mol.symmetry = 'c2v'
mol.build()
m = scf.RHF(mol)
m.conv_tol_grad = 1e-8
ehf = m.scf()
norb = m.mo_coeff.shape[1]
nelec = mol.nelectron
h1e = reduce(numpy.dot, (m.mo_coeff.T, scf.hf.get_hcore(mol), m.mo_coeff))
g2e = ao2mo.incore.full(m._eri, m.mo_coeff)
orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, m.mo_coeff)
cis = fci.direct_spin0_symm.FCISolver(mol)
cis.orbsym = orbsym

numpy.random.seed(15)
na = fci.cistring.num_strings(norb, nelec//2)
ci0 = numpy.random.random((na,na))
ci0 = (ci0 + ci0.T) * .5

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=0)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=0)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 83.221199436109003, 9)
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=1)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=1)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.571087072474697, 9)
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=2)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=2)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 81.010497935954916, 9)
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=3)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=3)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.257163492625622, 9)

    def test_kernel(self):
        e, c = fci.direct_spin0_symm.kernel(h1e, g2e, norb, nelec, orbsym=orbsym)
        self.assertAlmostEqual(e, -84.200905534209554, 8)
        e = fci.direct_spin0_symm.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -84.200905534209554, 8)


if __name__ == "__main__":
    print("Full Tests for spin0 symm")
    unittest.main()



