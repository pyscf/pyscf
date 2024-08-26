#!/usr/bin/env python
# Copyright 2023 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.cc import ccd

def setUpModule():
    global mol, mf, mycc
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = ccd.CCD(mf)
    mycc.kernel()

def tearDownModule():
    global mol, mf, mycc
    mol.stdout.close()
    del mol, mf, mycc


class KnownValues(unittest.TestCase):
    def test_ccd(self):
        self.assertAlmostEqual(mycc.e_corr, -0.134712806, 7)

    def test_rdm(self):
        dm1 = mycc.make_rdm1()
        dm2 = mycc.make_rdm2()
        h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
        nmo = mf.mo_coeff.shape[1]
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
        e1 = np.einsum('ij,ji', h1, dm1)
        e1+= np.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 6)

if __name__ == "__main__":
    print("Full Tests for CCD")
    unittest.main()
