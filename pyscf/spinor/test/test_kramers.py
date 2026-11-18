#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
import numpy
import scipy.linalg
from pyscf import gto, scf, x2c
from pyscf.spinor import kramers


def setUpModule():
    global mol
    mol = gto.M(atom='H 0 0 0; Br 0 0 1.41', basis='cc-pvdz-dk',
                verbose=5, output='/dev/null')


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def _partner(mol, c):
    '''Time reversal of AO coefficients c (two- or four-component), via the
    Kramers basis where T [p; q] = [-q*; p*].'''
    four_component = c.shape[0] != mol.nao_2c()
    idx = kramers.pair_sort_ao_idx(mol, four_component)
    ck = c[idx]
    n = ck.shape[0] // 2
    tk = numpy.concatenate((-ck[n:].conj(), ck[:n].conj()))
    out = numpy.empty_like(tk)
    out[idx] = tk
    return out


class KnownValues(unittest.TestCase):
    def _check(self, h, s):
        ref = scipy.linalg.eigh(h, s, eigvals_only=True)
        e, c = kramers.eigh(mol, h, s, iop=1)
        self.assertAlmostEqual(abs(e - ref).max() / abs(ref).max(), 0, 12)
        self.assertAlmostEqual(abs(c.conj().T.dot(s).dot(c) - numpy.eye(len(e))).max(), 0, 9)
        self.assertAlmostEqual(abs(c[:,1::2] - _partner(mol, c[:,0::2])).max(), 0, 12)
        e0, c0 = kramers.eigh(mol, h, s, iop=0)
        self.assertAlmostEqual(abs(e0 - numpy.hstack((e[0::2], e[1::2]))).max(), 0, 12)
        self.assertAlmostEqual(abs(c0 - numpy.hstack((c[:,0::2], c[:,1::2]))).max(), 0, 12)

    def test_two_component(self):
        mf = x2c.UHF(mol)
        self._check(mf.get_hcore(), mf.get_ovlp())

    def test_four_component(self):
        mf = scf.dhf.DHF(mol)
        self._check(mf.get_hcore(), mf.get_ovlp())


if __name__ == "__main__":
    print("Full Tests for spinor.kramers")
    unittest.main()
