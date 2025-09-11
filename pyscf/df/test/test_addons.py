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

import unittest
import itertools
import tempfile
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import df
from pyscf.gto.basis import bse

class KnownValues(unittest.TestCase):
    def test_aug_etb(self):
        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = 'cc-pvdz',
        )
        auxbasis = df.addons.aug_etb(mol)
        if df.addons.USE_VERSION_26_AUXBASIS:
            self.assertEqual(len(auxbasis['O']), 36)
            self.assertEqual(len(auxbasis['H']), 12)
        else:
            self.assertEqual(len(auxbasis['O']), 42)
            self.assertEqual(len(auxbasis['H']), 13)

        auxbasis = df.addons.autoaux(mol)
        self.assertEqual(len(auxbasis['O']), 37)
        # If basis-set-exchange pacakge is installed, this test will fail.
        # bse-0.9 produces 13 shells due to round-off errors.
        # The correct number of generated basis functions should be 12.
        if bse.basis_set_exchange:
            self.assertEqual(len(auxbasis['H']), 13)
        else:
            self.assertEqual(len(auxbasis['H']), 12)

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = ('cc-pvdz', [[4, (1., 1.)]])
        )
        auxbasis = df.addons.aug_etb(mol)
        if df.addons.USE_VERSION_26_AUXBASIS:
            self.assertEqual(len(auxbasis['O']), 36)
            self.assertEqual(len(auxbasis['H']), 12)
        else:
            self.assertEqual(len(auxbasis['O']), 59)
            self.assertEqual(len(auxbasis['H']), 16)

        auxbasis = df.addons.autoaux(mol)
        self.assertEqual(len(auxbasis['O']), 47)
        self.assertEqual(len(auxbasis['H']), 20)

    def test_make_auxbasis(self):
        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      H     0    -0.757   0.587
                      H     0    0.757    0.587
                GHOST-H     0    0.       0.587''',
            basis = 'cc-pvdz'
        )
        auxbasis = df.addons.make_auxbasis(mol)
        self.assertEqual(auxbasis['O'], 'cc-pvdz-jkfit')
        self.assertEqual(auxbasis['H'], 'cc-pvdz-jkfit')
        self.assertEqual(auxbasis['GHOST-H'], 'cc-pvdz-jkfit')
        self.assertEqual(len(auxbasis['GHOST-H']), len(auxbasis['H']))

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      H     0    -0.757   0.587
                      H     0    0.757    0.587''',
            basis = {'O':'cc-pvdz', 'H':'cc-pvtz'}
        )
        auxbasis = df.addons.make_auxbasis(mol)
        self.assertEqual(auxbasis['O'], 'cc-pvdz-jkfit')
        self.assertEqual(auxbasis['H'], 'cc-pvtz-jkfit')

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = {'default': 'cc-pvdz', 'O':'631g'}
        )
        auxmol = df.addons.make_auxmol(mol, auxbasis={'default':'ccpvdz-fit'})
        self.assertEqual(auxmol.nao_nr(), 146)
        auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
        self.assertEqual(auxbasis['O'], 'cc-pvdz-ri')
        self.assertEqual(auxbasis['H'], 'cc-pvdz-ri')

        mol = gto.M(
            verbose = 0,
            atom = '''O     0    0.       0.
                      1     0    -0.757   0.587
                      1     0    0.757    0.587''',
            basis = {'H': [[0,(1.,1.)], [1, (.5, 1.)]],
                     'O': ('631g', [[0, 0, (1., 1.)]])}
        )
        auxbasis = df.addons.make_auxbasis(mol)
        if df.addons.USE_VERSION_26_AUXBASIS:
            self.assertEqual(len(auxbasis['O']), 32)
            self.assertEqual(len(auxbasis['H']), 3)
        else:
            self.assertEqual(len(auxbasis['O']), 35)
            self.assertEqual(len(auxbasis['H']), 3)

    def test_default_auxbasis(self):
        mol = gto.M(atom='He 0 0 0; O 0 0 1', basis='ccpvdz')
        auxbasis = df.addons.make_auxbasis(mol)
        self.assertTrue(auxbasis['O'] == 'cc-pvdz-jkfit')
        self.assertTrue(isinstance(auxbasis['He'], list))

    @unittest.skipIf(bse.basis_set_exchange is None, "BSE library not installed.")
    def test_auto_aux(self):
        from pyscf.df.autoaux import autoaux, _auto_aux_element
        ref = flatten(bse.autoaux('STO-3G', 'K')['K'])
        mol = gto.M(atom='K', basis='STO-3G', spin=1)
        etb = _auto_aux_element(mol.atom_charge(0), mol._basis['K'])
        dat = flatten(gto.expand_etbs(etb))
        dat = np.array([float(f'{x:.6e}') for x in dat])
        self.assertAlmostEqual(abs(np.array(ref) - dat).max(), 0, 6)

        for key in ['H', 'Li', 'C', 'N', 'O', 'F', 'Si']:
            ref = flatten(bse.autoaux('cc-pVTZ', key)[key])
            basis = bse.get_basis('cc-pVTZ', key)[key]
            etb = _auto_aux_element(gto.charge(key), basis)
            dat = flatten(gto.expand_etbs(etb))
            dat = np.array([float(f'{x:.6e}') for x in dat])
            self.assertAlmostEqual(abs(np.array(ref) - dat).max(), 0, 6)

def flatten(lst):
    if not isinstance(lst, list):
        return [lst]
    elif len(lst) == 0:
        return lst
    return list(itertools.chain(*[flatten(l) for l in lst]))


if __name__ == "__main__":
    print("Full Tests for df.addons")
    unittest.main()
