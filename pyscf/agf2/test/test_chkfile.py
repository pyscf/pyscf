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

"""Checkpoint round trips without an AGF2 iteration or SCF calculation."""
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from pyscf import gto, lib
from pyscf.agf2 import chkfile, mpi_helper
from pyscf.agf2.aux_space import GreensFunction, SelfEnergy


class KnownValues(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = tmp.name + '/agf2.h5'
        self.mol = gto.M(atom='He', basis='sto-3g', verbose=0)
        self.gf = GreensFunction([-1., .5], [[.8, .6j]], chempot=.1)
        self.se = SelfEnergy([-2., 1.], [[.2, .3j]], chempot=.1)
        self.calculation = SimpleNamespace(
            mol=self.mol, chkfile=self.path, gf=self.gf, se=self.se,
            frozen=0, mo_energy=np.array([-.8]), mo_occ=np.array([2.]),
            mo_coeff=np.eye(1), e_1b=-2.8, e_2b=-.05, e_init=-.03,
            converged=True, _nmo=None, _nocc=None)

    def assert_aux_equal(self, actual, expected):
        self.assertIsInstance(actual, type(expected))
        np.testing.assert_array_equal(actual.energy, expected.energy)
        np.testing.assert_array_equal(actual.coupling, expected.coupling)
        self.assertEqual(actual.chempot, expected.chempot)

    def test_restricted_round_trip_and_overwrite(self):
        self.assertIs(chkfile.dump_agf2(self.calculation), self.calculation)
        mol, data = chkfile.load_agf2(self.path)
        self.assertEqual(mol.atom, self.mol.atom)
        self.assertEqual(mol.basis, self.mol.basis)
        self.assert_aux_equal(data['gf'], self.gf)
        self.assert_aux_equal(data['se'], self.se)
        for name in ('e_1b', 'e_2b', 'e_init', 'converged', 'frozen',
                     'mo_energy', 'mo_coeff', 'mo_occ'):
            np.testing.assert_array_equal(data[name], getattr(self.calculation, name))
        self.assertNotIn('_nmo', data)
        self.assertNotIn('nmom', data)
        # Rewriting the calculation must discard obsolete fields, while
        # preserving unrelated groups in the checkpoint.
        with h5py.File(self.path, 'a') as f:
            f['keep'] = [42]
            f['agf2/obsolete'] = [99]
        self.calculation.e_1b = -2.9
        chkfile.dump_agf2(self.calculation, chkfile=self.path)
        data = chkfile.load(self.path, 'agf2')
        self.assertNotIn('obsolete', data)
        self.assertEqual(data['e_1b'], -2.9)
        np.testing.assert_array_equal(chkfile.load(self.path, 'keep'), [42])

    def test_unrestricted_round_trip(self):
        for frozen in (1, [0], ([0], [])):
            with self.subTest(frozen=frozen):
                self.calculation.gf = (self.gf, self.gf.get_occupied())
                self.calculation.se = (self.se, self.se.get_virtual())
                self.calculation.frozen = frozen
                self.calculation.nmom = (1, 2)
                self.calculation._nmo = (1, 1)
                self.calculation._nocc = (1, 0)
                chkfile.dump_agf2(self.calculation)
                mol, data = chkfile.load_agf2(self.path)
                for name in ('gf', 'se'):
                    for actual, expected in zip(data[name], getattr(self.calculation, name)):
                        self.assert_aux_equal(actual, expected)
                self.assertEqual(data['nmom'], (1, 2))
                expected = frozen if isinstance(frozen, tuple) else (frozen, frozen)
                for actual, value in zip(data['frozen'], expected):
                    np.testing.assert_array_equal(actual, value)
                self.assertFalse({'gfa', 'gfb', 'sea', 'seb', 'frozena', 'frozenb'} & data.keys())

    def test_optional_moments(self):
        for nmom in ((None, 0), (0, None), (None, None)):
            with self.subTest(nmom=nmom):
                self.calculation.nmom = nmom
                chkfile.dump_agf2(self.calculation)
                mol, data = chkfile.load_agf2(self.path)
                self.assertEqual(data.get('nmom', (None, None)), nmom)

    def test_overrides_and_custom_key(self):
        self.calculation.nmom = (1, 2)
        chkfile.dump_agf2(self.calculation, key='restart', gf=self.gf.get_virtual(),
                         se=self.se.get_occupied(), frozen=[0], nmom=(2, 3),
                         mo_energy=[-.4], mo_coeff=[[.9]], mo_occ=[1.])
        data = chkfile.load(self.path, 'restart')
        np.testing.assert_array_equal(data['gf']['energy'], [.5])
        np.testing.assert_array_equal(data['se']['energy'], [-2.])
        np.testing.assert_array_equal(data['mo_energy'], [-.4])
        np.testing.assert_array_equal(data['mo_coeff'], [[.9]])
        np.testing.assert_array_equal(data['mo_occ'], [1.])
        self.assertEqual((data['ngf'], data['nse']), (2, 3))

    def test_absent_auxiliaries(self):
        self.calculation.gf = self.calculation.se = None
        chkfile.dump_agf2(self.calculation)
        mol, data = chkfile.load_agf2(self.path)
        self.assertNotIn('gf', data)
        self.assertNotIn('se', data)

    def test_nonroot_does_not_write(self):
        with lib.temporary_env(mpi_helper, rank=1):
            self.assertIs(chkfile.dump_agf2(self.calculation), self.calculation)
        self.assertFalse(h5py.is_hdf5(self.path))


if __name__ == '__main__':
    unittest.main()
