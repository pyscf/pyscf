# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from unittest import mock
import numpy
from pyscf import gto
from pyscf.sgx import sgx


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        self.sgx = sgx.SGX(self.mol)

    def check_full_dm(self, densities):
        delta_dm = numpy.ones_like(densities[1]) * 1e-5
        for density in densities:
            with self.subTest(full_dm_is_none=density is None):
                self.sgx._full_dm = density

                def get_k(rsh, dm, *args):
                    self.assertIs(rsh._full_dm, density)
                    # Full D supplies the screening bound; K still acts on delta D.
                    self.assertIs(dm, delta_dm)
                    self.assertEqual(rsh.mol.omega, -0.11)
                    return numpy.zeros_like(dm)

                with mock.patch.object(sgx.df_jk, 'get_j', return_value=None), \
                        mock.patch.object(sgx.sgx_jk, 'get_k_only', side_effect=get_k) as build_k:
                    self.sgx.get_jk(delta_dm, omega=-0.11)
                build_k.assert_called_once()
                self.assertIs(self.sgx._full_dm, density)
                self.assertIsNone(self.sgx._rsh_df['-0.110000']._full_dm)
                self.assertEqual(self.mol.omega, 0)

    def test_full_dm_rks(self):
        dm1 = numpy.eye(self.mol.nao)
        dm2 = dm1 * 0.9
        self.check_full_dm([None, dm1, dm2, None])

    def test_full_dm_uks(self):
        dm1 = numpy.array([numpy.eye(self.mol.nao), numpy.eye(self.mol.nao) * 0.5])
        dm2 = dm1 * 0.9
        # The first RSH copy must not retain a density inherited from its parent.
        self.check_full_dm([dm1, dm2, None])

    def test_full_dm_exception(self):
        dm = numpy.eye(self.mol.nao)
        self.sgx._full_dm = dm
        with mock.patch.object(sgx.df_jk, 'get_j', return_value=None), \
                mock.patch.object(sgx.sgx_jk, 'get_k_only', side_effect=RuntimeError('K failed')):
            with self.assertRaisesRegex(RuntimeError, 'K failed'):
                self.sgx.get_jk(dm * 1e-5, omega=-0.11)
        rsh = self.sgx._rsh_df['-0.110000']
        self.assertIsNone(rsh._full_dm)
        self.assertIs(self.sgx._full_dm, dm)
        self.assertEqual(self.mol.omega, 0)

        # Reusing a cached object must also restore its previous context on error.
        previous_dm = dm * 0.8
        rsh._full_dm = previous_dm
        with mock.patch.object(sgx.df_jk, 'get_j', return_value=None), \
                mock.patch.object(sgx.sgx_jk, 'get_k_only', side_effect=RuntimeError('K failed')):
            with self.assertRaisesRegex(RuntimeError, 'K failed'):
                self.sgx.get_jk(dm * 1e-5, omega=-0.11)
        self.assertIs(rsh._full_dm, previous_dm)
        self.assertIs(self.sgx._full_dm, dm)
        self.assertEqual(self.mol.omega, 0)

    def test_full_dm_separate_omega(self):
        dm = numpy.eye(self.mol.nao)
        delta_dm = dm * 1e-5
        with mock.patch.object(sgx.df_jk, 'get_j', return_value=None), \
                mock.patch.object(sgx.sgx_jk, 'get_k_only', return_value=delta_dm) as build_k:
            self.sgx.get_jk(delta_dm, omega=0.2)
            long_range = build_k.call_args.args[0]
            self.sgx._full_dm = dm

            def get_k(rsh, delta, *args):
                self.assertIs(rsh._full_dm, dm)
                self.assertIsNone(long_range._full_dm)
                self.assertIs(delta, delta_dm)
                return delta

            build_k.side_effect = get_k
            self.sgx.get_jk(delta_dm, omega=-0.11)
        short_range = build_k.call_args.args[0]
        self.assertIsNot(long_range, short_range)
        self.assertIsNone(short_range._full_dm)
        self.assertIsNone(long_range._full_dm)


if __name__ == '__main__':
    unittest.main()
