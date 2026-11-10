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


class ForwardingTests(unittest.TestCase):
    def setUp(self):
        self.mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
        self.sgx = sgx.SGX(self.mol)
        self.sgx.dfj = False
        self.sgx.direct_j = True
        self.dm = numpy.eye(self.mol.nao)

    def check_forwarding(self, with_j, with_k):
        vhfopt = object()
        direct_scf_tol = 2e-9
        expected_j = self.dm * 2
        expected_k = self.dm * 3
        with mock.patch.object(sgx._vhf, 'direct', return_value=(expected_j, None)) as build_j, \
                mock.patch.object(sgx.sgx_jk, 'get_k_only', return_value=expected_k) as build_k:
            vj, vk = self.sgx.get_jk(self.dm, hermi=0, vhfopt=vhfopt,
                                   with_j=with_j, with_k=with_k,
                                   direct_scf_tol=direct_scf_tol, omega=-0.11)
        self.assertIs(vj, expected_j if with_j else None)
        self.assertIs(vk, expected_k if with_k else None)
        if with_j:
            build_j.assert_called_once()
            self.assertIs(build_j.call_args.args[0], self.dm)
            self.assertIs(build_j.call_args.args[4], vhfopt)
            self.assertEqual(build_j.call_args.args[5], 0)
        else:
            build_j.assert_not_called()
        if with_k:
            build_k.assert_called_once()
            self.assertIs(build_k.call_args.args[1], self.dm)
            self.assertEqual(build_k.call_args.args[2:], (0, direct_scf_tol))
        else:
            build_k.assert_not_called()
        self.assertEqual(self.mol.omega, 0)

    def test_j_only(self):
        self.check_forwarding(with_j=True, with_k=False)

    def test_k_only(self):
        self.check_forwarding(with_j=False, with_k=True)

    def test_j_and_k(self):
        self.check_forwarding(with_j=True, with_k=True)

    def test_df_j_tolerance(self):
        self.sgx.dfj = True
        with mock.patch.object(sgx.df_jk, 'get_j', return_value=self.dm) as build_j, \
                mock.patch.object(sgx.sgx_jk, 'get_k_only') as build_k:
            vj, vk = self.sgx.get_jk(self.dm, hermi=0, with_k=False,
                                   direct_scf_tol=2e-9, omega=0.2)
        self.assertIs(vj, self.dm)
        self.assertIsNone(vk)
        build_j.assert_called_once()
        self.assertIs(build_j.call_args.args[1], self.dm)
        self.assertEqual(build_j.call_args.args[2:], (0, 2e-9))
        build_k.assert_not_called()


if __name__ == '__main__':
    unittest.main()
