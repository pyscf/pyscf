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
