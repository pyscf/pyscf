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
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf.sgx import sgx
from pyscf.sgx import sgx_jk

class KnownValues(unittest.TestCase):
    def test_sgx_jk(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis = 'ccpvdz',
        )
        nao = mol.nao
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T

        sgxobj = sgx.SGX(mol)
        sgxobj.grids = sgx_jk.get_gridss(mol, 0, 1e-10)

        with lib.temporary_env(sgxobj, debug=False):
            vj, vk = sgx_jk.get_jk_favork(sgxobj, dm)
        self.assertAlmostEqual(lib.finger(vj), -19.25235595827077,  9)
        self.assertAlmostEqual(lib.finger(vk), -16.711443399467267, 9)
        with lib.temporary_env(sgxobj, debug=True):
            vj1, vk1 = sgx_jk.get_jk_favork(sgxobj, dm)
        self.assertAlmostEqual(abs(vj1-vj).max(), 0, 9)
        self.assertAlmostEqual(abs(vk1-vk).max(), 0, 9)

        with lib.temporary_env(sgxobj, debug=False):
            vj, vk = sgx_jk.get_jk_favorj(sgxobj, dm)
        self.assertAlmostEqual(lib.finger(vj), -19.176378579757973, 9)
        self.assertAlmostEqual(lib.finger(vk), -16.750915356787406, 9)
        with lib.temporary_env(sgxobj, debug=True):
            vj1, vk1 = sgx_jk.get_jk_favorj(sgxobj, dm)
        self.assertAlmostEqual(abs(vj1-vj).max(), 0, 9)
        self.assertAlmostEqual(abs(vk1-vk).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for sgx_jk")
    unittest.main()

