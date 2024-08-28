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
from pyscf import dft
from pyscf.sgx import sgx
from pyscf.sgx import sgx_jk
import os

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = True

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

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
        #numpy.random.seed(1)
        #dm = numpy.random.random((nao,nao))
        #dm = dm + dm.T
        mf = scf.UHF(mol)
        dm = mf.get_init_guess()
        vjref, vkref = scf.hf.get_jk(mol, dm)

        sgxobj = sgx.SGX(mol)
        sgxobj.grids = sgx_jk.get_gridss(mol, 0, 1e-10)

        with lib.temporary_env(sgxobj, debug=False):
            vj, vk = sgx_jk.get_jk_favork(sgxobj, dm)
        #self.assertAlmostEqual(lib.finger(vj), -19.25235595827077,  9)
        #self.assertAlmostEqual(lib.finger(vk), -16.711443399467267, 9)
        with lib.temporary_env(sgxobj, debug=True):
            vj1, vk1 = sgx_jk.get_jk_favork(sgxobj, dm)
        self.assertAlmostEqual(abs(vj1-vj).max(), 0, 9)
        self.assertAlmostEqual(abs(vk1-vk).max(), 0, 9)
        self.assertAlmostEqual(abs(vjref-vj).max(), 0, 2)
        self.assertAlmostEqual(abs(vkref-vk).max(), 0, 2)

        with lib.temporary_env(sgxobj, debug=False):
            vj, vk = sgx_jk.get_jk_favorj(sgxobj, dm)
        #self.assertAlmostEqual(lib.finger(vj), -19.176378579757973, 9)
        #self.assertAlmostEqual(lib.finger(vk), -16.750915356787406, 9)
        with lib.temporary_env(sgxobj, debug=True):
            vj1, vk1 = sgx_jk.get_jk_favorj(sgxobj, dm)
        self.assertAlmostEqual(abs(vj1-vj).max(), 0, 9)
        self.assertAlmostEqual(abs(vk1-vk).max(), 0, 9)
        self.assertAlmostEqual(abs(vjref-vj).max(), 0, 2)
        self.assertAlmostEqual(abs(vkref-vk).max(), 0, 2)

    def test_dfj(self):
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

        mf = sgx.sgx_fit(scf.RHF(mol), 'weigend')
        mf.with_df.dfj = True
        mf.build()
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(lib.finger(vj), -19.100356543264645, 9)
        self.assertAlmostEqual(lib.finger(vk), -16.715352176119794, 9)

    def test_rsh_get_jk(self):
        mol = gto.M(verbose = 0,
            atom = 'H 0 0 0; H 0 0 1',
            basis = 'ccpvdz',
        )
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,nao,nao))
        sgxobj = sgx.SGX(mol)
        sgxobj.grids = sgx_jk.get_gridss(mol, 0, 1e-7)
        vj, vk = sgxobj.get_jk(dm, hermi=0, omega=1.1)
        self.assertAlmostEqual(lib.finger(vj), 4.783036401049238, 9)
        self.assertAlmostEqual(lib.finger(vk), 8.60666152195185 , 9)

        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=0, omega=1.1)
        self.assertAlmostEqual(abs(vj-vj1).max(), 0, 2)
        self.assertAlmostEqual(abs(vk-vk1).max(), 0, 2)


class PJunctionScreening(unittest.TestCase):
    @unittest.skip("computationally expensive test")
    def test_pjs(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(cwd, 'a12.xyz')
        mol = gto.M(atom=fname, basis='sto-3g')

        mf = dft.RKS(mol)
        mf.xc = 'PBE'
        mf.kernel()
        dm = mf.make_rdm1()

        mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
        mf.with_df.dfj = True
        mf.build()
        en0 = mf.energy_tot(dm=dm)
        en0scf = mf.kernel()

        # Turn on P-junction screening. dfj must also be true.
        mf.with_df.pjs = True
        mf.direct_scf_tol = 1e-10
        mf.build()
        en1 = mf.energy_tot(dm=dm)
        en1scf = mf.kernel()

        self.assertAlmostEqual(abs(en1-en0), 0, 10)
        self.assertAlmostEqual(abs(en1scf-en0scf), 0, 10)


if __name__ == "__main__":
    print("Full Tests for sgx_jk")
    unittest.main()
