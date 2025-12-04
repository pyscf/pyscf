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


WATER_CLUSTER = """
O       89.814000000   100.835000000   101.232000000
H       89.329200000    99.976800000   101.063000000
H       89.151600000   101.561000000   101.414000000
O       98.804000000    98.512200000    97.758100000
H       99.782100000    98.646900000    97.916700000
H       98.421800000    99.326500000    97.321300000
O      108.070300000    98.516900000   100.438000000
H      107.172800000    98.878600000   100.690000000
H      108.194000000    98.592200000    99.448100000
"""


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
        mf.with_df.grids_level_i = 0
        mf.with_df.grids_level_f = 1
        mf.with_df.use_opt_grids = False
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

    def test_sgx_dot(self):
        """
        The purpose of this test is to call the SGX dot functions
        with both localized and random density matrices and make
        sure they work in comparison to lib.einsum
        """
        mol = gto.Mole()
        mol.build(
            verbose=0,
            atom=WATER_CLUSTER,
            basis="ccpvdz",
        )
        ao_loc = mol.ao_loc_nr()
        sgxobj = sgx.SGX(mol)
        sgxobj.grids = sgx_jk.get_gridss(mol, 0, 1e-7)
        grids = sgxobj.grids
        ao = gto.eval_gto(mol, "GTOval", grids.coords)
        wao = ao * grids.weights[:, None]
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dmr = numpy.random.random((nao, nao))
        dmr = numpy.abs(dmr)
        dmr = dmr + dmr.T
        dmr = dmr[None, :]
        sgxobj.get_jk(dmr)
        mf = scf.UHF(mol)
        dm0 = mf.get_init_guess()
        ib = (grids.weights.size // 2) // sgx_jk.BLKSIZE
        im = ib * sgx_jk.BLKSIZE
        shls_slice = (0, mol.nbas)
        mask = grids.non0tab
        tmp_switch = sgx_jk.SWITCH_SIZE
        sgx_jk.SWITCH_SIZE = 0
        for dm in [dm0, dmr]:
            pair_mask = sgx_jk._get_sgx_dm_mask(sgxobj, dm, ao_loc)
            outr = lib.einsum("gu,xuv->xvg", wao, dm)
            out = sgx_jk._sgxdot_ao_dm(wao, dm, mask, shls_slice, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = sgx_jk._sgxdot_ao_dm(wao, dm, None, shls_slice, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = numpy.empty_like(out)
            sgx_jk._sgxdot_ao_dm(wao, dm, mask, shls_slice, ao_loc, out)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = sgx_jk._sgxdot_ao_dm_sparse(wao, dm, mask, pair_mask, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = sgx_jk._sgxdot_ao_dm_sparse(wao, dm, mask, None, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = sgx_jk._sgxdot_ao_dm_sparse(wao, dm, None, None, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = numpy.empty_like(out)
            out = sgx_jk._sgxdot_ao_dm_sparse(wao, dm, mask, pair_mask, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)

            gv = lib.einsum("xuv,gv->xug", dm, ao)
            outr = lib.einsum("gu,xvg->xuv", wao, gv)
            out = sgx_jk._sgxdot_ao_gv(wao, gv, mask, shls_slice, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            sgx_jk._sgxdot_ao_gv(wao, gv, mask, shls_slice, ao_loc, out=out)
            out[:] *= 0.5
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            out = sgx_jk._sgxdot_ao_gv_sparse(ao, gv, grids.weights, mask, None, ao_loc)
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
            sgx_jk._sgxdot_ao_gv_sparse(ao, gv, grids.weights, mask, None, ao_loc, out=out)
            out[:] *= 0.5
            self.assertAlmostEqual(abs(outr - out).max(), 0, 13)
        sgx_jk.SWITCH_SIZE = tmp_switch


class PJunctionScreening(unittest.TestCase):
    # @unittest.skip("computationally expensive test")
    def test_pjs(self):
        cwd = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(cwd, 'a12.xyz')
        mol = gto.M(atom=fname, basis='def2-svp')

        mf = dft.RKS(mol)
        mf.xc = 'PBE'
        mf.kernel()
        mf.conv_tol = 1e-9
        dm = mf.make_rdm1()

        mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
        mf.with_df.dfj = True
        mf.with_df.grids_level_i = 1
        mf.with_df.grids_level_f = 1
        mf.with_df.use_opt_grids = False
        mf.direct_scf_tol = 1e-13
        mf.build()
        import time
        t0 = time.monotonic()
        en0 = mf.energy_tot(dm=dm)
        # en0scf = mf.kernel()
        t1 = time.monotonic()

        # Turn on P-junction screening. dfj must also be true.
        mf.with_df.pjs = True
        mf.build()
        t2 = time.monotonic()
        en1 = mf.energy_tot(dm=dm)
        # en1scf = mf.kernel()
        t3 = time.monotonic()

        print(t3 - t2, t1 - t0)
        self.assertAlmostEqual(abs(en1-en0), 0, 10)
        # self.assertAlmostEqual(abs(en1scf-en0scf), 0, 10)


if __name__ == "__main__":
    print("Full Tests for sgx_jk")
    unittest.main()
