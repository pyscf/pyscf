#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto
from pyscf import lib
from pyscf import dft
from pyscf import scf
try:
    from pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None

def setUpModule():
    global h2o, h2osym, h2o_cation, h2osym_cation
    h2o = gto.Mole()
    h2o.verbose = 5
    h2o.output = '/dev/null'
    h2o.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]

    h2o.basis = {"H": '6-31g', "O": '6-31g',}
    h2o.build()

    h2osym = gto.Mole()
    h2osym.verbose = 5
    h2osym.output = '/dev/null'
    h2osym.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]

    h2osym.basis = {"H": '6-31g', "O": '6-31g',}
    h2osym.symmetry = 1
    h2osym.build()

    h2o_cation = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
        charge = 1,
        spin = 1,
        basis = '631g')

    h2osym_cation = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]],
        symmetry = True,
        charge = 1,
        spin = 1,
        basis = '631g')

def tearDownModule():
    global h2o, h2osym, h2o_cation, h2osym_cation
    h2o.stdout.close()
    h2osym.stdout.close()
    h2o_cation.stdout.close()
    h2osym_cation.stdout.close()
    del h2o, h2osym, h2o_cation, h2osym_cation


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_nr_lda(self):
        method = dft.RKS(h2o)
        method.init_guess = 'atom' # initial guess problem, issue #2056
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.01330948329084, 8)

    def test_nr_b88vwn(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690247578608236, 8)

    def test_nr_b3lypg(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 8)
        g = method.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.035648772973075241, 6)

    def test_nr_b3lypg_direct(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.radi_method = dft.radi.gauss_chebyshev
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 8)
        method.direct_scf = False
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 8)

    def test_nr_uks_lsda(self):
        method = dft.UKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 8)

    def test_nr_uks_b3lypg(self):
        method = dft.UKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.927304010489976, 8)

    def test_nr_uks_b3lypg_direct(self):
        method = dft.UKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 8)

    def test_nr_uks_b3lypg_cart(self):
        mol1 = h2o.copy()
        mol1.basis = 'ccpvdz'
        mol1.charge = 1
        mol1.spin = 1
        mol1.cart = True
        mol1.build(0, 0)
        method = dft.UKS(mol1)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.968190479564399, 8)

    def test_nr_roks_lsda(self):
        method = dft.RKS(h2o_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 8)

    def test_nr_roks_b3lypg(self):
        method = dft.ROKS(h2o_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 8)
        g = method.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.10184251826412283, 6)

    def test_nr_roks_b3lypg_direct(self):
        method = dft.ROKS(h2o_cation)
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 8)

    def test_nr_gks_lsda(self):
        method = dft.GKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 8)

    def test_nr_gks_b3lypg(self):
        method = dft.GKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.902391377392391, 8)

    def test_nr_gks_b3lypg_direct(self):
        method = dft.GKS(h2o_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.902391377392391, 8)

#########
    def test_nr_symm_lda(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.01330948329084, 8)

    def test_nr_symm_pw91pw91(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'pw91, pw91'
        # Small change from libxc3 to libxc4
        self.assertAlmostEqual(method.scf(), -76.355310330095563, 7)

    def test_nr_symm_b88vwn(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690247578608236, 8)

    def test_nr_symm_b88vwn_df(self):
        method = dft.density_fit(dft.RKS(h2osym), 'weigend')
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690346887915879, 8)

    def test_nr_symm_xlyp(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -76.4174879445209, 8)

    def test_nr_symm_b3lypg(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 8)

    def test_nr_symm_b3lypg_direct(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.radi_method = dft.radi.gauss_chebyshev
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 8)
        method.direct_scf = False
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 8)

    def test_nr_symm_ub3lypg(self):
        method = dft.UKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 8)

    def test_nr_symm_uks_lsda(self):
        method = dft.UKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 8)

    def test_nr_symm_uks_b3lypg(self):
        method = dft.UKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.927304010489976, 8)

    def test_nr_symm_uks_b3lypg_direct(self):
        method = dft.UKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 8)

    def test_nr_symm_roks_lsda(self):
        method = dft.RKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 8)

    def test_nr_symm_roks_b3lypg(self):
        method = dft.ROKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 8)

    def test_nr_symm_roks_b3lypg_direct(self):
        method = dft.ROKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 8)

    def test_nr_mgga(self):
        method = dft.RKS(h2o)
        method.xc = 'm06l,m06l'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -76.3772366, 6)

    def test_nr_rks_vv10(self):
        method = dft.RKS(h2o)
        dm = method.get_init_guess()
        method.xc = 'wB97M_V'
        method.nlc = 'vv10'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (30, 86), "O": (30, 86),}
        method.nlcgrids.prune = None
        method.nlcgrids.atom_grid = {"H": (20, 50), "O": (20, 50),}
        method.dump_flags()
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 22.767504283729778, 8)

        method._eri = None
        method.max_memory = 0
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 22.767504283729778, 8)

    def test_nr_uks_vv10(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method.xc = 'wB97M_V'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (30, 86), "O": (30, 86),}
        method.nlcgrids.prune = None
        method.nlcgrids.atom_grid = {"H": (20, 50), "O": (20, 50),}
        method.dump_flags()
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc[0]), 22.767504283729778, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 22.767504283729778, 8)

        method._eri = None
        method.max_memory = 0
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc[0]), 22.767504283729778, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 22.767504283729778, 8)

    def test_nr_rks_rsh(self):
        method = dft.RKS(h2o)
        dm = method.get_init_guess()
        method.xc = 'wB97'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 23.16975737295899, 8)

    def test_nr_rks_nlc(self):
        method = dft.RKS(h2o)
        dm = method.get_init_guess()
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 22.767792068559917, 8)

        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc), 23.067046560473408, 8)

        method.nlc = False
        assert method.do_nlc() == False
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc), 23.05881308880983, 8)

    def test_nr_rks_nlc_small_memory_high_cost(self):
        method = dft.RKS(h2o)
        dm = method.get_init_guess()
        method._eri = None
        method.max_memory = 0
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 22.767792068559917, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc), 23.067046560473408, 8)

    def test_nr_rks_rsh_cart_high_cost(self):
        mol1 = h2o.copy()
        mol1.basis = 'ccpvdz'
        mol1.cart = True
        mol1.build(0, 0)
        method = dft.RKS(mol1)
        method.xc = 'B97M_V'
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.kernel(), -76.39753789383619, 8)

    def test_nr_uks_rsh(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method.xc = 'wB97'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc[0]), 23.16975737295899, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 23.16975737295899, 8)

    def test_nr_uks_nlc_high_cost(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc[0]), 22.767792068559917, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 22.767792068559917, 8)

        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc[0]), 23.067046560473408, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 23.067046560473408, 8)

    def test_nr_uks_nlc_small_memory_high_cost(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method._eri = None
        method.max_memory = 0
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc[0]), 22.767792068559917, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 22.767792068559917, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc[0]), 23.067046560473408, 8)
        self.assertAlmostEqual(lib.fp(vxc[1]), 23.067046560473408, 8)

    def test_nr_gks_rsh(self):
        method = dft.GKS(h2o)
        dm = method.get_init_guess()
        dm = dm + numpy.sin(dm)*.02j
        dm = dm + dm.conj().T
        method.xc = 'wB97'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 5.115622298912124+0j, 8)

    def test_nr_gks_nlc_high_cost(self):
        method = dft.GKS(h2o)
        dm = method.get_init_guess()
        dm = dm + numpy.sin(dm)*.02j
        dm = dm + dm.conj().T
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 3.172920887028461+0j, 8)

        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 2.0041673361905317+0j, 8)

    def test_nr_gks_nlc_small_memory_high_cost(self):
        method = dft.GKS(h2o)
        dm = method.get_init_guess()
        dm = dm + numpy.sin(dm)*.02j
        dm = dm + dm.conj().T
        method._eri = None
        method.max_memory = 0
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.fp(vxc), 3.172920887028461+0j, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.fp(vxc), 2.0041673361905317+0j, 8)

    def test_nr_rks_vv10_high_cost(self):
        method = dft.RKS(h2o)
        method.xc = 'wB97M_V'
        method.nlc = 'vv10'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.nlcgrids.prune = None
        method.nlcgrids.atom_grid = {"H": (40, 110), "O": (40, 110),}
        self.assertAlmostEqual(method.scf(), -76.352381513158718, 8)

    def test_nr_uks_vv10_high_cost(self):
        method = dft.UKS(h2o)
        method.xc = 'wB97M_V'
        method.nlc = 'vv10'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.nlcgrids.prune = None
        method.nlcgrids.atom_grid = {"H": (40, 110), "O": (40, 110),}
        self.assertAlmostEqual(method.scf(), -76.352381513158718, 8)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_dft_parser(self):
        from pyscf.scf import dispersion
        method = dft.RKS(h2o, xc='wb97m-d3bj')
        assert method.do_nlc() == False
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0007551366628786623, 9)
        fn_facs = method._numint.libxc.parse_xc(method.xc)
        assert fn_facs[1][0][0] == 531

        method = dft.RKS(h2o, xc='wb97m-d3bj')
        assert method.do_nlc() == False
        method.xc = 'wb97m-v'
        method.nlc = False
        method.disp = 'd3bj'
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0007551366628786623, 9)
        fn_facs = method._numint.libxc.parse_xc(method.xc)
        assert fn_facs[1][0][0] == 531

        method = dft.RKS(h2o, xc='wb97x-d3bj')
        assert method.do_nlc() == False
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0005697890844546384, 9)
        fn_facs = method._numint.libxc.parse_xc(method.xc)
        assert fn_facs[1][0][0] == 466

        method = dft.RKS(h2o, xc='b3lyp-d3bj')
        assert method.xc == 'b3lyp-d3bj'
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0005738788210828446, 9)
        fn_facs = method._numint.libxc.parse_xc(method.xc)
        assert fn_facs[1][0][0] == 402

        method = dft.RKS(h2o, xc='b3lyp-d3bjm2b')
        assert method.xc == 'b3lyp-d3bjm2b'
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0006949127588605776, 9)

        method = dft.RKS(h2o, xc='b3lyp-d3bjmatm')
        assert method.xc == 'b3lyp-d3bjmatm'
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0006949125270554931, 9)

        method = dft.UKS(h2o, xc='b3lyp-d3bjmatm')
        assert method.xc == 'b3lyp-d3bjmatm'
        e_disp = dispersion.get_dispersion(method)
        self.assertAlmostEqual(e_disp, -0.0006949125270554931, 9)

    def test_d3_warning_msg(self):
        mf = dft.RKS(h2o)
        mf.xc = 'wb97m-v'
        mf.nlc = True
        mf.disp = 'd3bj'
        with self.assertWarnsRegex(UserWarning, 'double counting'):
            mf.build()

    def test_camb3lyp_rsh_omega(self):
        mf = dft.RKS(h2o)
        mf.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        mf.run(xc='camb3lyp')
        self.assertAlmostEqual(mf.e_tot, -76.35549300028714, 9)

        mf1 = dft.RKS(h2o)
        mf1.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        mf1.run(xc='camb3lyp', omega=0.15)
        self.assertAlmostEqual(mf1.e_tot, -76.36649222362115, 9)

        mf2 = dft.RKS(h2o)
        mf2.xc='RSH(.15,0.65,-0.46) + 0.46*ITYH + .35*B88 + VWN5*0.19, LYP*0.81'
        mf2.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        mf2.kernel()
        self.assertAlmostEqual(mf1.e_tot, -76.36649222362115, 9)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_dispersion(self):
        mf1 = dft.RKS(h2o)
        mf1.xc = 'B3LYP'
        mf1.disp = 'd3bj'
        mf1.run(xc='B3LYP')
        self.assertAlmostEqual(mf1.e_tot, -76.38552043811778, 9)

        mf2 = dft.RKS(h2o)
        mf2.xc = 'B3LYP-d3bj'
        mf2.run(xc='B3LYP-d3bj')
        self.assertAlmostEqual(mf1.e_tot, mf2.e_tot, 8)

    def test_reset(self):
        mf = dft.RKS(h2o).newton()
        mf.reset(h2osym)
        self.assertTrue(mf.mol is h2osym)
        self.assertTrue(mf.grids.mol is h2osym)
        self.assertTrue(mf.nlcgrids.mol is h2osym)

    def test_init_guess_by_vsap(self):
        dm = dft.RKS(h2o).get_init_guess(key='vsap')
        self.assertAlmostEqual(lib.fp(dm), 1.7285100188309719, 9)

        dm = dft.ROKS(h2osym).get_init_guess(key='vsap')
        self.assertEqual(dm.ndim, 3)
        self.assertAlmostEqual(lib.fp(dm), 1.9698972986009409, 9)

        dm = dft.UKS(h2osym).init_guess_by_vsap()
        self.assertEqual(dm.ndim, 3)
        self.assertAlmostEqual(lib.fp(dm), 1.9698972986009409, 9)

    def test_init(self):
        mol_r = h2o
        mol_u = gto.M(atom='Li', spin=1, verbose=0)
        mol_r1 = gto.M(atom='H', spin=1, verbose=0)
        sym_mol_r = h2osym
        sym_mol_u = gto.M(atom='Li', spin=1, symmetry=1, verbose=0)
        sym_mol_r1 = gto.M(atom='H', spin=1, symmetry=1, verbose=0)
        self.assertTrue(isinstance(dft.RKS(mol_r), dft.rks.RKS))
        self.assertTrue(isinstance(dft.RKS(mol_u), dft.roks.ROKS))
        self.assertTrue(isinstance(dft.UKS(mol_r), dft.uks.UKS))
        self.assertTrue(isinstance(dft.ROKS(mol_r), dft.roks.ROKS))
        self.assertTrue(isinstance(dft.GKS(mol_r), dft.gks.GKS))
        self.assertTrue(isinstance(dft.KS(mol_r), dft.rks.RKS))
        self.assertTrue(isinstance(dft.KS(mol_u), dft.uks.UKS))
        self.assertTrue(isinstance(dft.DKS(mol_u), dft.dks.UDKS))

        self.assertTrue(isinstance(mol_r.RKS(), dft.rks.RKS))
        self.assertTrue(isinstance(mol_u.RKS(), dft.roks.ROKS))
        self.assertTrue(isinstance(mol_r.UKS(), dft.uks.UKS))
        self.assertTrue(isinstance(mol_r.ROKS(), dft.roks.ROKS))
        self.assertTrue(isinstance(mol_r.GKS(), dft.gks.GKS))
        self.assertTrue(isinstance(mol_r.KS(), dft.rks.RKS))
        self.assertTrue(isinstance(mol_u.KS(), dft.uks.UKS))
        self.assertTrue(isinstance(mol_u.DKS(), dft.dks.UDKS))
        #TODO: self.assertTrue(isinstance(dft.X2C(mol_r), x2c.dft.UKS))

    def test_to_hf(self):
        self.assertEqual(dft.RKS(h2o).to_rhf().__class__, scf.rhf.RHF)
        self.assertEqual(dft.RKS(h2o).to_uhf().__class__, scf.uhf.UHF)
        self.assertEqual(dft.RKS(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(dft.RKS(h2o).to_hf() .__class__, scf.rhf.RHF)
        self.assertEqual(dft.RKS(h2o).to_rks().__class__, dft.rks.RKS)
        self.assertEqual(dft.RKS(h2o).to_uks().__class__, dft.uks.UKS)
        self.assertEqual(dft.RKS(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(dft.UKS(h2o).to_rhf().__class__, scf.rhf.RHF)
        self.assertEqual(dft.UKS(h2o).to_uhf().__class__, scf.uhf.UHF)
        self.assertEqual(dft.UKS(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(dft.UKS(h2o).to_hf() .__class__, scf.uhf.UHF)
        self.assertEqual(dft.UKS(h2o).to_rks().__class__, dft.rks.RKS)
        self.assertEqual(dft.UKS(h2o).to_uks().__class__, dft.uks.UKS)
        self.assertEqual(dft.UKS(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(dft.GKS(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(dft.GKS(h2o).to_hf() .__class__, scf.ghf.GHF)
        self.assertEqual(dft.GKS(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(dft.RKS(h2o).density_fit().to_rhf().__class__, scf.rhf.RHF(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_uhf().__class__, scf.uhf.UHF(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_hf() .__class__, scf.rhf.RHF(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_rks().__class__, dft.rks.RKS(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_uks().__class__, dft.uks.UKS(h2o).density_fit().__class__)
        self.assertEqual(dft.RKS(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

        self.assertEqual(dft.UKS(h2o).density_fit().to_rhf().__class__, scf.rhf.RHF(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_uhf().__class__, scf.uhf.UHF(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_hf() .__class__, scf.uhf.UHF(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_rks().__class__, dft.rks.RKS(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_uks().__class__, dft.uks.UKS(h2o).density_fit().__class__)
        self.assertEqual(dft.UKS(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

        self.assertEqual(dft.GKS(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(dft.GKS(h2o).density_fit().to_hf() .__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(dft.GKS(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

    def test_to_ks(self):
        self.assertEqual(scf.RHF(h2o).to_rhf().__class__, scf.rhf.RHF)
        self.assertEqual(scf.RHF(h2o).to_uhf().__class__, scf.uhf.UHF)
        self.assertEqual(scf.RHF(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(scf.RHF(h2o).to_ks() .__class__, dft.rks.RKS)
        self.assertEqual(scf.RHF(h2o).to_rks().__class__, dft.rks.RKS)
        self.assertEqual(scf.RHF(h2o).to_uks().__class__, dft.uks.UKS)
        self.assertEqual(scf.RHF(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(scf.UHF(h2o).to_rhf().__class__, scf.rhf.RHF)
        self.assertEqual(scf.UHF(h2o).to_uhf().__class__, scf.uhf.UHF)
        self.assertEqual(scf.UHF(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(scf.UHF(h2o).to_ks() .__class__, dft.uks.UKS)
        self.assertEqual(scf.UHF(h2o).to_rks().__class__, dft.rks.RKS)
        self.assertEqual(scf.UHF(h2o).to_uks().__class__, dft.uks.UKS)
        self.assertEqual(scf.UHF(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(scf.GHF(h2o).to_ghf().__class__, scf.ghf.GHF)
        self.assertEqual(scf.GHF(h2o).to_ks() .__class__, dft.gks.GKS)
        self.assertEqual(scf.GHF(h2o).to_gks().__class__, dft.gks.GKS)

        self.assertEqual(scf.RHF(h2o).density_fit().to_rhf().__class__, scf.rhf.RHF(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_uhf().__class__, scf.uhf.UHF(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_ks() .__class__, dft.rks.RKS(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_rks().__class__, dft.rks.RKS(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_uks().__class__, dft.uks.UKS(h2o).density_fit().__class__)
        self.assertEqual(scf.RHF(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

        self.assertEqual(scf.UHF(h2o).density_fit().to_rhf().__class__, scf.rhf.RHF(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_uhf().__class__, scf.uhf.UHF(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_ks() .__class__, dft.uks.UKS(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_rks().__class__, dft.rks.RKS(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_uks().__class__, dft.uks.UKS(h2o).density_fit().__class__)
        self.assertEqual(scf.UHF(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

        self.assertEqual(scf.GHF(h2o).density_fit().to_ghf().__class__, scf.ghf.GHF(h2o).density_fit().__class__)
        self.assertEqual(scf.GHF(h2o).density_fit().to_ks() .__class__, dft.gks.GKS(h2o).density_fit().__class__)
        self.assertEqual(scf.GHF(h2o).density_fit().to_gks().__class__, dft.gks.GKS(h2o).density_fit().__class__)

if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
