#!/usr/bin/env python
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

import unittest
from pyscf import gto
from pyscf import lib
from pyscf import dft

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
    def test_nr_lda(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.01330948329084, 9)

    def test_nr_pw91pw91(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -76.355310330095563, 9)

    def test_nr_b88vwn(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690247578608236, 9)

    def test_nr_xlyp(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -76.4174879445209, 9)

    def test_nr_b3lypg(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)
        g = method.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.finger(g), -0.035648772973075241, 6)

    def test_nr_b3lypg_direct(self):
        method = dft.RKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.radi_method = dft.radi.gauss_chebyshev
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 9)
        method.direct_scf = False
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 9)

    def test_nr_ub3lypg(self):
        method = dft.UKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)
        g = method.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.finger(g), -0.035648777277847155, 6)

    def test_nr_uks_lsda(self):
        method = dft.UKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 9)

    def test_nr_uks_b3lypg(self):
        method = dft.UKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.927304010489976, 9)

    def test_nr_uks_b3lypg_direct(self):
        method = dft.UKS(h2o)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)

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
        self.assertAlmostEqual(method.scf(), -75.968190479564399, 9)

    def test_nr_roks_lsda(self):
        method = dft.RKS(h2o_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 9)

    def test_nr_roks_b3lypg(self):
        method = dft.ROKS(h2o_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)
        g = method.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.finger(g), -0.10184251826412283, 6)

    def test_nr_roks_b3lypg_direct(self):
        method = dft.ROKS(h2o_cation)
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)

#########
    def test_nr_symm_lda(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -76.01330948329084, 9)

    def test_nr_symm_pw91pw91(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -76.355310330095563, 9)

    def test_nr_symm_b88vwn(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690247578608236, 9)

    def test_nr_symm_b88vwn_df(self):
        method = dft.density_fit(dft.RKS(h2osym), 'weigend')
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -76.690346887915879, 9)

    def test_nr_symm_xlyp(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -76.4174879445209, 9)

    def test_nr_symm_b3lypg(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)

    def test_nr_symm_b3lypg_direct(self):
        method = dft.RKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.radi_method = dft.radi.gauss_chebyshev
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 9)
        method.direct_scf = False
        self.assertAlmostEqual(method.scf(), -76.384928823070567, 9)

    def test_nr_symm_ub3lypg(self):
        method = dft.UKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)

    def test_nr_symm_uks_lsda(self):
        method = dft.UKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 9)

    def test_nr_symm_uks_b3lypg(self):
        method = dft.UKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.927304010489976, 9)

    def test_nr_symm_uks_b3lypg_direct(self):
        method = dft.UKS(h2osym)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -76.384928891413438, 9)

    def test_nr_symm_roks_lsda(self):
        method = dft.RKS(h2osym_cation)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 9)

    def test_nr_symm_roks_b3lypg(self):
        method = dft.ROKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)

    def test_nr_symm_roks_b3lypg_direct(self):
        method = dft.ROKS(h2osym_cation)
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)

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
        self.assertAlmostEqual(lib.finger(vxc), 22.767504283729778, 8)

        method._eri = None
        method.max_memory = 0
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc), 22.767504283729778, 8)

    def test_nr_uks_vv10(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method.xc = 'wB97M_V'
        method.nlc = 'vv10'
        method.grids.prune = None
        method.grids.atom_grid = {"H": (30, 86), "O": (30, 86),}
        method.nlcgrids.prune = None
        method.nlcgrids.atom_grid = {"H": (20, 50), "O": (20, 50),}
        method.dump_flags()
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc[0]), 22.767504283729778, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 22.767504283729778, 8)

        method._eri = None
        method.max_memory = 0
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc[0]), 22.767504283729778, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 22.767504283729778, 8)

    def test_nr_rks_rsh(self):
        method = dft.RKS(h2o)
        dm = method.get_init_guess()
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc), 22.759558596896344, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.finger(vxc), 22.759558596896344, 8)

        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc), 23.058813088809824, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.finger(vxc), 23.058813088809824, 8)

    def test_nr_uks_rsh(self):
        method = dft.UKS(h2o)
        dm = method.get_init_guess()
        dm = (dm[0], dm[0])
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc[0]), 22.759558596896344, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 22.759558596896344, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'wB97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.finger(vxc[0]), 22.759558596896344, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 22.759558596896344, 8)

        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm)
        self.assertAlmostEqual(lib.finger(vxc[0]), 23.058813088809824, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 23.058813088809824, 8)

        method._eri = None
        method.max_memory = 0
        method.xc = 'B97M_V'
        vxc = method.get_veff(h2o, dm, dm, vxc)
        self.assertAlmostEqual(lib.finger(vxc[0]), 23.058813088809824, 8)
        self.assertAlmostEqual(lib.finger(vxc[1]), 23.058813088809824, 8)

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


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
