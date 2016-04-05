#!/usr/bin/env python

import unittest
from pyscf import gto
from pyscf import lib
from pyscf import dft

h2o = gto.Mole()
h2o.verbose = 0
h2o.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

h2o.basis = {"H": '6-31g', "O": '6-31g',}
h2o.build()

h2osym = gto.Mole()
h2osym.verbose = 0
h2osym.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

h2osym.basis = {"H": '6-31g', "O": '6-31g',}
h2osym.symmetry = 1
h2osym.build()


class KnowValues(unittest.TestCase):
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

    def test_nr_uks_lsda(self):
        mol1 = h2o.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.UKS(mol1)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 9)

    def test_nr_uks_b3lypg(self):
        mol1 = h2o.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.UKS(mol1)
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

    def test_nr_roks_lsda(self):
        mol1 = h2o.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 9)

    def test_nr_roks_b3lypg(self):
        mol1 = h2o.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)

    def test_nr_roks_b3lypg_direct(self):
        mol1 = h2o.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
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
        method = dft.density_fit(dft.RKS(h2osym))
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
        mol1 = h2osym.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.UKS(mol1)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350995324984709, 9)

    def test_nr_symm_uks_b3lypg(self):
        mol1 = h2osym.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.UKS(mol1)
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
        mol1 = h2osym.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.350333965173704, 9)

    def test_nr_symm_roks_b3lypg(self):
        mol1 = h2osym.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
        method.xc = 'b3lypg'
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)

    def test_nr_symm_roks_b3lypg_direct(self):
        mol1 = h2osym.copy()
        mol1.charge = 1
        mol1.spin = 1
        mol1.build(0, 0)
        method = dft.ROKS(mol1)
        method.xc = 'b3lypg'
        method.max_memory = 0
        method.direct_scf = True
        method.grids.prune = dft.gen_grid.treutler_prune
        method.grids.atom_grid = {"H": (50, 194), "O": (50, 194),}
        self.assertAlmostEqual(method.scf(), -75.926526046608529, 9)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
