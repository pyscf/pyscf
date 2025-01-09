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
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import radi
from pyscf.grad import uks
try:
    from pyscf.dispersion import dftd3, dftd4
except ImportError:
    dftd3 = dftd4 = None


def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf = dft.UKS(mol)
    mf.conv_tol = 1e-14

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
        mf.kernel()

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_finite_diff_uks_grad(self):
#[[-5.23195019e-16 -5.70291415e-16  5.32918387e-02]
# [ 1.33417513e-16  6.75277008e-02 -2.66519852e-02]
# [ 1.72274651e-16 -6.75277008e-02 -2.66519852e-02]]
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.12090786243525126, 5)

#[[-2.95956939e-16 -4.22275612e-16  5.32998759e-02]
# [ 1.34532051e-16  6.75279140e-02 -2.66499379e-02]
# [ 1.68146089e-16 -6.75279140e-02 -2.66499379e-02]]
        g = mf.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.12091122429043633, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_df_uks_grad(self):
        mf1 = mf.density_fit ().run ()
        g = mf1.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.12093220501429122, 5)

        mol1 = mol.copy()
        mf_scanner = mf1.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_uks_d3_grad(self):
        mol1 = mol.copy()
        mf = dft.UKS(mol, xc='b3lyp')
        mf.disp = 'd3bj'
        mf.conv_tol = 1e-14
        mf.kernel()
        g = mf.nuc_grad_method().set(grid_response=True).kernel()

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_uks_d4_grad(self):
        mol1 = mol.copy()
        mf = dft.UKS(mol, xc='b3lyp')
        mf.disp = 'd4'
        mf.conv_tol = 1e-14
        mf.kernel()
        g = mf.nuc_grad_method().set(grid_response=True).kernel()

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_uks_grad_lda(self):
        mol = gto.Mole()
        mol.atom = [
            ['H' , (0. , 0. , 1.804)],
            ['F' , (0. , 0. , 0.   )], ]
        mol.unit = 'B'
        mol.basis = '631g'
        mol.charge = -1
        mol.spin = 1
        mol.build()

# sum over z direction non-zero, if without grid response
# H    -0.0000000000     0.0000000000    -0.1481125370
# F    -0.0000000000     0.0000000000     0.1481164667
        mf = dft.UKS(mol).run(conv_tol=1e-12)
        self.assertAlmostEqual(lib.fp(mf.Gradients().kernel()),
                               0.10365160440876001, 5)
        mf.grids.prune = None
        mf.grids.level = 6
        mf.run(conv_tol=1e-12)
# H     0.0000000000     0.0000000000    -0.1481124925
# F    -0.0000000000     0.0000000000     0.1481122913
        self.assertAlmostEqual(lib.fp(mf.Gradients().kernel()),
                               0.10365040148752827, 5)

    def test_finite_diff_uks_grad_gga(self):
#[[ 6.47874920e-16 -2.75292214e-16  3.97215970e-02]
# [-6.60278148e-17  5.87909340e-02 -1.98650384e-02]
# [ 6.75500259e-18 -5.87909340e-02 -1.98650384e-02]]
        mf = mol.UKS().run(xc='b3lypg', conv_tol=1e-12)
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.10202554999695367, 5)

#[[ 2.58483362e-16  5.82369026e-16  5.17616036e-02]
# [-5.46977470e-17  6.39273304e-02 -2.58849008e-02]
# [ 5.58302713e-17 -6.39273304e-02 -2.58849008e-02]]
        mf = mol.UKS().run(xc='b88,p86', conv_tol=1e-12)
        g = mf.Gradients().set().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.11509739136150157, 5)
        g = mf.Gradients().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.11507986316077731, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_uks_grad_nlc(self):
#[[ 3.19690405e-16 -9.39540337e-16  5.09520937e-02]
# [-5.47247224e-17  6.32050882e-02 -2.54793086e-02]
# [ 6.75779981e-17 -6.32050882e-02 -2.54793086e-02]]
        mf = mol.UKS()
        mf.set(xc='VV10', conv_tol=1e-12, nlc='VV10')
        mf.nlcgrids.level = 1
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.11368788988328639, 5)

        mf.nlcgrids.level = 0
        mf.kernel()
        g = mf.Gradients().set(grid_response=True).kernel()

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_uks_grad_mgga(self):
        mf = mol.UKS().run(xc='m06l', conv_tol=1e-12)
        g = mf.nuc_grad_method().set(grid_response=True).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.0980126030724174, 5)

        mol1 = mol.copy()
        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol1.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol1.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_different_grids_for_grad(self):
        grids1 = dft.gen_grid.Grids(mol)
        grids1.level = 1
        g = mf.nuc_grad_method().set(grids=grids1).kernel()
        self.assertAlmostEqual(lib.fp(g), -0.12085837432386037, 5)

    def test_get_vxc(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol.basis = '631g'
        mol.charge = -1
        mol.spin = 1
        mol.build()
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-12
        e0 = mf.scf()
        g = uks.Gradients(mf)
        g.grid_response = True
        g0 = g.kernel()
        dm0 = mf.make_rdm1()

        denom = 1/.00001 * lib.param.BOHR
        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = [
            ['O' , (0. , 0.     , 0.00001)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. ,  0.757 , 0.587)] ]
        mol1.basis = '631g'
        mol1.charge = -1
        mol1.spin = 1
        mol1.build()
        mf1 = dft.UKS(mol1)
        mf1.xc = 'b3lyp'
        mf1.conv_tol = 1e-12
        e1 = mf1.scf()
        self.assertAlmostEqual((e1-e0)*denom, g0[0,2], 3)

        grids0 = dft.gen_grid.Grids(mol)
        grids0.atom_grid = (20,86)
        grids0.build(with_non0tab=False)
        grids1 = dft.gen_grid.Grids(mol1)
        grids1.atom_grid = (20,86)
        grids1.build(with_non0tab=False)

        xc = 'lda,'
        exc0 = dft.numint.nr_uks(mf._numint, mol, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_uks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_uks(mf._numint, mol, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_uks(mf._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = uks.get_vxc(mf._numint, mol, grids0, xc, dm0)[1]
        ev1, vxc1 = uks.get_vxc_full_response(mf._numint, mol, grids0, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1], dm0[:,p0:p1])*2
        exc1_full = numpy.einsum('sxij,sij->x', vxc1[:,:,p0:p1], dm0[:,p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 3)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 5)

        xc = 'b88,'
        exc0 = dft.numint.nr_uks(mf._numint, mol, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_uks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_uks(mf._numint, mol, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_uks(mf._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = uks.get_vxc(mf._numint, mol, grids0, xc, dm0)[1]
        ev1, vxc1 = uks.get_vxc_full_response(mf._numint, mol, grids0, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1], dm0[:,p0:p1])*2
        exc1_full = numpy.einsum('sxij,sij->x', vxc1[:,:,p0:p1], dm0[:,p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 2)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 5)

        xc = 'm06l,'
        exc0 = dft.numint.nr_uks(mf._numint, mol, grids0, xc, dm0)[1]
        exc1 = dft.numint.nr_uks(mf1._numint, mol1, grids1, xc, dm0)[1]

        grids0_w = grids0.copy()
        grids0_w.weights = grids1.weights
        grids0_c = grids0.copy()
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_uks(mf._numint, mol, grids0_w, xc, dm0)[1]
        exc0_c = dft.numint.nr_uks(mf._numint, mol1, grids0_c, xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = uks.get_vxc(mf._numint, mol, grids0, xc, dm0)[1]
        ev1, vxc1 = uks.get_vxc_full_response(mf._numint, mol, grids0, xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1], dm0[:,p0:p1])*2
        exc1_full = numpy.einsum('sxij,sij->x', vxc1[:,:,p0:p1], dm0[:,p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 1)
        self.assertAlmostEqual(dexc_t, exc1_full[2], 5)

    def test_range_separated(self):
        mol = gto.M(atom="H; H 1 1.", basis='ccpvdz', verbose=0)
        mf = dft.UKS(mol)
        mf.xc = 'wb97x'
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        smf = mf.as_scanner()
        mol1 = gto.M(atom="H; H 1 1.001", basis='ccpvdz')
        mol2 = gto.M(atom="H; H 1 0.999", basis='ccpvdz')
        dx = (mol1.atom_coord(1) - mol2.atom_coord(1))[0]
        e1 = smf(mol1)
        e2 = smf(mol2)
        self.assertAlmostEqual((e1-e2)/dx, g[1,0], 5)


if __name__ == "__main__":
    print("Full Tests for UKS Gradients")
    unittest.main()
