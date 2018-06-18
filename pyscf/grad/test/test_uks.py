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
import copy
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import radi
from pyscf.grad import uks


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
mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_finite_diff_uks_grad(self):
        g = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(lib.finger(g), -0.12090786416814017, 6)

        mf_scanner = mf.as_scanner()
        e1 = mf_scanner(mol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(mol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 4)

    def test_different_grids_for_grad(self):
        grids1 = dft.gen_grid.Grids(mol)
        grids1.level = 1
        g = mf.nuc_grad_method().set(grids=grids1).kernel()
        self.assertAlmostEqual(lib.finger(g), -0.12085837432386037, 6)

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
        exc0 = dft.numint.nr_uks(mf._numint, mol, grids0, mf.xc, dm0)[1]
        exc1 = dft.numint.nr_uks(mf1._numint, mol1, grids1, mf1.xc, dm0)[1]

        grids0_w = copy.copy(grids0)
        grids0_w.weights = grids1.weights
        grids0_c = copy.copy(grids0)
        grids0_c.coords = grids1.coords
        exc0_w = dft.numint.nr_uks(mf._numint, mol, grids0_w, mf.xc, dm0)[1]
        exc0_c = dft.numint.nr_uks(mf._numint, mol1, grids0_c, mf.xc, dm0)[1]

        dexc_t = (exc1 - exc0) * denom
        dexc_c = (exc0_c - exc0) * denom
        dexc_w = (exc0_w - exc0) * denom
        self.assertAlmostEqual(dexc_t, dexc_c+dexc_w, 4)

        vxc = uks.get_vxc(mf._numint, mol, grids0, mf.xc, dm0)[1]
        ev1, vxc1 = uks.get_vxc_full_response(mf._numint, mol, grids0, mf.xc, dm0)
        p0, p1 = mol.aoslice_by_atom()[0][2:]
        exc1_approx = numpy.einsum('sxij,sij->x', vxc[:,:,p0:p1], dm0[:,p0:p1])*2
        exc1_full = numpy.einsum('sxij,sij->x', vxc1[:,:,p0:p1], dm0[:,p0:p1])*2 + ev1[0]
        self.assertAlmostEqual(dexc_t, exc1_approx[2], 3)
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
