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
from pyscf import grad, hessian

import sys
try:
    import dftd3
except ImportError:
    pass

try:
    import dftd4
except ImportError:
    pass

def setUpModule():
    global mol, h4
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()

    h4 = gto.Mole()
    h4.verbose = 0
    h4.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)]]
    h4.basis = '631g'
    h4.unit = 'B'
    h4.build()

def tearDownModule():
    global mol, h4
    mol.stdout.close()
    del mol, h4

def finite_diff(mf):
    mol = mf.mol
    mfs = mf.Gradients().set(grid_response=True).as_scanner()
    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mfs(mol.copy())[1]
            mol._env[ptr+i] = coord[i] - inc
            e1b = mfs(mol.copy())[1]
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    natm = mol.natm
    e2ref = [grad_full(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(natm,3,natm,3).transpose(0,2,1,3)
    return e2ref

def finite_partial_diff(mf):
    # \partial^2 E / \partial R \partial R'
    mol = mf.mol
    def grad_partial_R(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    natm = mol.natm
    e2ref = [grad_partial_R(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(natm,3,natm,3).transpose(0,2,1,3)
    return e2ref

class KnownValues(unittest.TestCase):
    def test_finite_diff_lda_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7828771346902333, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_finite_diff_b3lyp_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp5'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7590878171493624, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    @unittest.skipIf('dftd3' not in sys.modules, "requires the dftd3 library")
    def test_finite_diff_b3lyp_d3_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp'
        mf.disp = 'd3bj'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7586078053657133, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    @unittest.skipIf('dftd4' not in sys.modules, "requires the dftd4 library")
    def test_finite_diff_b3lyp_d4_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp'
        mf.disp = 'd4'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7588415571313422, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_finite_diff_wb97x_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'wb97x'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7637876979690904, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 2)

    def test_finite_diff_m06l_hess_high_cost(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'm06l'
        # Note MGGA hessian is sensitive to grids level
        mf.grids.level = 4
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7671113269563008, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 2)

    def test_finite_diff_lda_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5586931702431067, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 3)

    def test_finite_diff_b3lyp_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'b3lyp5'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5755856712112237, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 4)

    def test_finite_diff_m06l_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'm06l'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5479869865170132, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 3)

    def test_finite_diff_lda_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)

    def test_finite_diff_b3lyp_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)

    def test_finite_diff_m06l_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'm06l'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)


if __name__ == "__main__":
    print("Full Tests for RKS Hessian")
    unittest.main()
