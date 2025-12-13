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
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import mp
from pyscf import grad
from pyscf.grad import mp2 as mp2_grad

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-8
    mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_mp2_grad(self):
        pt = mp.mp2.MP2(mf)
        pt.kernel()
        g1 = pt.nuc_grad_method().kernel(pt.t2, atmlst=[0,1,2])
# O    -0.0000000000    -0.0000000000     0.0089211366
# H     0.0000000000     0.0222745046    -0.0044605683
# H     0.0000000000    -0.0222745046    -0.0044605683
        self.assertAlmostEqual(lib.fp(g1), -0.035681131697586257, 6)

        geom1 = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.55)],
            [1 , (0. , 0.757  , 0.54)]]
        mol1 = gto.M(atom=geom1, basis='631g')
        pt1 = mol1.MP2().Gradients()
        de_ref = pt1.kernel()
        e, de = pt.Gradients().as_scanner()(geom1)
        self.assertAlmostEqual(pt1.base.e_tot, e, 7)
        self.assertAlmostEqual(abs(de - de_ref).max(), 0, 5)

    def test_mp2_grad_finite_diff(self):
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.706',
            basis = '631g',
            unit='Bohr')
        mp_scanner = scf.RHF(mol).set(conv_tol=1e-14).apply(mp.MP2).as_scanner()
        e0 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.704',
            basis = '631g',
            unit='Bohr')
        e1 = mp_scanner(mol)
        mol = gto.M(
            verbose = 0,
            atom = 'H 0 0 0; H 0 0 1.705',
            basis = '631g',
            unit='Bohr')
        mp_scanner(mol)
        g_scan = mp_scanner.nuc_grad_method().as_scanner()
        g1 = g_scan(mol.atom)[1]
        self.assertAlmostEqual(g1[0,2], (e1-e0)*500, 6)

    def test_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        pt.max_memory = 1
        pt.kernel()
        g1 = mp2_grad.Gradients(pt).kernel(pt.t2)
# O    -0.0000000000    -0.0000000000     0.0037319667
# H    -0.0000000000    -0.0897959298    -0.0018659834
# H     0.0000000000     0.0897959298    -0.0018659834
        self.assertAlmostEqual(lib.fp(g1), 0.12457973399092415, 6)

    def test_as_scanner_with_frozen(self):
        pt = mp.mp2.MP2(mf)
        pt.frozen = [0,1,10,11,12]
        gscan = pt.nuc_grad_method().as_scanner().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -76.025166662910223, 9)
        self.assertAlmostEqual(lib.fp(g1), 0.12457973399092415, 6)

        pt = mf.MP2()
        pt.frozen = [0, 1]
        gscan = pt.nuc_grad_method().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -76.07095754926583, 9)
        self.assertAlmostEqual(lib.fp(g1), -0.028399476189179818, 6)

        pt.frozen = 2
        gscan = pt.nuc_grad_method().as_scanner()
        e, g1 = gscan(mol)
        self.assertTrue(gscan.converged)
        self.assertAlmostEqual(e, -76.07095754926583, 9)
        self.assertAlmostEqual(lib.fp(g1), -0.028399476189179818, 6)

    def test_with_x2c_scanner(self):
        with lib.light_speed(20.):
            pt = mp.mp2.MP2(mf.x2c())
            pt.frozen = [0,1,10,11,12]
            gscan = pt.nuc_grad_method().as_scanner().as_scanner()
            e, g1 = gscan(mol)

            ps = pt.as_scanner()
            e1 = ps([[8 , (0. , 0.     , 0.)],
                     [1 , (0. , -0.757 , 0.5871)],
                     [1 , (0. ,  0.757 , 0.587)]])
            e2 = ps([[8 , (0. , 0.     , 0.)],
                     [1 , (0. , -0.757 , 0.5869)],
                     [1 , (0. ,  0.757 , 0.587)]])
            self.assertAlmostEqual(g1[1,2], (e1-e2)/0.0002*lib.param.BOHR, 5)

    def test_with_qmmm_scanner(self):
        from pyscf import qmmm
        mol = gto.Mole()
        mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                       H                 -0.00000000   -0.84695236    0.59109389
                       H                 -0.00000000    0.89830571    0.52404783 '''
        mol.verbose = 0
        mol.basis = '6-31g'
        mol.build()

        coords = [(0.5,0.6,0.1)]
        #coords = [(0.0,0.0,0.0)]
        charges = [-0.1]
        mf = qmmm.add_mm_charges(mol.RHF(), coords, charges)
        ps = mf.MP2().as_scanner()
        g = ps.nuc_grad_method().as_scanner()(mol)[1]
        e1 = ps(''' O                  0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        e2 = ps(''' O                 -0.00100000    0.00000000   -0.11081188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
        ref = (e1 - e2)/0.002 * lib.param.BOHR
        self.assertAlmostEqual(g[0,0], ref, 4)

    def test_symmetrize(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='631g', symmetry=True)
        g = mol.RHF.run().MP2().run().Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), 0.049987975650731625, 6)

    # issue 1985
    def test_cart_gto(self):
        mol1 = mol.copy()
        mol1.cart = True
        mol1.basis = '6-31g*'
        g = mol.RHF.run().MP2().run().Gradients().kernel()
        self.assertAlmostEqual(lib.fp(g), -0.03568120792884476, 6)


if __name__ == "__main__":
    print("Tests for MP2 gradients")
    unittest.main()
