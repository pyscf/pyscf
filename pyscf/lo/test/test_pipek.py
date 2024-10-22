#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf import gto, lo


class Water(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 4
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.000000     0.000000     0.000000
        O     0.000000     0.000000     1.480000
        H     0.895669     0.000000    -0.316667
        H    -0.895669     0.000000     1.796667
        '''
        mol.basis = 'sto-3g'
        mol.precision = 1e-10
        mol.build()

        cls.mol = mol

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        mol = self.mol
        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)
        norb = mo_coeff.shape[1]
        u = np.eye(norb)

        loss_ref = {
            2: 11.0347269392,
            3: 10.5595998735,
            4: 10.1484720537
        }

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)
        for exponent in [2,3,4]:
            mlo.set(exponent=exponent)
            loss = mlo.cost_function(u)
            self.assertAlmostEqual(loss, loss_ref[exponent], 6)

    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        mol = self.mol
        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)

        mo_idx = [1,6,9]
        mo_coeff = mo_coeff[:,mo_idx]
        norb = mo_coeff.shape[1]
        u0 = np.eye(norb)

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)
        x0 = mlo.pack_uniq_var(np.zeros((norb,norb)))

        step_length = 1e-3
        precision = 4

        for exponent in [2,3,4]:
            mlo.set(exponent=exponent)
            g = mlo.get_grad(u0)
            g1, h_op, h_diag = mlo.gen_g_hop(u0)

            self.assertAlmostEqual(abs(g-g1).max(), 0, 6)

            H = np.zeros((norb,norb))
            for i in range(norb):
                xi = np.zeros(norb)
                xi[i] = 1
                H[:,i] = h_op(xi)

            self.assertAlmostEqual(abs(np.diagonal(H)-h_diag).max(), 0, 6)

            def func(x):
                u = mlo.extract_rotation(x)
                return -mlo.cost_function(u)

            num_g = _num_grad(func, x0, step_length)
            num_H = _num_hess(func, x0, step_length)

            self.assertAlmostEqual(abs(g-num_g).max(), 0, precision)
            self.assertAlmostEqual(abs(H-num_H).max(), 0, precision)


def _num_grad(func, x0, step_length):
    x0 = np.asarray(x0)
    n = x0.size
    g = np.zeros_like(x0)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = step_length
        yf = func(x0+dx)
        yb = func(x0-dx)
        g[i] = (yf-yb) / (2.*step_length)
    return g
def _num_hess(func, x0, step_length):
    x0 = np.asarray(x0)
    n = x0.size
    H = np.zeros((n,n))
    y0 = func(x0)
    for i in range(n):
        dxi = np.zeros(n)
        dxi[i] = step_length
        for j in range(i+1,n):
            dxj = np.zeros(n)
            dxj[j] = step_length
            yff = func(x0+dxi+dxj)
            yfb = func(x0+dxi-dxj)
            ybf = func(x0-dxi+dxj)
            ybb = func(x0-dxi-dxj)
            H[i,j] = H[j,i] = (yff+ybb-yfb-ybf) / (4.*step_length**2.)
        yf = func(x0+dxi)
        yb = func(x0-dxi)
        H[i,i] = (yf+yb-2*y0) / step_length**2.

    return H


if __name__ == "__main__":
    print("Full Tests for PipekMezey")
    unittest.main()
