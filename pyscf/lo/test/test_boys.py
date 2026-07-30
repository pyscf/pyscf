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
from pyscf.pbc import gto as pgto
from pyscf.lo.tools import findiff_grad, findiff_hess


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
        mol.build()

        cls.mol = mol

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()
        del cls.mol

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        def test1(mlo, loss_ref):
            loss = mlo.cost_function() # test full projector
            self.assertAlmostEqual(loss, loss_ref, 6)

        mol = self.mol

        # real orbitals
        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)
        norb = mo_coeff.shape[1]

        mlo = lo.boys.Boys(mol, mo_coeff)

        loss_ref = 16.4033477177
        test1(mlo, loss_ref)


        # complex orbitals
        mo_coeff = mo_coeff + np.cos(mo_coeff)*0.01j

        mlo = lo.boys.Boys(mol, mo_coeff)

        loss_ref = 16.4823032074
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        mol = self.mol
        precision = 6

        def test1(mo_coeff):
            norb = mo_coeff.shape[1]
            u0 = np.eye(norb)

            mlo = lo.boys.Boys(mol, mo_coeff)

            def func(x):
                u = mlo.extract_rotation(x)
                f = mlo.cost_function(u)
                if mlo.maximize:
                    return -f
                else:
                    return f

            def fgrad(x):
                u = mlo.extract_rotation(x)
                return mlo.get_grad(u)

            x0 = mlo.zero_uniq_var()

            g = mlo.get_grad(u0)
            g1, h_op, hdiag = mlo.gen_g_hop(u0)

            self.assertAlmostEqual(abs(g-g1).max(), 0, 6)

            h = np.zeros((x0.size,x0.size))
            for i in range(x0.size):
                x0[i] = 1
                h[:,i] = h_op(x0)
                x0[i] = 0

            num_g = findiff_grad(func, x0)
            num_h = findiff_hess(fgrad, x0)

            self.assertAlmostEqual(abs(g-num_g).max(), 0, precision)
            self.assertAlmostEqual(abs(h-num_h).max(), 0, precision)
            self.assertAlmostEqual(abs(np.diagonal(h)-hdiag).max(), 0, precision)

        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)

        mo_idx = [1,6,9]
        mo_coeff = mo_coeff[:,mo_idx]
        test1(mo_coeff) # real orbitals
        test1(mo_coeff + np.cos(mo_coeff)*0.01j)    # complex orbitals


if __name__ == "__main__":
    print("Full Tests for Boys")
    unittest.main()
