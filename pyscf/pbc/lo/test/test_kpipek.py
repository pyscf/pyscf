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
from pyscf.lo import orth
from pyscf.pbc import gto, lo
from pyscf.lo.tools import findiff_grad, findiff_hess


class Water(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O     0.000000     0.000000     0.000000
        O     0.000000     0.000000     1.480000
        H     0.895669     0.000000    -0.316667
        H    -0.895669     0.000000     1.796667
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'sto-3g'
        cell.precision = 1e-10
        cell.build()

        kmesh = [3,2,1]
        kpts = cell.make_kpts(kmesh)

        cls.cell = cell
        cls.kpts = kpts

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.kpts

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        def test1(mlo, loss_ref):
            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                loss = mlo.cost_function()
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)

        cell = self.cell
        kpts = self.kpts

        s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
        nocc = cell.nelectron // 2
        mo_coeff = np.asarray([orth.schmidt(x)[:,:nocc] for x in s])

        mlo = lo.kpipek.KPM(cell, mo_coeff, kpts)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 8.4790187372,
            3: 8.2253251038,
            4: 7.9972601241,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 8.5306636986,
            3: 8.3036701735,
            4: 8.1015160175,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 9.2813845238,
            3: 9.4235940269,
            4: 9.5731417037,
        }
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        cell = self.cell
        kpts = self.kpts
        precision = 6

        def test1(mo_coeff, pop_method):
            mlo = lo.kpipek.KPM(cell, mo_coeff, kpts)
            mlo.pop_method = pop_method

            mlo._proj_data = mlo.get_proj_data()

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
            u0 = mlo.extract_rotation(x0)

            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
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

            mlo._proj_data = None

        s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
        mo_coeff = np.asarray([orth.schmidt(x) for x in s])

        mo_idx = [1,6,9]
        mo_coeff = [x[:,mo_idx] for x in mo_coeff]
        test1(mo_coeff, 'meta-lowdin')
        test1(mo_coeff, 'iao')
        test1(mo_coeff, 'iao-biorth')


class WaterReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O     0.000000     0.000000     0.000000
        O     0.000000     0.000000     1.480000
        H     0.895669     0.000000    -0.316667
        H    -0.895669     0.000000     1.796667
        '''
        cell.a = np.eye(3) * 5
        cell.basis = 'sto-3g'
        cell.precision = 1e-10
        cell.build()

        kmesh = [3,2,1]
        kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)

        cls.cell = cell
        cls.kpts = kpts

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.kpts

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        def test1(mlo, loss_ref):
            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                loss = mlo.cost_function()
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)

        cell = self.cell
        kpts = self.kpts

        s = cell.pbc_intor('int1e_ovlp', kpts=kpts.kpts_ibz)
        nocc = cell.nelectron // 2
        mo_coeff = np.asarray([orth.schmidt(x)[:,:nocc] for x in s])

        mlo = lo.kpipek.KPMReal(cell, mo_coeff, kpts)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 8.4790187372,
            3: 8.2253251038,
            4: 7.9972601241,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 8.5306636986,
            3: 8.3036701735,
            4: 8.1015160175,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 9.2813845238,
            3: 9.4235940269,
            4: 9.5731417037,
        }
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        cell = self.cell
        kpts = self.kpts
        precision = 6

        def test1(mo_coeff, pop_method):
            mlo = lo.kpipek.KPMReal(cell, mo_coeff, kpts)
            mlo.pop_method = pop_method

            mlo._proj_data = mlo.get_proj_data()

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
            u0 = mlo.extract_rotation(x0)

            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
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
                # The Hessian diagonal is not exact by design
                # self.assertAlmostEqual(abs(np.diagonal(h)-hdiag).max(), 0, precision)

            mlo._proj_data = None

        s = cell.pbc_intor('int1e_ovlp', kpts=kpts.kpts_ibz)
        mo_coeff = np.asarray([orth.schmidt(x) for x in s])

        mo_idx = [1,6,9]
        mo_coeff = [x[:,mo_idx] for x in mo_coeff]
        test1(mo_coeff, 'meta-lowdin')
        test1(mo_coeff, 'iao')
        test1(mo_coeff, 'iao-biorth')




if __name__ == "__main__":
    print("Full Tests for KptsPipekMezey")
    unittest.main()
