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
            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                loss = mlo.cost_function(mode=None) # test full projector
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)
                loss = mlo.cost_function(mode='pop')
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)

        mol = self.mol

        # real orbitals
        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)
        norb = mo_coeff.shape[1]

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0347269392,
            3: 10.5595998735,
            4: 10.1484720537,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1065896746,
            3: 10.6685385701,
            4: 10.2886276646,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.4952121786,
            3: 12.7449891556,
            4: 13.0116401223,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'becke'
        loss_ref = {
            2: 9.4230597975,
            3: 8.2191241100,
            4: 7.3049902834,
        }
        test1(mlo, loss_ref)


        # complex orbitals
        mo_coeff = mo_coeff + np.cos(mo_coeff)*0.01j

        mlo = lo.pipek.PipekMezey(mol, mo_coeff)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0457506419,
            3: 10.5751208425,
            4: 10.1685286213,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1150294738,
            3: 10.6804270421,
            4: 10.3041148130,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.5189728118,
            3: 12.7814008434,
            4: 13.0613620163,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'becke'
        loss_ref = {
            2: 9.4333937604,
            3: 8.2317523225,
            4: 7.3198256336,
        }
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        mol = self.mol
        precision = 6

        def test1(mo_coeff):
            norb = mo_coeff.shape[1]
            u0 = np.eye(norb)

            mlo = lo.pipek.PipekMezey(mol, mo_coeff)
            mlo.pop_method = 'meta-lowdin'

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

        s = mol.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)

        mo_idx = [1,6,9]
        mo_coeff = mo_coeff[:,mo_idx]
        test1(mo_coeff) # real orbitals
        test1(mo_coeff + np.cos(mo_coeff)*0.01j)    # complex orbitals


class WaterComplex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = pgto.Cell()
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
        kpt = np.asarray([0.35, 0.77, 0.128])  # twisted

        cls.cell = cell
        cls.kpt = kpt

    @classmethod
    def tearDownClass(cls):
        cls.cell.stdout.close()
        del cls.cell, cls.kpt

    def test_cost(self):
        ''' Test for `cost_function`
        '''
        def test1(mlo, loss_ref):
            for exponent in [2,3,4]:
                mlo.set(exponent=exponent)
                loss = mlo.cost_function(mode=None) # test full projector
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)
                loss = mlo.cost_function(mode='pop')
                self.assertAlmostEqual(loss, loss_ref[exponent], 6)

        cell = self.cell
        kpt = self.kpt

        # real orbitals
        s = cell.pbc_intor('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)
        norb = mo_coeff.shape[1]

        mlo = lo.pipek.PipekMezeyComplex(cell, mo_coeff, kpt=kpt)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0315537212,
            3: 10.5554040521,
            4: 10.1434751701,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1017473605,
            3: 10.6620904324,
            4: 10.2808857565,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.4982814778,
            3: 12.7491741886,
            4: 13.0175149029,
        }
        test1(mlo, loss_ref)


        # complex orbitals
        mo_coeff = mo_coeff + np.cos(mo_coeff)*0.01j

        mlo = lo.pipek.PipekMezeyComplex(cell, mo_coeff, kpt=kpt)

        mlo.pop_method = 'meta-lowdin'
        loss_ref = {
            2: 11.0425593001,
            3: 10.5708660632,
            4: 10.1634296607,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao'
        loss_ref = {
            2: 11.1105783801,
            3: 10.6745426154,
            4: 10.2970568735,
        }
        test1(mlo, loss_ref)

        mlo.pop_method = 'iao-biorth'
        loss_ref = {
            2: 12.5233674757,
            3: 12.7875866137,
            4: 13.0699884808,
        }
        test1(mlo, loss_ref)


    def test_grad_hess(self):
        ''' Test for `get_grad` and `gen_g_hop`
        '''
        cell = self.cell
        kpt = self.kpt
        precision = 6

        def test1(mo_coeff):
            norb = mo_coeff.shape[1]
            u0 = np.eye(norb)

            mlo = lo.pipek.PipekMezeyComplex(cell, mo_coeff, kpt=kpt)
            mlo.pop_method = 'meta-lowdin'

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

        s = cell.intor_symmetric('int1e_ovlp')
        mo_coeff = lo.orth.schmidt(s)

        mo_idx = [1,6,9]
        mo_coeff = mo_coeff[:,mo_idx]
        test1(mo_coeff) # real orbitals
        test1(mo_coeff + np.cos(mo_coeff)*0.01j)    # complex orbitals


if __name__ == "__main__":
    print("Full Tests for PipekMezey")
    unittest.main()
