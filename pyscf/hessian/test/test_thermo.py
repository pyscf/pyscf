#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto, lib
from pyscf.hessian import thermo

class KnownValues(unittest.TestCase):
    def test_TR(self):
        # linear molecule
        m = numpy.asarray([1., 8., 3.])
        c = numpy.asarray([[1., 2., 3.], [2., 3., 4.,], [3., 4., 5.]])
        TR = thermo._get_TR(m, c)
        self.assertTrue(abs(TR[3]).max() > .1)
        self.assertTrue(abs(TR[4]).max() > .1)
        self.assertAlmostEqual(abs(TR[5]).max(), 0, 9)

        # atom
        m = numpy.asarray([3.])
        c = numpy.asarray([[1., 2., 3.]])
        TR = thermo._get_TR(m, c)
        self.assertAlmostEqual(abs(TR[3]).max(), 0, 9)
        self.assertAlmostEqual(abs(TR[4]).max(), 0, 9)
        self.assertAlmostEqual(abs(TR[5]).max(), 0, 9)

        # otherwise
        m = numpy.asarray([3., 2., 1.])
        c = numpy.asarray([[0., 2., 0.], [2., 1., 4.,], [1., 4., 5.]])
        TR = thermo._get_TR(m, c)
        self.assertTrue(abs(TR[3]).max() > .1)
        self.assertTrue(abs(TR[4]).max() > .1)
        self.assertTrue(abs(TR[5]).max() > .1)

    def test_rotation_const(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587')
        mass = mol.atom_mass_list(isotope_avg=True)
        r = mol.atom_coords() - numpy.random.random((1,3))
        c = thermo.rotation_const(mass, r, 'GHz')
        self.assertAlmostEqual(c[0], 819.20368462, 6)
        self.assertAlmostEqual(c[1], 437.4565388 , 6)
        self.assertAlmostEqual(c[2], 285.17335217, 6)

        c = thermo.rotation_const(mass[1:], r[1:], 'GHz')
        self.assertTrue(c[0] == numpy.inf)
        self.assertAlmostEqual(c[1], 437.4565388 , 6)
        self.assertAlmostEqual(c[2], 437.4565388 , 6)

        c = thermo.rotation_const(mass[2:], r[2:], 'GHz')
        self.assertTrue(c[0] == numpy.inf)
        self.assertTrue(c[1] == numpy.inf)
        self.assertTrue(c[2] == numpy.inf)

    def test_thermo(self):
        mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587')
        mf = mol.HF.run(conv_tol=1e-11)
        hess = mf.Hessian().kernel()
        results = thermo.harmonic_analysis(mol, hess)
        self.assertAlmostEqual(lib.fp(abs(results['norm_mode'])), 0.1144453655, 6)
        self.assertAlmostEqual(results['freq_wavenumber'][0], 2044.5497, 3)
        self.assertAlmostEqual(results['freq_wavenumber'][1], 4486.6572, 3)
        self.assertAlmostEqual(results['freq_wavenumber'][2], 4788.2700, 3)
        thermo.dump_normal_mode(mol, results)

        results = thermo.thermo(mf, results['freq_au'], 298.15, 101325)
        thermo.dump_thermo(mol, results)
        self.assertAlmostEqual(results['E_0K'][0], -74.93727546, 7)

if __name__ == "__main__":
    print("Full Tests for RHF Hessian")
    unittest.main()
