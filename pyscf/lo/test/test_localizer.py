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
from pyscf import gto, scf
from pyscf.lo import boys, edmiston, pipek

def setUpModule():
    global mol, mf, h2o, mf_h2o
    mol = gto.Mole()
    mol.atom = '''
         H    0.000000000000     2.491406946734     0.000000000000
         C    0.000000000000     1.398696930758     0.000000000000
         H    0.000000000000    -2.491406946734     0.000000000000
         C    0.000000000000    -1.398696930758     0.000000000000
         H    2.157597486829     1.245660462400     0.000000000000
         C    1.211265339156     0.699329968382     0.000000000000
         H    2.157597486829    -1.245660462400     0.000000000000
         C    1.211265339156    -0.699329968382     0.000000000000
         H   -2.157597486829     1.245660462400     0.000000000000
         C   -1.211265339156     0.699329968382     0.000000000000
         H   -2.157597486829    -1.245660462400     0.000000000000
         C   -1.211265339156    -0.699329968382     0.000000000000
      '''
    mol.basis = '6-31g'
    mol.symmetry = 0
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol).run()

    h2o = gto.Mole()
    h2o.atom = '''
         O    0.   0.       0
         h    0.   -0.757   0.587
         h    0.   0.757    0.587'''
    h2o.basis = 'unc-sto3g'
    h2o.verbose = 5
    h2o.output = '/dev/null'
    h2o.build()
    mf_h2o = scf.RHF(h2o).run()

def tearDownModule():
    global mol, mf, h2o, mf_h2o
    h2o.stdout.close()
    del mol, mf, h2o, mf_h2o

# note tests may fail due to initial guess problem

class KnownValues(unittest.TestCase):
    def test_boys(self):
        idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
        loc = boys.Boys(mol, mf.mo_coeff[:,idx])
        loc.max_cycle = 100
        mo = loc.kernel()
        dip = boys.dipole_integral(mol, mo)
        z = numpy.einsum('xii,xii->', dip, dip)
        self.assertAlmostEqual(z, 98.670988758151907, 4)

    def test_edmiston(self):
        idx = range(1, 5)
        loc = edmiston.EdmistonRuedenberg(h2o)
        mo = loc.kernel(mf_h2o.mo_coeff[:,idx])
        dip = boys.dipole_integral(h2o, mo)
        z = numpy.einsum('xii,xii->', dip, dip)
        self.assertAlmostEqual(z, 1.1566988026, 4)

    def test_pipek(self):
        idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
        # Initial guess from Boys localization. Otherwise uncertainty between
        # two solutions found in PM kernel
        mo = boys.Boys(mol, mf.mo_coeff[:,idx]).kernel()
        loc = pipek.PipekMezey(mol, mo)
        loc.max_cycle = 100
        mo = loc.kernel()
        pop = pipek.atomic_pops(mol, mo)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 12, 4)

    def test_pipek_atomic_pops(self):
        pop = pipek.atomic_pops(h2o, mf_h2o.mo_coeff[:,3:8], method='meta_lowdin')
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 2.8858772271976187, 6)

        pop = pipek.atomic_pops(h2o, mf_h2o.mo_coeff[:,3:8], method='lowdin')
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 2.8271629470491471, 6)

        pop = pipek.atomic_pops(h2o, mf_h2o.mo_coeff[:,3:8], method='mulliken')
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 2.9542028242581448, 6)

    def test_pipek_exp4(self):
        loc = pipek.PipekMezey(h2o, mf_h2o.mo_coeff[:,3:8])
        loc.exponent = 4
        loc.max_cycle = 100
        mo = loc.kernel()
        pop = pipek.atomic_pops(h2o, mo)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.5368940222128247, 4)

    def test_pipek_becke_scheme(self):
        loc = pipek.PipekMezey(h2o, mf_h2o.mo_coeff[:,3:8])
        loc.pop_method = 'becke'
        mo = loc.kernel()
        pop = pipek.atomic_pops(h2o, mo)
        z = numpy.einsum('xii,xii->', pop, pop)
        self.assertAlmostEqual(z, 3.548139685217463, 4)

    def test_1orbital(self):
        lmo = boys.Boys(mol, mf.mo_coeff[:,:1]).kernel()
        self.assertTrue(numpy.all(mf.mo_coeff[:,:1] == lmo))

        lmo = boys.Boys(mol, mf.mo_coeff[:,:0]).kernel()
        self.assertTrue(lmo.size == 0)


if __name__ == "__main__":
    print("Full Tests for localizer")
    unittest.main()
