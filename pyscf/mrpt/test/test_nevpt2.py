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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf.mrpt import nevpt2

mol = gto.Mole()
mol.verbose = 5
mol.output = '/dev/null' #None
mol.atom = [
    ['H', ( 0., 0.    , 0.    )],
    ['H', ( 0., 0.    , 0.8   )],
    ['H', ( 0., 0.    , 2.    )],
    ['H', ( 0., 0.    , 2.8   )],
    ['H', ( 0., 0.    , 4.    )],
    ['H', ( 0., 0.    , 4.8   )],
    ['H', ( 0., 0.    , 6.    )],
    ['H', ( 0., 0.    , 6.8   )],
    ['H', ( 0., 0.    , 8.    )],
    ['H', ( 0., 0.    , 8.8   )],
    ['H', ( 0., 0.    , 10.    )],
    ['H', ( 0., 0.    , 10.8   )],
    ['H', ( 0., 0.    , 12     )],
    ['H', ( 0., 0.    , 12.8   )],
]
mol.basis = 'sto3g'
mol.build()
mf = scf.RHF(mol)
mf.conv_tol = 1e-16
mf.kernel()
norb = 6
nelec = 8
mc = mcscf.CASCI(mf, norb, nelec)
mc.fcisolver.conv_tol = 1e-15
mc.kernel()
mc.canonicalize_()
mo_cas = mf.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
h1e = mc.h1e_for_cas()[0]
h2e = ao2mo.incore.full(mf._eri, mo_cas)
h2e = ao2mo.restore(1, h2e, norb).transpose(0,2,1,3)
dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
                                         mc.ci, mc.ci, norb, nelec)
hdm1 = 2.0*numpy.eye(dm1.shape[0])-dm1.T
eris = nevpt2._ERIS(mc, mc.mo_coeff)
dms = {'1': dm1, '2': dm2, '3': dm3, '4': dm4}

class KnowValues(unittest.TestCase):
    def test_Sr(self):
        norm, e = nevpt2.Sr(mc, mc.ci, dms, eris)
        self.assertAlmostEqual(e, -0.020245617857870119, 7)
        self.assertAlmostEqual(norm, 0.039479583324952064, 7)

    def test_Si(self):
        norm, e = nevpt2.Si(mc, mc.ci, dms, eris)
        self.assertAlmostEqual(e, -0.0021281408063186956, 7)
        self.assertAlmostEqual(norm, 0.0037402334190064367, 7)

    def test_Sijrs(self):
        norm, e = nevpt2.Sijrs(mc, eris)
        self.assertAlmostEqual(e, -0.0071504286486605891, 7)
        self.assertAlmostEqual(norm, 0.023107592349719219, 7)

    def test_Sijr(self):
        norm, e = nevpt2.Sijr(mc,dms, eris)
        self.assertAlmostEqual(e, -0.0050340133565470449, 7)
        self.assertAlmostEqual(norm, 0.012664066951786257, 7)

    def test_Srsi(self):
        norm, e = nevpt2.Srsi(mc,dms, eris)
        self.assertAlmostEqual(e, -0.013695728508982102, 7)
        self.assertAlmostEqual(norm, 0.040695892654346914, 7)

    def test_Srs(self):
        norm, e = nevpt2.Srs(mc, dms, eris)
        self.assertAlmostEqual(e, -0.017531645975808627, 7)
        self.assertAlmostEqual(norm, 0.056323606234166601, 7)

    def test_Sir(self):
        norm, e = nevpt2.Sir(mc, dms, eris)
        self.assertAlmostEqual(e, -0.033866295344083322, 7)
        self.assertAlmostEqual(norm, 0.074269050656629421, 7)

    def test_energy(self):
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.10315217594326213, 7)

    def test_energy1(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
            O   0 0 0
            O   0 0 1.207''',
            basis = '6-31g',
            spin = 2)
        m = scf.RHF(mol)
        m.scf()
        mc = mcscf.CASCI(m, 6, 8)
        mc.fcisolver.conv_tol = 1e-16
        mc.kernel()
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.16978532268234559, 6)


if __name__ == "__main__":
    print("Full Tests for nevpt2")
    unittest.main()

