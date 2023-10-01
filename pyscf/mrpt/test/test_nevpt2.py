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
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import mcscf
from pyscf import fci
from pyscf.mrpt import nevpt2


def nevpt2_dms(mc):
    mo_cas = mf.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    h1e = mc.h1e_for_cas()[0]
    h2e = ao2mo.incore.full(mf._eri, mo_cas)
    h2e = ao2mo.restore(1, h2e, norb).transpose(0,2,1,3)
    dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf',
                                             mc.ci, mc.ci, norb, nelec)
    # Test integral transformation incore algorithm
    eris = nevpt2._ERIS(mc, mc.mo_coeff, method='incore')
    # Test integral transformation outcore algorithm
    eris = nevpt2._ERIS(mc, mc.mo_coeff, method='outcore')
    dms = {'1': dm1, '2': dm2, '3': dm3, '4': dm4}
    return eris, dms

def setUpModule():
    global mol, mf, mc, eris, dms, norb, nelec
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
    eris, dms = nevpt2_dms(mc)

def tearDownModule():
    global mol, mf, mc, eris, dms
    mol.stdout.close()
    del mol, mf, mc, eris, dms


class KnownValues(unittest.TestCase):
    # energy values for H14 from Dalton

    def test_Sr(self):
        norm, e = nevpt2.Sr(mc, mc.ci, dms, eris)
        self.assertAlmostEqual(e, -0.0202461540, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.039479583324952064, delta=1.0e-7)

    def test_Si(self):
        norm, e = nevpt2.Si(mc, mc.ci, dms, eris)
        self.assertAlmostEqual(e, -0.0021282083, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.0037402334190064367, delta=1.0e-7)

    def test_Sijrs(self):
        norm, e = nevpt2.Sijrs(mc, eris)
        self.assertAlmostEqual(e, -0.0071505004, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.023107592349719219, delta=1.0e-7)

    def test_Sijr(self):
        norm, e = nevpt2.Sijr(mc, dms, eris)
        self.assertAlmostEqual(e, -0.0050346117, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.012664066951786257, delta=1.0e-7)

    def test_Srsi(self):
        norm, e = nevpt2.Srsi(mc, dms, eris)
        self.assertAlmostEqual(e, -0.0136954715, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.040695892654346914, delta=1.0e-7)

    def test_Srs(self):
        norm, e = nevpt2.Srs(mc, dms, eris)
        self.assertAlmostEqual(e, -0.0175312323, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.056323606234166601, delta=1.0e-7)

    def test_Sir(self):
        norm, e = nevpt2.Sir(mc, dms, eris)
        self.assertAlmostEqual(e, -0.0338666048, delta=1.0e-6)
        self.assertAlmostEqual(norm, 0.074269050656629421, delta=1.0e-7)

    def test_energy(self):
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.1031529251, delta=1.0e-6)

    def test_energy1(self):
        o2 = gto.M(
            verbose = 0,
            atom = '''
            O   0 0 0
            O   0 0 1.207''',
            basis = '6-31g',
            spin = 2)
        mf_o2 = scf.RHF(o2).run()
        mc = mcscf.CASCI(mf_o2, 6, 8)
        mc.fcisolver.conv_tol = 1e-16
        mc.kernel()
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.16978532268234559, 6)

    def test_reset(self):
        mol1 = gto.M(atom='C')
        pt = nevpt2.NEVPT(mc)
        pt.reset(mol1)
        self.assertTrue(pt.mol is mol1)
        self.assertTrue(pt._mc.mol is mol1)

    def test_for_occ2(self):
        mol = gto.Mole(
            verbose=0,
            atom = '''
                O    -3.3006173    2.2577663    0.0000000
                H    -4.0301066    2.8983985    0.0000000
                H    -2.5046061    2.8136052    0.0000000
                ''')
        mf = mol.UHF().run()

        mc = mcscf.CASSCF(mf, 3, 6)
        caslist = [4, 5, 3]
        mc.kernel(mc.sort_mo(caslist))
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.031179434919517, 6)

        mc = mcscf.CASSCF(mf, 3, 6)
        caslist = [4, 5, 1]
        mc.kernel(mc.sort_mo(caslist))
        e = nevpt2.NEVPT(mc).kernel()
        self.assertAlmostEqual(e, -0.031179434919517, 6)

    def test_multistate(self):
        # See issue #1081
        mol = gto.M(atom='''
        O  0.0000000000 0.0000000000 -0.1302052882
        H  1.4891244004 0.0000000000  1.0332262019
        H -1.4891244004 0.0000000000  1.0332262019
        ''',
            basis = '631g',
            symmetry = False)
        mf = scf.RHF(mol).run()

        mc = mcscf.CASSCF(mf, 6, [4,4])
        mc.fcisolver=fci.solver(mol,singlet=True)
        mc.fcisolver.nroots=2
        mc = mcscf.state_average_(mc, [0.5,0.5])
        mc.kernel()
        orbital = mc.mo_coeff.copy()

        mc = mcscf.CASCI(mf, 6, 8)
        mc.fcisolver=fci.solver(mol,singlet=True)
        mc.fcisolver.nroots=2
        mc.kernel(orbital)

        # Ground State
        mp0 = nevpt2.NEVPT(mc, root=0)
        mp0.kernel()
        e0 = mc.e_tot[0] + mp0.e_corr
        self.assertAlmostEqual(e0, -75.867171, 4) # From ORCA (4.2.1)

        # First Excited State
        mp1 = nevpt2.NEVPT(mc, root=1)
        mp1.kernel()
        e1 = mc.e_tot[1] + mp1.e_corr
        self.assertAlmostEqual(e1, -75.828469, 4) # From ORCA (4.2.1)


if __name__ == "__main__":
    print("Full Tests for nevpt2")
    unittest.main()
