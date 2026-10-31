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
from pyscf import gto, scf, solvent
from pyscf.solvent import ddcosmo, ddpcm

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_ddpcm_scf(self):
        mol = gto.M(atom='''
               6        0.000000    0.000000   -0.542500
               8        0.000000    0.000000    0.677500
               1        0.000000    0.935307   -1.082500
               1        0.000000   -0.935307   -1.082500
                    ''', basis='sto3g', verbose=7,
                    output='/dev/null')
        pcm = ddpcm.DDPCM(mol)
        pcm.lmax = 6
        pcm.lebedev_order = 17
        mf = scf.RHF(mol).ddPCM(pcm).run()
        self.assertAlmostEqual(mf.e_tot, -112.3544929827, 5)

    def test_reset(self):
        mol1 = gto.M(atom='H 0 0 0; H 0 0 .9', basis='cc-pvdz')
        mf = scf.RHF(mol).density_fit().ddPCM().newton()
        mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertTrue(mf.with_solvent.mol is mol1)
        self.assertTrue(mf._scf.with_df.mol is mol1)
        self.assertTrue(mf._scf.with_solvent.mol is mol1)

class NamedSolvent(unittest.TestCase):
    def test_solvent_name(self):
        cm = ddpcm.DDPCM(mol, 'toluene')
        self.assertEqual(cm.solvent, 'toluene')
        self.assertAlmostEqual(cm.eps, 2.3741, 12)
        self.assertAlmostEqual(cm.eps_optical, 1.4961**2, 12)
        self.assertAlmostEqual(cm.get_eps_optical(), 1.4961**2, 12)

    def test_default_solvent(self):
        cm = ddpcm.DDPCM(mol)
        self.assertEqual(cm.solvent, '')
        self.assertAlmostEqual(cm.eps, ddcosmo.EPS_WATER, 12)
        self.assertIsNone(cm.eps_optical)

    def test_unknown_solvent(self):
        self.assertRaises(RuntimeError, ddpcm.DDPCM, mol, 'unobtainium')

    def test_solvent_energy(self):
        cm = ddpcm.DDPCM(mol, 'acetonitrile')
        e_tot = scf.RHF(mol).ddPCM(cm).kernel()

        cm_ref = ddpcm.DDPCM(mol)
        cm_ref.eps = 35.688
        e_ref = scf.RHF(mol).ddPCM(cm_ref).kernel()
        self.assertAlmostEqual(e_tot, e_ref, 12)

    def test_solvent_name_in_constructor(self):
        e_tot = scf.RHF(mol).ddPCM('acetonitrile').kernel()
        e_ref = scf.RHF(mol).ddPCM(ddpcm.DDPCM(mol, 'acetonitrile')).kernel()
        self.assertAlmostEqual(e_tot, e_ref, 12)
        e_ref = solvent.ddPCM(scf.RHF(mol), 'acetonitrile').kernel()
        self.assertAlmostEqual(e_tot, e_ref, 12)
        self.assertEqual(solvent.ddPCM(mol, 'acetonitrile').solvent, 'acetonitrile')

    def test_solvent_updated_after_build(self):
        cm = ddpcm.DDPCM(mol, 'water')
        cm.build()
        A_diele = cm._intermediates['A_diele']
        cm.solvent = 'toluene'
        # A_diele was computed for water. The intermediates have to be discarded.
        self.assertFalse(cm._intermediates)
        cm.build()
        self.assertTrue(abs(cm._intermediates['A_diele'] - A_diele).max() > 1e-6)

if __name__ == "__main__":
    print("Full Tests for ddpcm")
    unittest.main()
