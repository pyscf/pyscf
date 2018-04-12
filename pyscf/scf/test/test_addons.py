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
import scipy.linalg

from pyscf import gto
from pyscf import scf, dft

mol = gto.Mole()
mol.verbose = 0
mol.output = '/dev/null'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ]

mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()


class KnowValues(unittest.TestCase):
    def test_project_mo_nr2nr(self):
        nao = mol.nao_nr()
        c = numpy.random.random((nao,nao))
        c1 = scf.addons.project_mo_nr2nr(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2nr(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 83.342096002254607, 11)

        mol2.cart = True
        mo2 = scf.addons.project_mo_nr2nr(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 83.436359425591888, 11)

    def test_project_mo_r2r(self):
        nao = mol.nao_2c()
        c = numpy.random.random((nao*2,nao*2))
        c = c + numpy.sin(c)*1j
        c1 = scf.addons.project_mo_r2r(mol, c, mol)
        self.assertTrue(numpy.allclose(c, c1))

        numpy.random.seed(15)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        mo1 = numpy.random.random((n4c,n4c)) + numpy.random.random((n4c,n4c))*1j
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_r2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 2159.3715489514038, 11)

    def test_project_mo_nr2r(self):
        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mol2 = gto.Mole()
        mol2.atom = mol.atom
        mol2.basis = {'H': 'cc-pvdz', 'O': 'cc-pvdz'}
        mol2.build(False, False)
        mo2 = scf.addons.project_mo_nr2r(mol, mo1, mol2)
        self.assertAlmostEqual(abs(mo2).sum(), 172.66468850263556, 11)

    def test_frac_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -107.13465364012296, 9)

        mol.charge = -1
        mol.spin = 1
        mf = scf.RHF(mol)
        mf = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.3626325837689, 9)

        mol.charge = 1
        mol.spin = 1
        mf = scf.rhf.RHF(mol)
        mf = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.10375514714799, 9)

        mol.charge = 1
        mol.spin = 1
        mf = scf.UHF(mol)
        mf = scf.addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.17458104180083, 9)

    def test_dynamic_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            6      0.   0  -0.7
            6      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = 2
        mol.build()
        mf = scf.RHF(mol)
        mf = scf.addons.dynamic_occ(mf)
        self.assertAlmostEqual(mf.scf(), -74.214503776693817, 9)

    def test_follow_state(self):
        mf = scf.RHF(mol)
        mf.scf()
        mo0 = mf.mo_coeff[:,[0,1,2,3,5]]
        mf = scf.addons.follow_state(mf, mo0)
        self.assertAlmostEqual(mf.scf(), -75.178145727548511, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,0,2]))

    def test_float_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            C      0.   0   0'''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.UHF(mol)
        mf = scf.addons.float_occ(mf)
        self.assertAlmostEqual(mf.scf(), -37.590712883365917, 9)

    def test_mom_occ(self):
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.scf()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        occ[0][4] = 0.
        occ[0][5] = 1.
        mf = scf.addons.mom_occ(mf, mo0, occ)
        dm = mf.make_rdm1(mo0, occ)
        self.assertAlmostEqual(mf.scf(dm), -76.0606858747, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[0][:6], [1,1,1,1,0,1]))

        mf = dft.ROKS(mol)
        mf.xc = 'b3lyp'
        mf.scf()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        setocc = numpy.zeros((2, occ.size))
        setocc[:, occ==2] = 1
        setocc[0][4] = 0
        setocc[0][5] = 1
        newocc = setocc[0][:] + setocc[1][:]
        mf = scf.addons.mom_occ(mf, mo0, setocc)
        dm = mf.make_rdm1(mo0, newocc)
        self.assertAlmostEqual(mf.scf(dm), -76.0692546639, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,1,1]))

    def test_convert_to_scf(self):
        from pyscf.x2c import x2c
        from pyscf.df import df_jk
        from pyscf.soscf import newton_ah
        scf.addons.convert_to_rhf(dft.RKS(mol))
        scf.addons.convert_to_uhf(dft.RKS(mol))
        #scf.addons.convert_to_ghf(dft.RKS(mol))
        scf.addons.convert_to_rhf(dft.UKS(mol))
        scf.addons.convert_to_uhf(dft.UKS(mol))
        #scf.addons.convert_to_ghf(dft.UKS(mol))
        #scf.addons.convert_to_rhf(dft.GKS(mol))
        #scf.addons.convert_to_uhf(dft.GKS(mol))
        #scf.addons.convert_to_ghf(dft.GKS(mol))

        scf.addons.convert_to_rhf(scf.RHF(mol).density_fit())
        scf.addons.convert_to_uhf(scf.RHF(mol).density_fit())
        scf.addons.convert_to_ghf(scf.RHF(mol).density_fit())
        scf.addons.convert_to_rhf(scf.UHF(mol).density_fit())
        scf.addons.convert_to_uhf(scf.UHF(mol).density_fit())
        scf.addons.convert_to_ghf(scf.UHF(mol).density_fit())
        #scf.addons.convert_to_rhf(scf.GHF(mol).density_fit())
        #scf.addons.convert_to_uhf(scf.GHF(mol).density_fit())
        scf.addons.convert_to_ghf(scf.GHF(mol).density_fit())

        scf.addons.convert_to_rhf(scf.RHF(mol).x2c().density_fit())
        scf.addons.convert_to_uhf(scf.RHF(mol).x2c().density_fit())
        scf.addons.convert_to_ghf(scf.RHF(mol).x2c().density_fit())
        scf.addons.convert_to_rhf(scf.UHF(mol).x2c().density_fit())
        scf.addons.convert_to_uhf(scf.UHF(mol).x2c().density_fit())
        scf.addons.convert_to_ghf(scf.UHF(mol).x2c().density_fit())
        #scf.addons.convert_to_rhf(scf.GHF(mol).x2c().density_fit())
        #scf.addons.convert_to_uhf(scf.GHF(mol).x2c().density_fit())
        scf.addons.convert_to_ghf(scf.GHF(mol).x2c().density_fit())

        self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))

        self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        #self.assertFalse(isinstance(scf.addons.convert_to_rhf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))
        #self.assertFalse(isinstance(scf.addons.convert_to_uhf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(scf.addons.convert_to_ghf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))

        self.assertTrue(isinstance(scf.addons.convert_to_rhf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_uhf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_ghf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_rhf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_uhf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_ghf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        #self.assertTrue(isinstance(scf.addons.convert_to_rhf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))
        #self.assertTrue(isinstance(scf.addons.convert_to_uhf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(scf.addons.convert_to_ghf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))

if __name__ == "__main__":
    print("Full Tests for addons")
    unittest.main()

