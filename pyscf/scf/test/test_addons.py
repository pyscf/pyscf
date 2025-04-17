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

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf.scf import addons


def setUpModule():
    global mol, mf, mol_dz, mol1, mf_u, mol2, sym_mf, mol3, sym_mf_u
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]

    mol.basis = {"H": '6-31g',
                 "O": '6-31g',}
    mol.build()
    mf = scf.RHF(mol).run()

    mol_dz = mol.copy()
    mol_dz.basis = 'cc-pvdz'
    mol_dz.cart = True
    mol_dz.build(False, False)

    mol1 = mol.copy()
    mol1.spin = 2
    mf_u = scf.UHF(mol1).run()

    mol2 = mol.copy()
    mol2.symmetry = True
    mol2.build(0,0)
    sym_mf = scf.RHF(mol2).run()

    mol3 = mol1.copy()
    mol3.symmetry = True
    mol3.spin = 2
    mol3.build(0,0)
    sym_mf_u = scf.UHF(mol3).run()

def tearDownModule():
    global mol, mf, mol_dz, mol1, mf_u, mol2, sym_mf, mol3, sym_mf_u
    mol.stdout.close()
    del mol, mf, mol_dz, mol1, mf_u, mol2, sym_mf, mol3, sym_mf_u


class KnownValues(unittest.TestCase):
    def test_project_mo_nr2nr(self):
        nao = mol.nao_nr()
        c = numpy.random.random((nao,nao))
        c1 = addons.project_mo_nr2nr(mol, c, mol)
        self.assertAlmostEqual(abs(c-c1).max(), 0, 12)

        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mo2 = addons.project_mo_nr2nr(mol, [mo1,mo1], mol_dz)
        self.assertAlmostEqual(abs(mo2[0]).sum(), 83.436359425591888, 11)
        self.assertAlmostEqual(abs(mo2[1]).sum(), 83.436359425591888, 11)

    def test_project_mo_r2r(self):
        nao = mol.nao_2c()
        c = numpy.random.random((nao*2,nao*2))
        c = c + numpy.sin(c)*1j
        c1 = addons.project_mo_r2r(mol, c, mol)
        self.assertAlmostEqual(abs(c-c1).max(), 0, 12)

        numpy.random.seed(15)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        mo1 = numpy.random.random((n4c,n4c)) + numpy.random.random((n4c,n4c))*1j
        mo2 = addons.project_mo_r2r(mol, [mo1,mo1], mol_dz)
        self.assertAlmostEqual(abs(mo2[0]).sum(), 2159.3715489514038, 11)
        self.assertAlmostEqual(abs(mo2[1]).sum(), 2159.3715489514038, 11)

    def test_project_mo_nr2r(self):
        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo1 = numpy.random.random((nao,nao))
        mo2 = addons.project_mo_nr2r(mol, [mo1,mo1], mol_dz)
        self.assertAlmostEqual(abs(mo2[0]).sum(), 172.66468850263556, 11)
        self.assertAlmostEqual(abs(mo2[1]).sum(), 172.66468850263556, 11)

        mo2 = addons.project_mo_nr2r(mol, mo1, mol_dz)
        self.assertAlmostEqual(abs(mo2).sum(), 172.66468850263556, 11)

    def test_project_dm_nr2nr(self):
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        x1 = addons.project_dm_nr2nr(mol, dm, mol)
        self.assertAlmostEqual(abs(dm-x1).max(), 0, 12)

        numpy.random.seed(15)
        mo = numpy.random.random((nao,10))
        mo1 = addons.project_mo_nr2nr(mol, mo, mol_dz)
        dm = numpy.dot(mo, mo.T)
        dmref = numpy.dot(mo1, mo1.T)
        dm1 = addons.project_dm_nr2nr(mol, [dm,dm], mol_dz)

        self.assertAlmostEqual(abs(dmref-dm1[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(dmref-dm1[1]).max(), 0, 11)
        self.assertAlmostEqual(lib.finger(dm1[0]), 73.603267455214876, 11)

    def test_project_dm_r2r(self):
        nao = mol.nao_2c()
        dm = numpy.random.random((nao*2,nao*2))
        dm = dm + numpy.sin(dm)*1j
        x1 = addons.project_dm_r2r(mol, dm, mol)
        self.assertTrue(numpy.allclose(dm, x1))

        numpy.random.seed(15)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        mo = numpy.random.random((n4c,10)) + numpy.random.random((n4c,10))*1j
        mo1 = addons.project_mo_r2r(mol, mo, mol_dz)
        dm = numpy.dot(mo, mo.T.conj())
        dmref = numpy.dot(mo1, mo1.T.conj())
        dm1 = addons.project_dm_r2r(mol, [dm,dm], mol_dz)

        self.assertAlmostEqual(abs(dmref-dm1[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(dmref-dm1[1]).max(), 0, 11)
        self.assertAlmostEqual(lib.finger(dm1[0]), -5.3701392643370607+15.484616570244016j, 11)

    def test_project_dm_nr2r(self):
        numpy.random.seed(15)
        nao = mol.nao_nr()
        mo = numpy.random.random((nao,10))
        mo1 = addons.project_mo_nr2r(mol, mo, mol_dz)
        dm = numpy.dot(mo, mo.T.conj())
        dmref = numpy.dot(mo1, mo1.T.conj())
        dm1 = addons.project_dm_nr2r(mol, [dm,dm], mol_dz)

        self.assertAlmostEqual(abs(dmref-dm1[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(dmref-dm1[1]).max(), 0, 11)
        self.assertAlmostEqual(lib.finger(dm1[0]), -13.580612999088892-20.209297457056557j, 11)

        dm1 = addons.project_dm_nr2r(mol, dm, mol_dz)
        self.assertAlmostEqual(abs(dmref-dm1).max(), 0, 11)

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
        mf = addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -107.13465364012296, 9)

        mol.charge = -1
        mol.spin = 1
        mf = scf.RHF(mol)
        mf = addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.3626325837689, 9)

        mol.charge = 1
        mol.spin = 1
        mf = scf.rhf.RHF(mol)
        mf = addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.10375514714799, 9)

        mol.charge = 1
        mol.spin = 1
        mf = scf.UHF(mol)
        mf = addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.17458104180083, 9)

        mol.charge = 0
        mol.spin = 0
        mf = scf.RHF(mol)
        mf = addons.frac_occ(mf)
        self.assertAlmostEqual(mf.scf(), -108.76171800006837, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:7], [2,2,2,2,2,2,2]))
        mol.stdout.close()

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
        mf = addons.dynamic_occ(mf)
        self.assertAlmostEqual(mf.scf(), -74.214503776693817, 9)
        mol.stdout.close()

    def test_follow_state(self):
        mf1 = addons.follow_state(mf).run()
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 9)

        mo0 = mf.mo_coeff[:,[0,1,2,3,5]]
        mf1 = addons.follow_state(mf, mo0)
        self.assertAlmostEqual(mf1.scf(), -75.178145727548511, 9)
        self.assertTrue(numpy.allclose(mf1.mo_occ[:6], [2,2,2,2,0,2]))

    def test_float_occ(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            C      0.   0   0'''
        mol.basis = 'cc-pvdz'
        mol.build()
        mf = scf.UHF(mol)
        mf = addons.float_occ(mf)
        self.assertAlmostEqual(mf.scf(), -37.590712883365917, 9)
        mol.stdout.close()

    def test_mom_occ(self):
        mf = dft.UKS(mol)
        mf.xc = 'b3lyp5'
        mf.scf()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        occ[0][4] = 0.
        occ[0][5] = 1.
        mf = addons.mom_occ(mf, mo0, occ)
        dm = mf.make_rdm1(mo0, occ)
        self.assertAlmostEqual(mf.scf(dm), -76.0606858747, 7)
        self.assertTrue(numpy.allclose(mf.mo_occ[0][:6], [1,1,1,1,0,1]))

        mf = scf.ROHF(mol).run()
        mo0 = mf.mo_coeff
        occ = mf.mo_occ
        setocc = numpy.zeros((2, occ.size))
        setocc[:, occ==2] = 1
        setocc[0][4] = 0
        setocc[0][5] = 1
        newocc = setocc[0][:] + setocc[1][:]
        mf = addons.mom_occ(mf, mo0, setocc)
        dm = mf.make_rdm1(mo0, newocc)
        mf.kernel(dm)
        self.assertAlmostEqual(mf.e_tot, -75.723654936331542, 9)
        self.assertTrue(numpy.allclose(mf.mo_occ[:6], [2,2,2,2,1,1]))

    def test_dynamic_level_shift(self):
        mf = scf.RHF(mol)
        mf = addons.dynamic_level_shift(mf)
        mf.init_guess = 'hcore'
        mf.diis = False
        mf.max_cycle = 4
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -75.91511679613092, 9)

        mf = dft.UKS(mol)
        mf = addons.dynamic_level_shift(mf)
        mf.init_guess = 'hcore'
        mf.max_cycle = 2
        mf.diis = False
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -71.6072956194109, 5)

    def test_convert_to_scf(self):
        from pyscf.x2c import x2c
        from pyscf.df import df_jk
        from pyscf.soscf import newton_ah
        addons.convert_to_rhf(dft.RKS(mol))
        addons.convert_to_uhf(dft.RKS(mol))
        addons.convert_to_ghf(dft.RKS(mol))
        addons.convert_to_rhf(dft.UKS(mol))
        addons.convert_to_uhf(dft.UKS(mol))
        addons.convert_to_ghf(dft.UKS(mol))
        #addons.convert_to_rhf(dft.GKS(mol))
        #addons.convert_to_uhf(dft.GKS(mol))
        addons.convert_to_ghf(dft.GKS(mol))

        self.assertTrue(isinstance(addons.convert_to_rhf(mf), scf.rhf.RHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(mf), scf.uhf.UHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(mf), scf.ghf.GHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(scf.UHF(mol)), scf.rhf.RHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(mf_u), scf.rohf.ROHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(mf_u), scf.uhf.UHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(mf_u), scf.ghf.GHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(sym_mf), scf.hf_symm.RHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(sym_mf), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(sym_mf), scf.ghf_symm.GHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(sym_mf_u), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(sym_mf_u), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(sym_mf_u), scf.ghf_symm.GHF))

        mf1 = mf.copy()
        self.assertTrue(isinstance(mf1.convert_from_(mf), scf.rhf.RHF))
        self.assertTrue(isinstance(mf1.convert_from_(mf_u), scf.rhf.RHF))
        self.assertFalse(isinstance(mf1.convert_from_(mf_u), scf.rohf.ROHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf), scf.rhf.RHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf_u), scf.rhf.RHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf_u), scf.rohf.ROHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf), scf.hf_symm.RHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf_u), scf.hf_symm.RHF))
        mf1 = mf_u.copy()
        self.assertTrue(isinstance(mf1.convert_from_(mf), scf.uhf.UHF))
        self.assertTrue(isinstance(mf1.convert_from_(mf_u), scf.uhf.UHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf), scf.uhf.UHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf_u), scf.uhf.UHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf), scf.uhf_symm.UHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf_u), scf.uhf_symm.UHF))
        mf1 = scf.GHF(mol)
        self.assertTrue(isinstance(mf1.convert_from_(mf), scf.ghf.GHF))
        self.assertTrue(isinstance(mf1.convert_from_(mf_u), scf.ghf.GHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf), scf.ghf.GHF))
        self.assertTrue(isinstance(mf1.convert_from_(sym_mf_u), scf.ghf.GHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf), scf.ghf_symm.GHF))
        self.assertFalse(isinstance(mf1.convert_from_(sym_mf_u), scf.ghf_symm.GHF))

        self.assertTrue(isinstance(addons.convert_to_rhf(scf.RHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(scf.RHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.RHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(scf.UHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(scf.UHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.UHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        #self.assertTrue(isinstance(addons.convert_to_rhf(scf.GHF(mol).density_fit(), remove_df=False),df_jk. _DFHF))
        #self.assertTrue(isinstance(addons.convert_to_uhf(scf.GHF(mol).density_fit(), remove_df=False),df_jk. _DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.GHF(mol).density_fit(), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.RHF(mol).density_fit(), out=scf.RHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.RHF(mol).density_fit(), out=scf.UHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.RHF(mol).density_fit(), out=scf.GHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.UHF(mol).density_fit(), out=scf.RHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.UHF(mol).density_fit(), out=scf.UHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.UHF(mol).density_fit(), out=scf.GHF(mol), remove_df=False), df_jk._DFHF))
        #self.assertFalse(isinstance(addons.convert_to_rhf(scf.GHF(mol).density_fit(), out=scf.RHF(mol), remove_df=False),df_jk. _DFHF))
        #self.assertFalse(isinstance(addons.convert_to_uhf(scf.GHF(mol).density_fit(), out=scf.UHF(mol), remove_df=False),df_jk. _DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.GHF(mol).density_fit(), out=scf.GHF(mol), remove_df=False), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.RHF(mol).density_fit(), out=scf.RHF(mol), remove_df=True), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.RHF(mol).density_fit(), out=scf.UHF(mol), remove_df=True), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.RHF(mol).density_fit(), out=scf.GHF(mol), remove_df=True), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.UHF(mol).density_fit(), out=scf.RHF(mol), remove_df=True), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.UHF(mol).density_fit(), out=scf.UHF(mol), remove_df=True), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.UHF(mol).density_fit(), out=scf.GHF(mol), remove_df=True), df_jk._DFHF))
        #self.assertFalse(isinstance(addons.convert_to_rhf(scf.GHF(mol).density_fit(), out=scf.RHF(mol), remove_df=True),df_jk. _DFHF))
        #self.assertFalse(isinstance(addons.convert_to_uhf(scf.GHF(mol).density_fit(), out=scf.UHF(mol), remove_df=True),df_jk. _DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.GHF(mol).density_fit(), out=scf.GHF(mol), remove_df=True), df_jk._DFHF))

        addons.convert_to_rhf(scf.RHF(mol).x2c().density_fit())
        addons.convert_to_uhf(scf.RHF(mol).x2c().density_fit())
        addons.convert_to_ghf(scf.RHF(mol).x2c().density_fit())
        addons.convert_to_rhf(scf.UHF(mol).x2c().density_fit())
        addons.convert_to_uhf(scf.UHF(mol).x2c().density_fit())
        addons.convert_to_ghf(scf.UHF(mol).x2c().density_fit())
        #addons.convert_to_rhf(scf.GHF(mol).x2c().density_fit())
        #addons.convert_to_uhf(scf.GHF(mol).x2c().density_fit())
        addons.convert_to_ghf(scf.GHF(mol).x2c().density_fit())

        self.assertFalse(isinstance(addons.convert_to_rhf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.RHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.UHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(addons.convert_to_rhf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(addons.convert_to_uhf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.GHF(mol).x2c().newton().density_fit()), newton_ah._CIAH_SOSCF))

        self.assertFalse(isinstance(addons.convert_to_rhf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.RHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_rhf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_uhf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.UHF(mol).newton().density_fit()), df_jk._DFHF))
        #self.assertFalse(isinstance(addons.convert_to_rhf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))
        #self.assertFalse(isinstance(addons.convert_to_uhf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))
        self.assertFalse(isinstance(addons.convert_to_ghf(scf.GHF(mol).newton().density_fit()), df_jk._DFHF))

        self.assertTrue(isinstance(addons.convert_to_rhf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.RHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_rhf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_uhf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.UHF(mol).density_fit().newton()), df_jk._DFHF))
        #self.assertTrue(isinstance(addons.convert_to_rhf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))
        #self.assertTrue(isinstance(addons.convert_to_uhf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))
        self.assertTrue(isinstance(addons.convert_to_ghf(scf.GHF(mol).density_fit().newton()), df_jk._DFHF))

    def test_get_ghf_orbspin(self):
        orbspin = addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ)
        self.assertEqual(list(orbspin), [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])

        orbspin = addons.get_ghf_orbspin(mf_u.mo_energy, mf_u.mo_occ, is_rhf=False)
        self.assertEqual(list(orbspin), [0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,1])

    def test_remove_lindep(self):
        mol = gto.M(verbose = 0,
                    atom = [('H', 0, 0, i*.5) for i in range(4)],
                    basis = ('sto-3g',[[0, [.002,1]]]))
        mf = addons.remove_linear_dep_(scf.RHF(mol), threshold=1e-8,
                                       lindep=1e-9).run()
        self.assertAlmostEqual(mf.e_tot, -1.6291001503057689, 7)

    @unittest.skip('Smearing ROHF with fix_spin does not work')
    def test_rohf_smearing(self):
        # Fe2 from https://doi.org/10.1021/acs.jpca.1c05769
        mol = gto.M(
            atom='''
        Fe       0. 0. 0.
        Fe       2.01 0. 0.
        ''', basis="lanl2dz", ecp="lanl2dz", symmetry=False, unit='Angstrom',
            spin=6, charge=0, verbose=0)
        myhf_s = scf.ROHF(mol)
        myhf_s = addons.smearing(myhf_s, sigma=0.01, method='gaussian', fix_spin=True)
        myhf_s.kernel()
        # macos py3.7 CI -242.4828982467762
        # linux py3.11 CI -242.48289824670388
        self.assertAlmostEqual(myhf_s.e_tot, -242.482898246, 6)
        self.assertAlmostEqual(myhf_s.entropy, 0.45197, 4)
        myhf2 = myhf_s.undo_smearing().newton()
        myhf2.kernel(myhf_s.make_rdm1())
        # FIXME: note 16mHa lower energy than myhf
        self.assertAlmostEqual(myhf2.e_tot, -244.9808750, 6)

        myhf_s.smearing_method = 'fermi'
        myhf_s.kernel()
        self.assertAlmostEqual(myhf_s.e_tot, -244.200255453, 6)
        self.assertAlmostEqual(myhf_s.entropy, 3.585155, 4)

    def test_rohf_smearing1(self):
        mol = gto.M(atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7''',
            charge = -1,
            spin = 1)
        mf = mol.RHF()
        mf = addons.smearing(mf, sigma=0.1)
        mf.kernel()
        self.assertAlmostEqual(mf.mo_occ.sum(), 15, 8)
        self.assertAlmostEqual(mf.e_tot, -106.9310800402, 8)

    def test_uhf_smearing(self):
        mol = gto.M(
            atom='''
        Fe       0. 0. 0.
        Fe       2.01 0. 0.
        ''', basis="lanl2dz", ecp="lanl2dz", symmetry=False, unit='Angstrom',
            spin=6, charge=0, verbose=0)
        myhf_s = scf.UHF(mol)
        myhf_s = addons.smearing_(myhf_s, sigma=0.01, method='fermi', fix_spin=True)
        myhf_s.kernel()
        self.assertAlmostEqual(myhf_s.e_tot, -244.9873314, 5)
        self.assertAlmostEqual(myhf_s.entropy, 0, 3)

        myhf_s = scf.UHF(mol)
        myhf_s = addons.smearing_(myhf_s, sigma=0.01, method='fermi', fix_spin=True)
        myhf_s.sigma = 0.1
        myhf_s.fix_spin = False
        myhf_s.conv_tol = 1e-7
        myhf_s.kernel()
        self.assertAlmostEqual(myhf_s.e_tot, -243.086989253, 5)
        self.assertAlmostEqual(myhf_s.entropy, 17.11431, 4)
        self.assertTrue(myhf_s.converged)

        myhf_s.mu = -0.2482816
        myhf_s.kernel(dm0=myhf_s.make_rdm1())
        self.assertAlmostEqual(myhf_s.e_tot, -243.086989253, 5)
        self.assertAlmostEqual(myhf_s.entropy, 17.11431, 4)

    def test_rhf_smearing_nelec(self):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = +1
        mol.spin = 1
        mol.build()
        mf = scf.hf.RHF(mol)
        mf = addons.frac_occ(mf)
        e_frac = mf.kernel()

        mf_smear = scf.RHF(mol)
        # a small sigma amplifies the errors in orbital energies, breaking the
        # orbital degeneracy.
        mf_smear = addons.smearing(mf_smear, sigma=1e-3, method='fermi')
        e_smear = mf_smear.kernel()
        self.assertAlmostEqual(abs(mf.mo_occ - mf_smear.mo_occ).max(), 0, 3)
        self.assertAlmostEqual(e_frac, e_smear, 6)

        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
            7      0.   0  -0.7
            7      0.   0   0.7'''
        mol.basis = 'cc-pvdz'
        mol.charge = +1
        mol.spin = 1
        mol.symmetry = 1
        mol.build()
        mf = scf.hf.RHF(mol)
        mf = addons.frac_occ(mf)
        e_frac = mf.kernel()

        mf_smear = scf.RHF(mol)
        mf_smear = addons.smearing(mf_smear, sigma=1e-6, method='fermi')
        e_smear = mf_smear.kernel()
        self.assertAlmostEqual(abs(mf.mo_occ - mf_smear.mo_occ).max(), 0, 5)
        self.assertAlmostEqual(e_frac, e_smear, 9)

    def test_smearing_mu0(self):
        def _hubbard_hamilts_pbc(L, U):
            h1e = numpy.zeros((L, L))
            g2e = numpy.zeros((L,)*4)
            for i in range(L):
                h1e[i, (i+1)%L] = h1e[(i+1)%L, i] = -1
                g2e[i, i, i, i] = U
            return h1e, g2e

        L = 10
        U = 4

        mol = gto.M()
        mol.nelectron = L
        mol.nao = L
        mol.incore_anyway = True
        mol.build()

        h1e, eri = _hubbard_hamilts_pbc(L, U)
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1e
        mf._eri = eri
        mf.get_ovlp = lambda *args: numpy.eye(L)
        mf_ft = addons.smearing(mf, sigma=.1, mu0=2., fix_spin=True)
        mf_ft.kernel()
        self.assertAlmostEqual(mf_ft.e_tot, -2.93405853397115, 5)
        self.assertAlmostEqual(mf_ft.entropy, 0.11867520273160392, 5)


if __name__ == "__main__":
    print("Full Tests for addons")
    unittest.main()
