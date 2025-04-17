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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import df

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 0,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_ao2mo(self):
        dfobj = df.DF(mol)
        # force DF intermediates to be saved on disk
        dfobj.max_memory = 0.01

        # Initialize _call_count, to test DF.prange function
        dfobj._call_count = 0

        # dfobj.build is called in dfobj.get_naoaux()
        self.assertEqual(dfobj.get_naoaux(), 116)

        #dfobj.build()
        cderi = dfobj._cderi

        nao = mol.nao_nr()
        eri0 = dfobj.get_eri()
        numpy.random.seed(1)
        mos = numpy.random.random((nao,nao*10))
        mos = (mos[:,:5], mos[:,5:11], mos[:,3:9], mos[:,2:4])
        mo_eri0 = ao2mo.kernel(eri0, mos)

        mo_eri1 = dfobj.ao2mo(mos)
        self.assertAlmostEqual(abs(mo_eri0-mo_eri1).max(), 0, 9)

        mo = numpy.random.random((nao,nao))
        mo_eri0 = ao2mo.kernel(eri0, mo)
        mo_eri1 = dfobj.ao2mo(mo)
        self.assertAlmostEqual(abs(mo_eri0-mo_eri1).max(), 0, 9)

    def test_cderi_to_save(self):
        with open(os.devnull, 'w') as f:
            ftmp = tempfile.NamedTemporaryFile()
            dfobj = df.DF(mol)
            dfobj.auxmol = df.addons.make_auxmol(mol, 'weigend')
            dfobj.verbose = 5
            dfobj.stdout = f
            dfobj._cderi_to_save = ftmp.name
            dfobj._cderi = 'abc'
            dfobj.kernel()
            eri0 = dfobj.get_eri()
        dfobj = df.DF(mol)
        dfobj._cderi = ftmp.name
        eri1 = dfobj.get_eri()
        self.assertAlmostEqual(abs(eri0-eri1).max(), 0, 9)

    def test_init_density_fit(self):
        from pyscf.df import df_jk
        from pyscf import cc
        from pyscf.cc import dfccsd
        self.assertTrue(isinstance(df.density_fit(scf.RHF(mol)), df_jk._DFHF))
        self.assertTrue(isinstance(df.density_fit(cc.CCSD(scf.RHF(mol))),
                                   dfccsd.RCCSD))

        mf = mol.RHF().density_fit().newton().x2c1e().undo_df()
        self.assertTrue(not isinstance(mf, df_jk._DFHF))
        self.assertEqual(mf.__class__.__name__, 'sfX2C1eSecondOrderRHF')

    def test_rsh_get_jk(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,nao,nao))
        dfobj = df.DF(mol)
        vj, vk = dfobj.get_jk(dm, hermi=0, omega=1.1)
        self.assertAlmostEqual(lib.fp(vj), -181.5033531437091, 4)
        self.assertAlmostEqual(lib.fp(vk), -37.78854217974532, 4)

        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=0, omega=1.1)
        self.assertAlmostEqual(abs(vj-vj1).max(), 0, 2)
        self.assertAlmostEqual(abs(vk-vk1).max(), 0, 2)

    def test_rsh_df4c_get_jk(self):
        with lib.temporary_env(lib.param, LIGHT_SPEED=1):
            nao = mol.nao_nr() * 4
            numpy.random.seed(1)
            dm = numpy.random.random((2,nao,nao)) + numpy.random.random((2,nao,nao))*1j
            dm[0] += scf.dhf.time_reversal_matrix(mol, dm[0])
            dm[1] += scf.dhf.time_reversal_matrix(mol, dm[1])
            dfobj = df.DF4C(mol)
            vj, vk = dfobj.get_jk(dm, hermi=0, omega=0.9)
            self.assertAlmostEqual(lib.fp(vj), 1364.9812117487595+215.73320482400422j, 3)
            self.assertAlmostEqual(lib.fp(vk), 159.036202745021+687.903428296142j , 3)

            vj1, vk1 = scf.dhf.get_jk(mol, dm, hermi=0, omega=0.9)
            self.assertAlmostEqual(abs(vj-vj1).max(), 0, 2)
            self.assertAlmostEqual(abs(vk-vk1).max(), 0, 2)

    def test_rsh_df_custom_storage(self):
        mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis='ccpvdz', max_memory=10, verbose=0)
        mf = mol.RKS().density_fit()
        mf.xc = 'lda+0.5*SR_HF(0.3)'
        with tempfile.NamedTemporaryFile() as ftmp:
            mf.with_df._cderi_to_save = ftmp.name
            mf.run()
        self.assertAlmostEqual(mf.e_tot, -103.4965622991, 6)

        mol.max_memory = 4000
        mf = mol.RKS(xc='lda+0.5*SR_HF(0.3)').density_fit()
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -103.4965622991, 6)

    def test_only_dfj(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')
        dm = numpy.eye(mol.nao)
        mf = mol.RHF().density_fit()
        refj, refk = mf.get_jk(mol, dm)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)

        mf = mol.RHF().density_fit(only_dfj=True)
        refk = mol.RHF().get_k(mol, dm)
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)

        dm = numpy.eye(mol.nao*2)
        mf = mol.GHF().density_fit()
        refj, refk = mf.get_jk(mol, dm)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)

        mf = mol.GHF().density_fit(only_dfj=True)
        refk = mol.GHF().get_k(mol, dm)
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)
        vk = mf.get_k(mol, dm)
        self.assertAlmostEqual(abs(vk - refk).max(), 0, 9)
        vj = mf.get_j(mol, dm)
        self.assertAlmostEqual(abs(vj - refj).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()
