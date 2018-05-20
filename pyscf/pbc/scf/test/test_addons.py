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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
cell = pbcgto.Cell()
cell.atom = '''
He 0 0 1
He 1 0 1
'''
cell.basis = [[0, [1., 1.]], [0, [0.5, 1]]]
cell.a = numpy.eye(3) * 3
cell.mesh = [10] * 3
cell.verbose = 5
cell.output = '/dev/null'
cell.build()
nao = cell.nao_nr()

kpts = cell.make_kpts([2,1,1])
kmf_ro = pscf.KROHF(cell, kpts=kpts).run()
kmf_r = pscf.KRHF(cell, kpts=kpts).convert_from_(kmf_ro)
kmf_u = pscf.addons.convert_to_uhf(kmf_r)
kmf_g = pscf.addons.convert_to_ghf(kmf_r)


def tearDownModule():
    global cell, kmf_ro, kmf_r, kmf_u, kmf_g
    cell.stdout.close()
    del cell, kmf_ro, kmf_r, kmf_u, kmf_g

class KnowValues(unittest.TestCase):
    def test_krhf_smearing(self):
        mf = pscf.KRHF(cell, cell.make_kpts([2,1,1]))
        nkpts = len(mf.kpts)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy_kpts = numpy.array([numpy.arange(nao)*.2+numpy.cos(i+.5)*.1
                                      for i in range(nkpts)])
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1656394960533021/2, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.94924016074521311/2, 9)

        mf.kernel()
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_kuhf_smearing(self):
        mf = pscf.KUHF(cell, cell.make_kpts([2,1,1]))
        nkpts = len(mf.kpts)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy_kpts = numpy.array([numpy.arange(nao)*.2+numpy.cos(i+.5)*.1
                                      for i in range(nkpts)])
        mo_energy_kpts = numpy.array([mo_energy_kpts,
                                      mo_energy_kpts+numpy.cos(mo_energy_kpts)*.02])
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 6.1803390081500869/2, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(mf.entropy, 0.9554526863670467/2, 9)

    def test_rhf_smearing(self):
        mf = pscf.RHF(cell)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy = numpy.arange(nao)*.2+numpy.cos(.5)*.1
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.0922723199786408, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.4152467504725415, 9)

        mf.kernel()
        self.assertAlmostEqual(mf.entropy, 0, 15)

    def test_uhf_smearing(self):
        mf = pscf.UHF(cell)
        pscf.addons.smearing_(mf, 0.1, 'fermi')
        mo_energy = numpy.arange(nao)*.2+numpy.cos(.5)*.1
        mo_energy = numpy.array([mo_energy, mo_energy+numpy.cos(mo_energy)*.02])
        mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 3.1007387905421022, 9)

        mf.smearing_method = 'gauss'
        occ = mf.get_occ(mo_energy)
        self.assertAlmostEqual(mf.entropy, 0.42189309944541731, 9)

    def test_convert_to_scf(self):
        from pyscf.pbc import dft
        from pyscf.pbc import df
        from pyscf.soscf import newton_ah
        cell1 = cell.copy()
        cell1.verbose = 0
        pscf.addons.convert_to_rhf(dft.RKS(cell1))
        pscf.addons.convert_to_uhf(dft.RKS(cell1))
        #pscf.addons.convert_to_ghf(dft.RKS(cell1))
        pscf.addons.convert_to_rhf(dft.UKS(cell1))
        pscf.addons.convert_to_uhf(dft.UKS(cell1))
        #pscf.addons.convert_to_ghf(dft.UKS(cell1))
        #pscf.addons.convert_to_rhf(dft.GKS(cell1))
        #pscf.addons.convert_to_uhf(dft.GKS(cell1))
        #pscf.addons.convert_to_ghf(dft.GKS(cell1))

        pscf.addons.convert_to_rhf(pscf.RHF(cell1).density_fit())
        pscf.addons.convert_to_uhf(pscf.RHF(cell1).density_fit())
        pscf.addons.convert_to_ghf(pscf.RHF(cell1).density_fit())
        pscf.addons.convert_to_rhf(pscf.ROHF(cell1).density_fit())
        pscf.addons.convert_to_uhf(pscf.ROHF(cell1).density_fit())
        pscf.addons.convert_to_ghf(pscf.ROHF(cell1).density_fit())
        pscf.addons.convert_to_rhf(pscf.UHF(cell1).density_fit())
        pscf.addons.convert_to_uhf(pscf.UHF(cell1).density_fit())
        pscf.addons.convert_to_ghf(pscf.UHF(cell1).density_fit())
        #pscf.addons.convert_to_rhf(pscf.GHF(cell1).density_fit())
        #pscf.addons.convert_to_uhf(pscf.GHF(cell1).density_fit())
        pscf.addons.convert_to_ghf(pscf.GHF(cell1).density_fit())

        pscf.addons.convert_to_rhf(pscf.RHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_uhf(pscf.RHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_ghf(pscf.RHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_rhf(pscf.ROHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_uhf(pscf.ROHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_ghf(pscf.ROHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_rhf(pscf.UHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_uhf(pscf.UHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_ghf(pscf.UHF(cell1).x2c().density_fit())
        #pscf.addons.convert_to_rhf(pscf.GHF(cell1).x2c().density_fit())
        #pscf.addons.convert_to_uhf(pscf.GHF(cell1).x2c().density_fit())
        pscf.addons.convert_to_ghf(pscf.GHF(cell1).x2c().density_fit())

        self.assertTrue (isinstance(pscf.addons.convert_to_rhf(pscf.RHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_uhf(pscf.RHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_ghf(pscf.RHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_rhf(pscf.UHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(pscf.addons.convert_to_uhf(pscf.UHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_ghf(pscf.UHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(pscf.addons.convert_to_rhf(pscf.GHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(pscf.addons.convert_to_uhf(pscf.GHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(pscf.addons.convert_to_ghf(pscf.GHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))

        mf1 = pscf.rhf.RHF(cell1)
        cell2 = cell1.copy()
        cell2.spin = 2
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell1)), pscf.hf.RHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2)), pscf.hf.RHF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.UHF(cell2)), pscf.rohf.ROHF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.UHF(cell1).newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2).density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2).mix_density_fit()).with_df, df.mdf.MDF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.ROHF(cell2)), pscf.rohf.ROHF))
        self.assertRaises(AssertionError, mf1.convert_from_, kmf_u)

        mf1 = pscf.rohf.ROHF(cell1)
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell1)), pscf.rohf.ROHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2)), pscf.rohf.ROHF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.UHF(cell1).newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2).density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell2).mix_density_fit()).with_df, df.mdf.MDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.RHF(cell1)), pscf.rohf.ROHF))
        self.assertRaises(AssertionError, mf1.convert_from_, kmf_u)

        mf1 = pscf.uhf.UHF(cell1)
        self.assertTrue (isinstance(mf1.convert_from_(pscf.RHF(cell1)), pscf.uhf.UHF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.RHF(cell1).newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.RHF(cell1).density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.RHF(cell1).mix_density_fit()).with_df, df.mdf.MDF))
        self.assertRaises(AssertionError, mf1.convert_from_, kmf_u)

        mf1 = pscf.ghf.GHF(cell1)
        self.assertTrue (isinstance(mf1.convert_from_(pscf.RHF(cell1)), pscf.ghf.GHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.UHF(cell1)), pscf.ghf.GHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.ROHF(cell1)), pscf.ghf.GHF))
        self.assertRaises(AssertionError, mf1.convert_from_, kmf_u)

    def test_convert_to_kscf(self):
        from pyscf.pbc import df
        from pyscf.soscf import newton_ah
        cell1 = cell.copy()
        cell1.verbose = 0
        pscf.addons.convert_to_rhf(pscf.KRHF(cell1))
        pscf.addons.convert_to_uhf(pscf.KRHF(cell1))
        pscf.addons.convert_to_ghf(pscf.KRHF(cell1))
        pscf.addons.convert_to_rhf(pscf.KROHF(cell1))
        pscf.addons.convert_to_uhf(pscf.KROHF(cell1))
        pscf.addons.convert_to_ghf(pscf.KROHF(cell1))
        pscf.addons.convert_to_rhf(pscf.KUHF(cell1))
        pscf.addons.convert_to_uhf(pscf.KUHF(cell1))
        pscf.addons.convert_to_ghf(pscf.KUHF(cell1))
        #pscf.addons.convert_to_rhf(pscf.KGHF(cell1))
        #pscf.addons.convert_to_uhf(pscf.KGHF(cell1))
        pscf.addons.convert_to_ghf(pscf.KGHF(cell1))

        self.assertTrue (isinstance(pscf.addons.convert_to_rhf(pscf.KRHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_uhf(pscf.KRHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_ghf(pscf.KRHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_rhf(pscf.KUHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(pscf.addons.convert_to_uhf(pscf.KUHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        self.assertFalse(isinstance(pscf.addons.convert_to_ghf(pscf.KUHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(pscf.addons.convert_to_rhf(pscf.KGHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        #self.assertFalse(isinstance(pscf.addons.convert_to_uhf(pscf.KGHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))
        #self.assertTrue (isinstance(pscf.addons.convert_to_ghf(pscf.KGHF(cell1).newton().density_fit().x2c()), newton_ah._CIAH_SOSCF))

        mf1 = pscf.khf.KRHF(cell1)
        cell2 = cell1.copy()
        cell2.spin = 2
        self.assertTrue (isinstance(mf1.convert_from_(kmf_u), pscf.khf.KRHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2)), pscf.khf.KRHF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.KUHF(cell2)), pscf.krohf.KROHF))
        self.assertFalse(isinstance(mf1.convert_from_(kmf_u.newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2).density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2).mix_density_fit()).with_df, df.mdf.MDF))
        self.assertFalse(isinstance(mf1.convert_from_(pscf.KROHF(cell2)), pscf.krohf.KROHF))
        self.assertRaises(AssertionError, mf1.convert_from_, pscf.UHF(cell1))

        mf1 = pscf.krohf.KROHF(cell1)
        self.assertTrue (isinstance(mf1.convert_from_(kmf_u), pscf.krohf.KROHF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2)), pscf.krohf.KROHF))
        self.assertFalse(isinstance(mf1.convert_from_(kmf_u.newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2).density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(pscf.KUHF(cell2).mix_density_fit()).with_df, df.mdf.MDF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_r), pscf.krohf.KROHF))
        self.assertRaises(AssertionError, mf1.convert_from_, pscf.UHF(cell1))

        mf1 = pscf.kuhf.KUHF(cell1)
        self.assertTrue (isinstance(mf1.convert_from_(kmf_r), pscf.kuhf.KUHF))
        self.assertFalse(isinstance(mf1.convert_from_(kmf_r.newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_r.density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_ro.mix_density_fit()).with_df, df.mdf.MDF))
        self.assertRaises(AssertionError, mf1.convert_from_, pscf.UHF(cell1))

    def test_convert_to_kghf(self):
        from pyscf.pbc import df
        from pyscf.soscf import newton_ah
        mf1 = pscf.kghf.KGHF(cell)
        self.assertTrue (isinstance(mf1.convert_from_(kmf_r), pscf.kghf.KGHF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_u), pscf.kghf.KGHF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_ro), pscf.kghf.KGHF))
        self.assertRaises(AssertionError, mf1.convert_from_, pscf.UHF(cell))

        self.assertTrue (isinstance(mf1.convert_from_(kmf_u), pscf.kghf.KGHF))
        self.assertFalse(isinstance(mf1.convert_from_(kmf_u.newton()), newton_ah._CIAH_SOSCF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_u.density_fit()).with_df, df.df.GDF))
        self.assertTrue (isinstance(mf1.convert_from_(kmf_u.mix_density_fit()).with_df, df.mdf.MDF))

    def test_canonical_occ(self):
        mf1 = pscf.kuhf.KUHF(cell)
        mo_energy_kpts = [numpy.array([[0, 2, 3, 4],[0, 0, 1, 2]])] * 2
        occ_ref = numpy.array([[[1, 0, 0, 0], [1, 1, 1, 0]]]*2)
        occ = mf1.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(abs(occ - occ_ref).max(), 0, 14)

        mf1 = pscf.addons.canonical_occ_(mf1)
        occ_ref = numpy.array([[[1, 1, 0, 0], [1, 1, 0, 0]]]*2)
        occ = mf1.get_occ(mo_energy_kpts)
        self.assertAlmostEqual(abs(occ - occ_ref).max(), 0, 14)



if __name__ == '__main__':
    print("Full Tests for pbc.scf.addons")
    unittest.main()
