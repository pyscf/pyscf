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
import tempfile
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import dft
from pyscf import scf

def setUpModule():
    global mol, method, mol1
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [[2, (0.,0.,0.)], ]
    mol.basis = {"He": 'cc-pvdz'}
    mol.build()
    method = dft.RKS(mol)

    mol1 = gto.Mole()
    mol1.verbose = 0
    mol1.output = None
    mol1.atom = 'He'
    mol1.basis = 'cc-pvdz'
    mol1.charge = 1
    mol1.spin = 1
    mol1.build()

def tearDownModule():
    global mol, method, mol1
    mol.stdout.close()
    del mol, method, mol1


class KnownValues(unittest.TestCase):
    def test_nr_lda(self):
        method.xc = 'lda, vwn_rpa'
        self.assertAlmostEqual(method.scf(), -2.8641551904776055, 9)

    def test_dks_lda(self):
        m = mol.DKS()
        self.assertAlmostEqual(m.kernel(), -2.8268242330361373, 9)

        m = mol.DKS().x2c()
        self.assertAlmostEqual(m.kernel(), -2.826788817256218, 9)

    def test_udks_lda(self):
        m = dft.dks.UDKS(mol)
        self.assertAlmostEqual(m.kernel(), -2.8268242330361373, 9)

        m = dft.dks.UDKS(mol).x2c()
        self.assertAlmostEqual(m.kernel(), -2.826788817256218, 9)

    def test_nr_pw91pw91(self):
        method.xc = 'pw91, pw91'
        self.assertAlmostEqual(method.scf(), -2.8914066724838849, 9)

    def test_nr_b88vwn(self):
        method.xc = 'b88, vwn'
        self.assertAlmostEqual(method.scf(), -2.9670729652962606, 9)

    def test_nr_xlyp(self):
        method.xc = 'xlyp'
        self.assertAlmostEqual(method.scf(), -2.9045738259332161, 9)

    def test_nr_b3lypg(self):
        method.xc = 'b3lypg'
        self.assertAlmostEqual(method.scf(), -2.9070540942168002, 9)

        m = mol.UKS()
        m.xc = 'b3lyp5'
        self.assertAlmostEqual(m.scf(), -2.89992555753, 9)

    def test_camb3lyp(self):
        self.assertAlmostEqual(mol.RKS(xc='camb3lyp').kernel(), -2.89299475730048, 9)
        self.assertAlmostEqual(mol.GKS(xc='camb3lyp').kernel(), -2.89299475730048, 9)
        self.assertAlmostEqual(mol.UKS(xc='camb3lyp').kernel(), -2.89299475730048, 9)

    def test_wb97(self):
        self.assertAlmostEqual(mol.RKS(xc='wb97').kernel(), -2.89430888240579, 9)
        self.assertAlmostEqual(mol.GKS(xc='wb97').kernel(), -2.89430888240579, 9)
        self.assertAlmostEqual(mol.UKS(xc='wb97').kernel(), -2.89430888240579, 9)
        # The old way to compute RSH, short-range = full-range - long-range
        xc = 'wb97 + 1e-9*HF'
        self.assertAlmostEqual(mol.RKS(xc=xc).kernel(), -2.89430888240579, 8)

    def test_hse(self):
        self.assertAlmostEqual(mol.RKS(xc='hse06').kernel(), -2.88908568982727, 9)
        self.assertAlmostEqual(mol.GKS(xc='hse06').kernel(), -2.88908568982727, 9)
        self.assertAlmostEqual(mol.UKS(xc='hse06').kernel(), -2.88908568982727, 9)
        # The old way to compute RSH, short-range = full-range - long-range
        xc = 'hse06 + 1e-9*HF'
        self.assertAlmostEqual(mol.RKS(xc=xc).kernel(), -2.88908568982727, 8)

    def test_nr_lda_1e(self):
        mf = dft.RKS(mol1).run()
        self.assertAlmostEqual(mf.e_tot, -1.936332393935281, 9)

    def test_nr_b3lypg_1e(self):
        mf = dft.ROKS(mol1).set(xc='b3lypg').run()
        self.assertAlmostEqual(mf.e_tot, -1.9931564410562266, 9)

    def test_xcfun_nr_blyp(self):
        m = mol.RKS()
        m._numint.libxc = dft.xcfun
        m.xc = 'b88,lyp'
        self.assertAlmostEqual(m.scf(), -2.8978518405, 9)

    def test_nr_m06l(self):
        m = mol.RKS()
        m.xc = 'm06l'
        self.assertAlmostEqual(m.scf(), -2.9039230673864243, 7)

        m = mol.UKS()
        m.xc = 'm06l'
        self.assertAlmostEqual(m.scf(), -2.9039230673864243, 7)

    def test_1e(self):
        mf = dft.RKS(gto.M(atom='H', spin=1)).run()
        self.assertTrue(isinstance(mf, dft.roks.ROKS))
        self.assertAlmostEqual(mf.e_tot, -0.43567023283650547)

        mf = dft.RKS(gto.M(atom='H', spin=1, symmetry=1)).run()
        self.assertTrue(isinstance(mf, dft.rks_symm.ROKS))
        self.assertAlmostEqual(mf.e_tot, -0.43567023283650547)

    def test_convert(self):
        rhf = scf.RHF(mol)
        uhf = scf.UHF(mol)
        ghf = scf.GHF(mol)
        rks = dft.RKS(mol)
        uks = dft.UKS(mol)
        gks = dft.GKS(mol)
        dhf = scf.DHF(mol)
        dks = dft.DKS(mol)
        udhf = scf.dhf.UDHF(mol)
        udks = dft.dks.UDKS(mol)

        self.assertTrue(isinstance(rhf.to_rhf(), scf.rhf.RHF))
        self.assertTrue(isinstance(rhf.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(rhf.to_ghf(), scf.ghf.GHF))
        self.assertTrue(isinstance(rhf.to_rks(), dft.rks.RKS))
        self.assertTrue(isinstance(rhf.to_uks(), dft.uks.UKS))
        self.assertTrue(isinstance(rhf.to_gks(), dft.gks.GKS))

        self.assertTrue(isinstance(rks.to_rhf(), scf.rhf.RHF))
        self.assertTrue(isinstance(rks.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(rks.to_ghf(), scf.ghf.GHF))
        self.assertTrue(isinstance(rks.to_rks('pbe'), dft.rks.RKS))
        self.assertTrue(isinstance(rks.to_uks('pbe'), dft.uks.UKS))
        self.assertTrue(isinstance(rks.to_gks('pbe'), dft.gks.GKS))

        self.assertTrue(isinstance(uhf.to_rhf(), scf.rhf.RHF))
        self.assertTrue(isinstance(uhf.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(uhf.to_ghf(), scf.ghf.GHF))
        self.assertTrue(isinstance(uhf.to_rks(), dft.rks.RKS))
        self.assertTrue(isinstance(uhf.to_uks(), dft.uks.UKS))
        self.assertTrue(isinstance(uhf.to_gks(), dft.gks.GKS))

        self.assertTrue(isinstance(rks.to_rhf(), scf.rhf.RHF))
        self.assertTrue(isinstance(rks.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(rks.to_ghf(), scf.ghf.GHF))
        self.assertTrue(isinstance(uks.to_rks('pbe'), dft.rks.RKS))
        self.assertTrue(isinstance(uks.to_uks('pbe'), dft.uks.UKS))
        self.assertTrue(isinstance(uks.to_gks('pbe'), dft.gks.GKS))

        #self.assertTrue(isinstance(ghf.to_rhf(), scf.rhf.RHF))
        #self.assertTrue(isinstance(ghf.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(ghf.to_ghf(), scf.ghf.GHF))
        #self.assertTrue(isinstance(ghf.to_rks(), dft.rks.RKS))
        #self.assertTrue(isinstance(ghf.to_uks(), dft.uks.UKS))
        self.assertTrue(isinstance(ghf.to_gks(), dft.gks.GKS))

        #self.assertTrue(isinstance(gks.to_rhf(), scf.rhf.RHF))
        #self.assertTrue(isinstance(gks.to_uhf(), scf.uhf.UHF))
        self.assertTrue(isinstance(gks.to_ghf(), scf.ghf.GHF))
        #self.assertTrue(isinstance(gks.to_rks('pbe'), dft.rks.RKS))
        #self.assertTrue(isinstance(gks.to_uks('pbe'), dft.uks.UKS))
        self.assertTrue(isinstance(gks.to_gks('pbe'), dft.gks.GKS))

        self.assertRaises(RuntimeError, dhf.to_rhf)
        self.assertRaises(RuntimeError, dhf.to_uhf)
        self.assertRaises(RuntimeError, dhf.to_ghf)
        self.assertRaises(RuntimeError, dks.to_rks)
        self.assertRaises(RuntimeError, dks.to_uks)
        self.assertRaises(RuntimeError, dks.to_gks)

        if scf.dhf.zquatev is not None:
            self.assertTrue(isinstance(dhf.to_dhf(), scf.dhf.RDHF))
            self.assertTrue(isinstance(dhf.to_dks(), dft.dks.RDKS))
            self.assertTrue(isinstance(dks.to_dhf(), scf.dhf.RDHF))
            self.assertTrue(isinstance(dks.to_dks('pbe'), dft.dks.RDKS))
        self.assertTrue(isinstance(dhf.to_dhf(), scf.dhf.DHF))
        self.assertTrue(isinstance(dhf.to_dks(), dft.dks.DKS))
        self.assertTrue(isinstance(dks.to_dhf(), scf.dhf.DHF))
        self.assertTrue(isinstance(dks.to_dks('pbe'), dft.dks.DKS))
        self.assertTrue(isinstance(udhf.to_dhf(), scf.dhf.DHF))
        self.assertTrue(isinstance(udhf.to_dks(), dft.dks.DKS))
        self.assertTrue(isinstance(udks.to_dhf(), scf.dhf.DHF))
        self.assertTrue(isinstance(udks.to_dks('pbe'), dft.dks.DKS))

    # issue 1986
    def test_init_guess_chkfile(self):
        with tempfile.NamedTemporaryFile() as tmpf:
            mol = gto.M(atom='He 0 0 0', basis='631g', charge=1, spin=1)
            mf = dft.RKS(mol)
            mf.chkfile = tmpf.name
            e1 = mf.kernel()
            mf = dft.RKS(mol)
            mf.init_guess = 'chkfile'
            mf.chkfile = tmpf.name
            mf.max_cycle = 1
            e2 = mf.kernel()
            self.assertAlmostEqual(e1, e2, 9)


if __name__ == "__main__":
    print("Full Tests for He")
    unittest.main()
