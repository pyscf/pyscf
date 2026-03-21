#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

import tempfile
import unittest
import copy
import numpy
import h5py
from functools import reduce

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import mp
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import addons
from pyscf.fci import direct_uhf
from pyscf.cc import uccsdt

def setUpModule():
    global mol, rhf, mf, myucc, mol_s2, mf_s2, myucc2, eris
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol_grad = 1e-8
    rhf.kernel()
    mf = scf.addons.convert_to_uhf(rhf)
    myucc = cc.UCCSDT(mf).run(conv_tol=1e-10)

    mol_s2 = gto.Mole()
    mol_s2.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol_s2.basis = '631g'
    mol_s2.spin = 2
    mol_s2.verbose = 5
    mol_s2.output = '/dev/null'
    mol_s2.build()
    mf_s2 = scf.UHF(mol_s2).run()
    eris = uccsdt.UCCSDT(mf_s2).ao2mo()
    myucc2 = cc.UCCSDT(mf_s2).run(conv_tol=1e-10)

def tearDownModule():
    global mol, rhf, mf, myucc, mol_s2, mf_s2, myucc2, eris
    mol.stdout.close()
    mol_s2.stdout.close()
    del mol, rhf, mf, myucc, mol_s2, mf_s2, myucc2, eris

class KnownValues(unittest.TestCase):

    def test_rccsdt_s0(self):
        mf = scf.RHF(mol).run()
        myrcc = cc.RCCSDT(mf, compact_tamps=False).run(conv_tol=1e-10)
        self.assertAlmostEqual(myrcc.e_tot, myucc.e_tot, 8)

    def test_blocksize_s0(self):
        mycc = cc.UCCSDT(mf)
        mycc.conv_tol = 1e-10
        mycc.blksize_o_aaa = 4
        mycc.blksize_v_aaa = 7
        mycc.blksize_o_aab = 5
        mycc.blksize_v_aab = 3
        mycc.kernel()
        self.assertAlmostEqual(mycc.e_tot, myucc.e_tot, 8)
        # mycc = None
        # mycc = cc.UCCSDT(mf)
        # mycc.conv_tol = 1e-10
        # mycc.blksize_o_aaa = 3
        # mycc.blksize_v_aaa = 8
        # mycc.blksize_o_aab = 4
        # mycc.blksize_v_aab = 2
        # mycc.kernel()
        # self.assertAlmostEqual(mycc.e_tot, myucc.e_tot, 8)
        # mycc = None
        # mycc = cc.UCCSDT(mf)
        # mycc.conv_tol = 1e-10
        # mycc.blksize_o_aaa = 1
        # mycc.blksize_v_aaa = 1
        # mycc.blksize_o_aab = 1
        # mycc.blksize_v_aab = 1
        # mycc.kernel()
        # self.assertAlmostEqual(mycc.e_tot, myucc.e_tot, 8)

    def test_blocksize_s2(self):
        mycc = cc.UCCSDT(mf_s2)
        mycc.conv_tol = 1e-10
        mycc.blksize_o_aaa = 4
        mycc.blksize_v_aaa = 7
        mycc.blksize_o_aab = 5
        mycc.blksize_v_aab = 3
        mycc.kernel()
        self.assertAlmostEqual(mycc.e_tot, myucc2.e_tot, 8)
        # mycc = None
        # mycc = cc.UCCSDT(mf_s2)
        # mycc.conv_tol = 1e-10
        # mycc.blksize_o_aaa = 3
        # mycc.blksize_v_aaa = 8
        # mycc.blksize_o_aab = 4
        # mycc.blksize_v_aab = 2
        # mycc.kernel()
        # self.assertAlmostEqual(mycc.e_tot, myucc2.e_tot, 8)
        # mycc = None
        # mycc = cc.UCCSDT(mf_s2)
        # mycc.conv_tol = 1e-10
        # mycc.blksize_o_aaa = 1
        # mycc.blksize_v_aaa = 1
        # mycc.blksize_o_aab = 1
        # mycc.blksize_v_aab = 1
        # mycc.kernel()
        # self.assertAlmostEqual(mycc.e_tot, myucc2.e_tot, 8)

    def test_high_memory_s0(self):
        cc1 = cc.UCCSDT(mf, compact_tamps=False).run(conv_tol=1e-10)
        self.assertAlmostEqual(cc1.e_tot, myucc.e_tot, 8)
        cc2 = cc.UCCSDT(mf, compact_tamps=True).run(conv_tol=1e-10)
        self.assertAlmostEqual(cc2.e_tot, myucc.e_tot, 8)
        self.assertAlmostEqual(abs(cc1.tamps[0][0] - cc2.tamps[0][0]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[0][1] - cc2.tamps[0][1]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][0] - cc2.tamps[1][0]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][1] - cc2.tamps[1][1]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][2] - cc2.tamps[1][2]).max(), 0, 7)
        t3_full, t3_tri = cc1.tamps[2], cc2.tamps[2]
        t3_full2tri = cc1.tamps_full2tri(t3_full)
        t3_tri2full = cc2.tamps_tri2full(t3_tri)
        self.assertAlmostEqual(abs(t3_full[0] - t3_tri2full[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[1] - t3_tri2full[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[2] - t3_tri2full[2]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[3] - t3_tri2full[3]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[0] - t3_full2tri[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[1] - t3_full2tri[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[2] - t3_full2tri[2]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[3] - t3_full2tri[3]).max(), 0, 7)

    def test_high_memory_s2(self):
        cc1 = cc.UCCSDT(mf_s2, compact_tamps=False).run(conv_tol=1e-10)
        self.assertAlmostEqual(cc1.e_tot, myucc2.e_tot, 8)
        cc2 = cc.UCCSDT(mf_s2, compact_tamps=True).run(conv_tol=1e-10)
        self.assertAlmostEqual(cc2.e_tot, myucc2.e_tot, 8)
        self.assertAlmostEqual(abs(cc1.tamps[0][0] - cc2.tamps[0][0]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[0][1] - cc2.tamps[0][1]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][0] - cc2.tamps[1][0]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][1] - cc2.tamps[1][1]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1][2] - cc2.tamps[1][2]).max(), 0, 7)
        t3_full, t3_tri = cc1.tamps[2], cc2.tamps[2]
        t3_full2tri = cc1.tamps_full2tri(t3_full)
        t3_tri2full = cc2.tamps_tri2full(t3_tri)
        self.assertAlmostEqual(abs(t3_full[0] - t3_tri2full[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[1] - t3_tri2full[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[2] - t3_tri2full[2]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_full[3] - t3_tri2full[3]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[0] - t3_full2tri[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[1] - t3_full2tri[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[2] - t3_full2tri[2]).max(), 0, 7)
        self.assertAlmostEqual(abs(t3_tri[3] - t3_full2tri[3]).max(), 0, 7)

    def test_with_df_s0(self):
        mf = scf.UHF(mol).density_fit(auxbasis='weigend').run()
        mycc = cc.UCCSDT(mf).run(conv_tol=1e-10)
        self.assertAlmostEqual(mycc.e_tot, -76.11948106436947, 8)

    def test_with_df_s2(self):
        mf = scf.UHF(mol_s2).density_fit(auxbasis='weigend').run()
        mycc = cc.UCCSDT(mf).run(conv_tol=1e-10)
        self.assertAlmostEqual(mycc.e_tot, -75.83479685448731, 8)

    def test_restart_s0(self):
        ftmp = tempfile.NamedTemporaryFile()
        cc1 = cc.UCCSDT(mf)
        cc1.max_cycle = 5
        cc1.kernel()
        ref = cc1.e_corr

        adiis = lib.diis.DIIS(mol)
        adiis.filename = ftmp.name
        cc1.diis = adiis
        cc1.max_cycle = 3
        cc1.kernel(tamps=None)
        self.assertAlmostEqual(cc1.e_corr, -0.13617537767875998, 7)

        tamps = cc1.vector_to_amplitudes(adiis.extrapolate())
        self.assertAlmostEqual(abs(tamps[0][0] - cc1.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[0][1] - cc1.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][0] - cc1.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][1] - cc1.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][2] - cc1.t2[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][0] - cc1.t3[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][1] - cc1.t3[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][2] - cc1.t3[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][3] - cc1.t3[3]).max(), 0, 9)
        cc1.diis = None
        cc1.max_cycle = 1
        import copy
        tmp_tamps = copy.deepcopy(tamps)
        cc1.kernel(tmp_tamps)
        self.assertAlmostEqual(cc1.e_corr, -0.13636112399459543, 7)

        cc1.diis = adiis
        cc1.max_cycle = 2
        cc1.kernel(tamps)
        self.assertAlmostEqual(cc1.e_corr, ref, 8)

        cc2 = cc.UCCSDT(mf)
        cc2.restore_from_diis_(ftmp.name)
        self.assertAlmostEqual(abs(cc1.t1[0] - cc2.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t1[1] - cc2.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[0] - cc2.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[1] - cc2.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[2] - cc2.t2[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[0] - cc2.t3[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[1] - cc2.t3[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[2] - cc2.t3[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[3] - cc2.t3[3]).max(), 0, 9)

    def test_restart_s2(self):
        ftmp = tempfile.NamedTemporaryFile()
        cc1 = cc.UCCSDT(mf_s2)
        cc1.max_cycle = 5
        cc1.kernel()
        ref = cc1.e_corr

        adiis = lib.diis.DIIS(mol)
        adiis.filename = ftmp.name
        cc1.diis = adiis
        cc1.max_cycle = 3
        cc1.kernel(tamps=None)
        self.assertAlmostEqual(cc1.e_corr, -0.10899528342067309, 7)

        tamps = cc1.vector_to_amplitudes(adiis.extrapolate())
        self.assertAlmostEqual(abs(tamps[0][0] - cc1.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[0][1] - cc1.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][0] - cc1.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][1] - cc1.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][2] - cc1.t2[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][0] - cc1.t3[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][1] - cc1.t3[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][2] - cc1.t3[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2][3] - cc1.t3[3]).max(), 0, 9)
        cc1.diis = None
        cc1.max_cycle = 1
        import copy
        tmp_tamps = copy.deepcopy(tamps)
        cc1.kernel(tmp_tamps)
        self.assertAlmostEqual(cc1.e_corr, -0.10909663534556953, 7)

        cc1.diis = adiis
        cc1.max_cycle = 2
        cc1.kernel(tamps)
        self.assertAlmostEqual(cc1.e_corr, ref, 8)

        cc2 = cc.UCCSDT(mf_s2)
        cc2.restore_from_diis_(ftmp.name)
        self.assertAlmostEqual(abs(cc1.t1[0] - cc2.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t1[1] - cc2.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[0] - cc2.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[1] - cc2.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[2] - cc2.t2[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[0] - cc2.t3[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[1] - cc2.t3[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[2] - cc2.t3[2]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3[3] - cc2.t3[3]).max(), 0, 9)

    def test_restart_s2_not_do_diis_max_t(self):
        ftmp = tempfile.NamedTemporaryFile()
        cc1 = cc.UCCSDT(mf_s2)
        cc1.max_cycle = 5
        cc1.do_diis_max_t = False
        cc1.kernel()
        ref = cc1.e_corr

        adiis = lib.diis.DIIS(mol)
        adiis.filename = ftmp.name
        cc1.diis = adiis
        cc1.max_cycle = 3
        cc1.kernel(tamps=None)
        self.assertAlmostEqual(cc1.e_corr, -0.10900065442286336, 7)

        tamps = cc1.vector_to_amplitudes(adiis.extrapolate())
        tamps.append(cc1.tamps[2])
        self.assertAlmostEqual(abs(tamps[0][0] - cc1.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[0][1] - cc1.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][0] - cc1.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][1] - cc1.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1][2] - cc1.t2[2]).max(), 0, 9)
        cc1.diis = None
        cc1.max_cycle = 1
        import copy
        tmp_tamps = copy.deepcopy(tamps)
        cc1.kernel(tmp_tamps)
        self.assertAlmostEqual(cc1.e_corr, -0.10907414414270558, 7)

        cc1.diis = adiis
        cc1.max_cycle = 2
        cc1.kernel(tamps)
        self.assertAlmostEqual(cc1.e_corr, ref, 8)

        cc2 = cc.UCCSDT(mf_s2)
        cc2.do_diis_max_t = False
        cc2.restore_from_diis_(ftmp.name)
        self.assertAlmostEqual(abs(cc1.t1[0] - cc2.t1[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t1[1] - cc2.t1[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[0] - cc2.t2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[1] - cc2.t2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2[2] - cc2.t2[2]).max(), 0, 9)

    def test_ERIS(self):
        ucc1 = cc.UCCSDT(mf)
        nao, nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2, nao, nmo))
        eris = cc.uccsdt._make_eris_incore_ucc(ucc1, mo_coeff)

        self.assertAlmostEqual(lib.fp(eris.pppp), -74.1982420227933, 11)
        self.assertAlmostEqual(lib.fp(eris.PPPP), 135.11334409711856, 11)
        self.assertAlmostEqual(lib.fp(eris.pPpP), -35.264209872485665, 11)

    def test_uccsdt_frozen(self):
        ucc1 = myucc.copy()
        ucc1.frozen = 1
        self.assertEqual(ucc1.nmo, (12, 12))
        self.assertEqual(ucc1.nocc, (4, 4))
        ucc1.set_frozen()
        self.assertEqual(ucc1.nmo, (12, 12))
        self.assertEqual(ucc1.nocc, (4, 4))
        ucc1.frozen = [0, 1]
        self.assertEqual(ucc1.nmo, (11, 11))
        self.assertEqual(ucc1.nocc, (3, 3))
        ucc1.frozen = [[0, 1], [0, 1]]
        self.assertEqual(ucc1.nmo, (11, 11))
        self.assertEqual(ucc1.nocc, (3, 3))
        ucc1.frozen = [1, 9]
        self.assertEqual(ucc1.nmo, (11, 11))
        self.assertEqual(ucc1.nocc, (4, 4))
        ucc1.frozen = [[1, 9], [1, 9]]
        self.assertEqual(ucc1.nmo, (11, 11))
        self.assertEqual(ucc1.nocc, (4, 4))
        ucc1.frozen = [9, 10, 12]
        self.assertEqual(ucc1.nmo, (10, 10))
        self.assertEqual(ucc1.nocc, (5, 5))
        ucc1.nmo = (13, 12)
        ucc1.nocc = (5, 4)
        self.assertEqual(ucc1.nmo, (13, 12))
        self.assertEqual(ucc1.nocc, (5, 4))

    def test_uccsdt_frozen(self):
        # Freeze 1s electrons
        frozen = [[0, 1], [0, 1]]
        ucc = cc.UCCSDT(mf_s2, frozen=frozen)
        ucc.diis_start_cycle = 1
        ucc.conv_tol = 1e-10
        ecc, tamps = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.07489537030646895, 8)

    def test_vector_to_amplitudes(self):
        tamps = myucc.vector_to_amplitudes(myucc.amplitudes_to_vector(myucc.tamps))
        self.assertAlmostEqual(abs(tamps[0][0] - myucc.tamps[0][0]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[0][1] - myucc.tamps[0][1]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[1][0] - myucc.tamps[1][0]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[1][1] - myucc.tamps[1][1]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[1][2] - myucc.tamps[1][2]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[2][0] - myucc.tamps[2][0]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[2][1] - myucc.tamps[2][1]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[2][2] - myucc.tamps[2][2]).max(), 0, 12)
        self.assertAlmostEqual(abs(tamps[2][3] - myucc.tamps[2][3]).max(), 0, 12)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.UHF(mol).apply(uccsdt.UCCSDT)
        nelec = (3, 3)
        nocc = nelec
        nmo = (5, 5)
        mycc.nocc = nocc
        mycc.nmo = nmo
        vec = numpy.zeros(mycc.vector_size())
        vec_orig = vec.copy()
        tamps = mycc.vector_to_amplitudes(vec)
        t1a, t1b = tamps[0]
        t2aa, t2ab, t2bb = tamps[1]
        t3aaa, t3aab, t3bba, t3bbb = tamps[2]
        t1a[:] = 1
        t1b[:] = 1
        t2aa[:] = 1
        t2ab[:] = 1
        t2bb[:] = 1
        t3aaa[:] = 1
        t3aab[:] = 1
        t3bba[:] = 1
        t3bbb[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    def test_vector_size(self):
        self.assertEqual(myucc.vector_size(), 25760)

    def test_zero_beta_electrons(self):
        mol = gto.M(atom='H', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]), spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        mycc = uccsdt.UCCSDT(mf).run()
        self.assertAlmostEqual(mycc.e_corr, 0, 9)

        mol = gto.M(atom='He', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]), spin=2, verbose=0)
        mf = scf.UHF(mol).run()
        mycc = uccsdt.UCCSDT(mf).run(conv_tol=1e-10)
        self.assertAlmostEqual(mycc.e_corr, -2.6906684460278273e-05, 9)
        self.assertEqual(mycc.tamps[0][1].size, 0)
        self.assertEqual(mycc.tamps[1][1].size, 0)
        self.assertEqual(mycc.tamps[1][2].size, 0)
        self.assertEqual(mycc.tamps[2][1].size, 0)
        self.assertEqual(mycc.tamps[2][2].size, 0)
        self.assertEqual(mycc.tamps[2][3].size, 0)
        mycc2 = cc.UCCSD(mf).run(conv_tol=1e-10)
        self.assertAlmostEqual(mycc2.e_corr, -2.6906684460278273e-05, 9)
        self.assertAlmostEqual(abs(mycc.t1[0] - mycc2.t1[0]).max(), 0, 8)
        self.assertAlmostEqual(abs(mycc.t2[0] - mycc2.t2[0]).max(), 0, 8)

    def test_reset(self):
        mycc = cc.UCCSDT(scf.UHF(mol).newton())
        mycc.reset(mol_s2)
        self.assertTrue(mycc.mol is mol_s2)
        self.assertTrue(mycc._scf.mol is mol_s2)


if __name__ == "__main__":
    print("Full Tests for uccsdt.UCCSDT")
    unittest.main()
