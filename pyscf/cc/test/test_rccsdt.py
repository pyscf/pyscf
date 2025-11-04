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
from functools import reduce
import unittest
import copy
import numpy
import numpy as np
import h5py

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import df
from pyscf import cc
from pyscf import ao2mo
from pyscf import mp
from pyscf.cc import rccsdt

def setUpModule():
    global mol, mf, eris, mycc
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    mycc = rccsdt.RCCSDT(mf)
    mycc.conv_tol = 1e-10
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)

def tearDownModule():
    global mol, mf, eris, mycc
    mol.stdout.close()
    del mol, mf, eris, mycc


class KnownValues(unittest.TestCase):
    def test_roccsdt(self):
        mf = scf.ROHF(mol).run()
        mycc = cc.RCCSDT(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -76.12042524193436, 6)

    def test_ERIS(self):
        mycc = rccsdt.RCCSDT(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = rccsdt._make_eris_incore_rcc(mycc, mo_coeff)
        self.assertAlmostEqual(lib.fp(eris.pppp), -74.19824202279376, 11)

    def test_dump_chk(self):
        cc1 = mycc.copy()
        cc1.nmo = mf.mo_energy.size
        cc1.nocc = mol.nelectron // 2
        cc1.dump_chk()
        cc1 = cc.RCCSDT(mf)
        cc1.__dict__.update(lib.chkfile.load(cc1.chkfile, 'rccsdt'))
        e = cc1.energy(cc1.tamps, eris)
        self.assertAlmostEqual(e, -0.1364767434621007, 7)

    def test_blksize(self):
        cc1 = cc.RCCSDT(mf)
        cc1.conv_tol = 1e-10
        cc1.blksize = 2
        cc1.blksize_oovv = 2
        cc1.blksize_oooo = 2
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.1364767434621007, 7)
        cc2 = cc.RCCSDT(mf)
        cc2.conv_tol = 1e-10
        cc2.blksize = 3
        cc2.blksize_oovv = 3
        cc2.blksize_oooo = 3
        cc2.kernel()
        self.assertAlmostEqual(cc2.e_corr, -0.1364767434621007, 7)
        cc3 = cc.RCCSDT(mf)
        cc3.conv_tol = 1e-10
        cc3.blksize = 1
        cc3.blksize_oovv = 1
        cc3.blksize_oooo = 1
        cc3.kernel()
        self.assertAlmostEqual(cc3.e_corr, -0.1364767434621007, 7)

    def test_incore_complete(self):
        cc1 = cc.RCCSDT(mf)
        cc1.incore_complete = True
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.1364767434621007, 7)

    def test_no_do_diis_max_t(self):
        cc1 = cc.RCCSDT(mf)
        cc1.do_diis_max_t = False
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.1364767434621007, 7)

    def test_high_memory(self):
        cc1 = cc.RCCSDT(mf, compact_tamps=False)
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.1364767434621007, 7)
        cc2 = cc.RCCSDT(mf)
        cc2.conv_tol = 1e-10
        cc2.kernel()
        self.assertAlmostEqual(cc2.e_corr, -0.1364767434621007, 7)
        self.assertAlmostEqual(abs(cc1.tamps[0] - cc2.tamps[0]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[1] - cc2.tamps[1]).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps[2] - cc2.tamps_tri2full(cc2.tamps[2])).max(), 0, 7)
        self.assertAlmostEqual(abs(cc1.tamps_full2tri(cc1.tamps[2]) - cc2.tamps[2]).max(), 0, 7)

    def test_no_diis(self):
        cc1 = cc.RCCSDT(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.1362172678103062, 7)

    def test_restart(self):
        ftmp = tempfile.NamedTemporaryFile()
        cc1 = cc.RCCSDT(mf)
        cc1.max_cycle = 5
        cc1.kernel()
        ref = cc1.e_corr

        adiis = lib.diis.DIIS(mol)
        adiis.filename = ftmp.name
        cc1.diis = adiis
        cc1.max_cycle = 3
        cc1.kernel(tamps=None)
        self.assertAlmostEqual(cc1.e_corr, -0.13618790413398396, 7)

        tamps = cc1.vector_to_amplitudes(adiis.extrapolate())
        self.assertAlmostEqual(abs(tamps[0] - cc1.t1).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[1] - cc1.t2).max(), 0, 9)
        self.assertAlmostEqual(abs(tamps[2] - cc1.t3).max(), 0, 9)
        cc1.diis = None
        cc1.max_cycle = 1
        import copy
        tmp_tamps = copy.deepcopy(tamps)
        cc1.kernel(tmp_tamps)
        self.assertAlmostEqual(cc1.e_corr, -0.13636637468987364, 7)

        cc1.diis = adiis
        cc1.max_cycle = 2
        cc1.kernel(tamps)
        self.assertAlmostEqual(cc1.e_corr, ref, 8)

        cc2 = cc.RCCSDT(mf)
        cc2.restore_from_diis_(ftmp.name)
        self.assertAlmostEqual(abs(cc1.t1 - cc2.t1).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2 - cc2.t2).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t3 - cc2.t3).max(), 0, 9)

    def test_amplitudes_to_vector(self):
        vec = mycc.amplitudes_to_vector(mycc.tamps)
        tamps = mycc.vector_to_amplitudes(vec)
        self.assertAlmostEqual(abs(tamps[0] - mycc.t1).max(), 0, 14)
        self.assertAlmostEqual(abs(tamps[1] - mycc.t2).max(), 0, 14)
        self.assertAlmostEqual(abs(tamps[2] - mycc.t3).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        tamps = mycc.vector_to_amplitudes(vec)
        vec1 = mycc.amplitudes_to_vector(tamps)
        tamps2 = mycc.vector_to_amplitudes(vec1)
        vec2 = mycc.amplitudes_to_vector(tamps2)
        self.assertAlmostEqual(abs(vec1 - vec2).max(), 0.0, 14)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.RHF(mol).apply(rccsdt.RCCSDT)
        nelec = (3, 3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        mycc.nocc = nocc
        mycc.nmo = nmo
        vec = numpy.zeros(mycc.vector_size())
        vec_orig = vec.copy()
        tamps = mycc.vector_to_amplitudes(vec)
        tamps[0][:] = 1
        tamps[1][:] = 1
        tamps[2][:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    def test_vector_size(self):
        self.assertEqual(mycc.vector_size(), 18920)

    def test_rccsdt_frozen(self):
        cc1 = mycc.copy()
        cc1.frozen = 1
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.set_frozen()
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [0, 1]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 3)
        cc1.frozen = [1, 9]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [9, 10, 12]
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 5)
        cc1.nmo = 10
        cc1.nocc = 6
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 6)

    def test_two_electrons(self):
        mol = gto.M(atom='He', basis=('631g', [[0, (.2, 1)], [0, (.5, 1)]]), verbose=0)
        mf = scf.RHF(mol).run()
        mycc1 = cc.CCSD(mf).run(conv_tol=1e-10)
        mycc2 = cc.RCCSDT(mf).run(conv_tol=1e-10)
        self.assertAlmostEqual(mycc1.e_corr, mycc2.e_corr, 9)
        self.assertAlmostEqual(abs(mycc1.t1 - mycc2.t1).max(), 0, 8)
        self.assertAlmostEqual(abs(mycc1.t2 - mycc2.t2).max(), 0, 8)
        self.assertAlmostEqual(abs(mycc2.tamps[2]).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for rccsdt.RCCSDT")
    unittest.main()
