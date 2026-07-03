#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
from functools import reduce

from pyscf import gto, scf, lib, symm
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import rccsdt_q


def setUpModule():
    global mol, rhf, mcc, mcc2
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .487)],
        [1 , (0. ,  .757 , .687)]]
    mol.symmetry = True
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()

    mcc = cc.CCSDT(rhf, compact_tamps=True)
    mcc.conv_tol = 1e-10
    mcc.blksize = 2
    mcc.blksize_oooo = 2
    mcc.blksize_oovv = 2
    mcc.ccsdt()

    mcc2 = cc.CCSDT(rhf, compact_tamps=False)
    mcc2.conv_tol = 1e-10
    mcc2.ccsdt()

def tearDownModule():
    global mol, rhf, mcc, mcc2
    mol.stdout.close()
    del mol, rhf, mcc, mcc2

class KnownValues(unittest.TestCase):
    def test_rccsdt_q(self):
        e_q_bracket, e_q_paren  = mcc.ccsdt_q()
        self.assertAlmostEqual(e_q_bracket, -0.00044374834015582527, 9)
        self.assertAlmostEqual(e_q_paren, -0.0004917163848923114, 9)
        e_q_bracket2, e_q_paren2 = mcc2.ccsdt_q()
        self.assertAlmostEqual(e_q_bracket2, -0.00044374834015582527, 9)
        self.assertAlmostEqual(e_q_paren2, -0.0004917163848923114, 9)

    def test_random(self):
        mol = gto.M()
        numpy.random.seed(42)
        nocc, nvir = 5, 9
        nmo = nocc + nvir

        eris = cc.rccsdt._PhysicistsERIs()
        eri1 = numpy.random.random((nmo, nmo, nmo, nmo)) - .5
        eri1 = eri1 + eri1.transpose(2, 1, 0, 3)
        eri1 = eri1 + eri1.transpose(0, 3, 2, 1)
        eri1 = eri1 + eri1.transpose(1, 0, 3, 2)
        eri1 *= .1
        eris.pppp = eri1
        f = numpy.random.random((nmo, nmo)) * .1
        eris.fock = f + f.T + numpy.diag(numpy.arange(nmo))
        eris.mo_energy = eris.fock.diagonal()

        t1 = numpy.random.random((nocc, nvir)) * .1
        t2 = numpy.random.random((nocc, nocc, nvir, nvir)) * .1
        t2 = t2 + t2.transpose(1, 0, 3, 2)
        t3_full = numpy.random.random((nocc, nocc, nocc, nvir, nvir, nvir)) * .1
        t3_full = t3_full + t3_full.transpose(1, 0, 2, 4, 3, 5) + t3_full.transpose(2, 1, 0, 5, 4, 3)
        t3_full = t3_full + t3_full.transpose(0, 2, 1, 3, 5, 4)
        mf = scf.RHF(mol)
        mycc = cc.CCSDT(mf, compact_tamps=False)
        mycc.incore_complete = True
        mycc.mo_energy = mycc._scf.mo_energy = numpy.arange(0., nocc + nvir)
        e_q_bracket, e_q_paren = rccsdt_q.kernel(mycc, eris, (t1, t2, t3_full))
        self.assertAlmostEqual(e_q_bracket, -1.1359579193293403, 9)
        self.assertAlmostEqual(e_q_paren, -256.1325101409764, 9)

        idx_i, idx_j, idx_k = numpy.meshgrid(numpy.arange(nocc), numpy.arange(nocc), numpy.arange(nocc), indexing='ij')
        t3_tri = t3_full[(idx_i <= idx_j) & (idx_j <= idx_k)].reshape(-1, nvir, nvir, nvir)
        mycc2 = cc.CCSDT(mf, compact_tamps=True)
        mycc2.incore_complete = True
        mycc2.mo_energy = mycc2._scf.mo_energy = numpy.arange(0., nocc + nvir)
        mycc2.nocc, mycc2.nmo = nocc, nmo
        e_q_bracket2, e_q_paren2 = rccsdt_q.kernel(mycc2, eris, (t1, t2, t3_tri))
        self.assertAlmostEqual(e_q_bracket2, -1.1359579193293403, 9)
        self.assertAlmostEqual(e_q_paren2, -256.1325101409764, 9)

if __name__ == "__main__":
    print("Full Tests for RCCSDT(Q)")
    unittest.main()
