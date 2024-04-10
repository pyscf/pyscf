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
from functools import reduce

from pyscf import gto, scf, lib, symm
from pyscf import cc
from pyscf.cc import uccsd_t
from pyscf.cc import gccsd_t

def setUpModule():
    global mol, mol1, mf, myucc, mygcc
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.spin = 2
    mol.basis = '3-21g'
    mol.symmetry = 'C2v'
    mol.build()
    mol1 = mol.copy()
    mol1.symmetry = False

    mf = scf.UHF(mol1).run(conv_tol=1e-14)
    myucc = cc.UCCSD(mf).run()
    mygcc = cc.GCCSD(mf).run()

def tearDownModule():
    global mol, mol1, mf, myucc, mygcc
    mol.stdout.close()
    del mol, mol1, mf, myucc, mygcc

class KnownValues(unittest.TestCase):
    def test_gccsd_t_compare_uccsd_t(self):
        self.assertAlmostEqual(myucc.ccsd_t(), mygcc.ccsd_t(t1=None), 7)

    def test_gccsd_t(self):
        mf1 = mf.copy()
        nao, nmo = mf.mo_coeff[0].shape
        numpy.random.seed(10)
        mf1.mo_coeff = numpy.random.random((2,nao,nmo))

        numpy.random.seed(12)
        nocca, noccb = mol.nelec
        nmo = mf1.mo_occ[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        t1a  = .1 * numpy.random.random((nocca,nvira))
        t1b  = .1 * numpy.random.random((noccb,nvirb))
        t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira))
        t2aa = t2aa - t2aa.transpose(0,1,3,2)
        t2aa = t2aa - t2aa.transpose(1,0,2,3)
        t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb))
        t2bb = t2bb - t2bb.transpose(0,1,3,2)
        t2bb = t2bb - t2bb.transpose(1,0,2,3)
        t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb))

        mycc = cc.GCCSD(mf1)
        t1 = mycc.spatial2spin((t1a, t1b        ))
        t2 = mycc.spatial2spin((t2aa, t2ab, t2bb))
        eris = mycc.ao2mo()
        e3a = gccsd_t.kernel(mycc, eris, t1, t2)
        self.assertAlmostEqual(e3a, 9877.2780859693339, 5)

    def test_gccsd_t_complex(self):
        nocc, nvir = 4, 6
        nmo = nocc + nvir
        numpy.random.seed(1)
        eris = cc.gccsd._PhysicistsERIs()
        h = (numpy.random.random((nmo,nmo)) +
             numpy.random.random((nmo,nmo)) * .6j - .5-.3j)
        eris.fock = h + h.T.conj() + numpy.diag(numpy.arange(nmo)) * 2
        eri1 = (numpy.random.random((nmo,nmo,nmo,nmo)) +
                numpy.random.random((nmo,nmo,nmo,nmo))*.8j - .5-.4j)
        eri1 = eri1 - eri1.transpose(0,1,3,2)
        eri1 = eri1 - eri1.transpose(1,0,2,3)
        eri1 = eri1 + eri1.transpose(2,3,0,1).conj()
        eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]
        eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:]
        eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:]
        t2 = (numpy.random.random((nocc,nocc,nvir,nvir)) +
              numpy.random.random((nocc,nocc,nvir,nvir))*.8j - .5-.4j)
        t2 = t2 - t2.transpose(0,1,3,2)
        t2 = t2 - t2.transpose(1,0,2,3)
        t1 = (numpy.random.random((nocc,nvir)) +
              numpy.random.random((nocc,nvir))*.8j - .5-.4j)
        eris.mo_energy = eris.fock.diagonal().real

        gcc = cc.gccsd.GCCSD(scf.GHF(gto.M()))
        self.assertAlmostEqual(gccsd_t.kernel(gcc, eris, t1, t2),
                               (-104.15886718888137+0.30739952563327672j), 9)


if __name__ == "__main__":
    print("Full Tests for GCCSD(T)")
    unittest.main()
