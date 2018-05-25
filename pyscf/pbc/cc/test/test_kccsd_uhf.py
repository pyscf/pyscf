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

import unittest
import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc.cc import kccsd_uhf
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper

cell = gto.Cell()
cell.atom='''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.build()


def rand_t1_t2(cell, kpts, nocc, nvir):
    nkpts = len(kpts)
    nocca, noccb = nocc
    nvira, nvirb = nvir
    t1a = (numpy.random.random((nkpts,nocca,nvira)) +
           numpy.random.random((nkpts,nocca,nvira))*1j - .5-.5j)
    t1b = (numpy.random.random((nkpts,noccb,nvirb)) +
           numpy.random.random((nkpts,noccb,nvirb))*1j - .5-.5j)
    t2aa = (numpy.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira)) +
            numpy.random.random((nkpts,nkpts,nkpts,nocca,nocca,nvira,nvira))*1j - .5-.5j)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    t2aa = t2aa - t2aa.transpose(0,2,1,4,3,5,6)
    tmp = t2aa.copy()
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kk, kj]
        t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
    t2ab = (numpy.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
            numpy.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
    t2bb = (numpy.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
            numpy.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
    t2bb = t2bb - t2bb.transpose(0,2,1,4,3,5,6)
    tmp = t2bb.copy()
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kk, kj]
        t2bb[ki,kj,kk] = t2bb[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)

    t1 = (t1a, t1b)
    t2 = (t2aa, t2ab, t2bb)
    return t1, t2

class KnownValues(unittest.TestCase):
    def test_amplitudes_to_vector(self):
        numpy.random.seed(1)
        kpts = cell.make_kpts([1,2,3])
        nmo = (8, 8)
        nocc = (3, 2)
        nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
        t1, t2 = rand_t1_t2(cell, kpts, nocc, nvir)
        vec = kccsd_uhf.amplitudes_to_vector(t1, t2)
        r1, r2 = kccsd_uhf.vector_to_amplitudes(vec, nmo, nocc, len(kpts))
        self.assertAlmostEqual(abs(t1[0]-r1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t1[1]-r1[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[0]-r2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[1]-r2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[2]-r2[2]).max(), 0, 12)

        vec1 = kccsd_uhf.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(vec-vec1).max(), 0, 12)

    def test_spatial2spin(self):
        numpy.random.seed(1)
        kpts = cell.make_kpts([1,2,3])
        nmo = (8, 8)
        nocc = (3, 2)
        nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
        t1, t2 = rand_t1_t2(cell, kpts, nocc, nvir)

        orbspin = numpy.zeros((len(kpts),nmo[0]+nmo[1]), dtype=int)
        orbspin[:,1::2] = 1
        kconserv = kpts_helper.get_kconserv(cell, kpts)

        r1 = kccsd.spatial2spin(t1, orbspin, kconserv)
        r1 = kccsd.spin2spatial(r1, orbspin, kconserv)
        self.assertAlmostEqual(abs(r1[0]-t1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(r1[1]-t1[1]).max(), 0, 12)

        r2 = kccsd.spatial2spin(t2, orbspin, kconserv)
        r2 = kccsd.spin2spatial(r2, orbspin, kconserv)
        self.assertAlmostEqual(abs(r2[0]-t2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[1]-t2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(r2[2]-t2[2]).max(), 0, 12)

    def test_eris(self):
        numpy.random.seed(1)
        kpts = cell.make_kpts([1,1,3])
        kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None)
        nmo = cell.nao_nr()
        kmf.mo_occ = numpy.zeros((2,3,nmo))
        kmf.mo_occ[0,:,:3] = 1
        kmf.mo_occ[1,:,:1] = 1
        kmf.mo_energy = (numpy.arange(nmo) +
                         numpy.random.random((2,3,nmo)) * .3)
        kmf.mo_energy[kmf.mo_occ == 0] += 2
        kmf.mo_coeff = (numpy.random.random((2,3,nmo,nmo)) +
                        numpy.random.random((2,3,nmo,nmo))*1j - .5-.5j)

        mycc = kccsd_uhf.UCCSD(kmf)
        eris = mycc.ao2mo()

        self.assertAlmostEqual(lib.finger(eris.fock[0]), 0.53719738596848132 +0.83031462049142502j, 12)
        self.assertAlmostEqual(lib.finger(eris.fock[1]), 0.043623927844025398+0.20815796288019522j, 12)

        self.assertAlmostEqual(lib.finger(eris.oooo), 0.069272005353335234-0.91918598849814437j   , 12)
        self.assertAlmostEqual(lib.finger(eris.ooov),-0.11102591602412985 -0.22547559257192468j   , 12)
        self.assertAlmostEqual(lib.finger(eris.ovoo),-0.43316611085314383 +0.46332928875833412j   , 12)
        self.assertAlmostEqual(lib.finger(eris.oovv),-0.0210790434747509  +0.014579420546946333j  , 12)
        self.assertAlmostEqual(lib.finger(eris.ovov),-0.012502652811465032-0.17935352765240994j   , 12)
        self.assertAlmostEqual(lib.finger(eris.voov),-0.19896007985076614 +0.1506712105555727j    , 12)
        self.assertAlmostEqual(lib.finger(eris.vovv), 0.030821226129365279-0.0066862873208788564j , 12)
        self.assertAlmostEqual(lib.finger(eris.vvvv),-0.034440875664143202+0.0046064112779059859j , 12)

        self.assertAlmostEqual(lib.finger(eris.OOOO), 0.022244585921946797 +0.0098510485963121189j, 12)
        self.assertAlmostEqual(lib.finger(eris.OOOV), 0.0017350076109628478-0.17585332697537598j  , 12)
        self.assertAlmostEqual(lib.finger(eris.OVOO), 0.0029885784141976046-0.012933429922292152j , 12)
        self.assertAlmostEqual(lib.finger(eris.OOVV), 0.12114074555132036  -0.37147217578941899j  , 12)
        self.assertAlmostEqual(lib.finger(eris.OVOV), 0.27027558833057647  +0.078863401294151342j , 12)
        self.assertAlmostEqual(lib.finger(eris.VOOV), 0.021533851752780438 -0.1132174227278781j   , 12)
        self.assertAlmostEqual(lib.finger(eris.VOVV),-0.11543283026534926  -0.0061589707191600073j, 12)
        self.assertAlmostEqual(lib.finger(eris.VVVV), 0.88474864164857403  +0.073151618320953293j , 12)

        self.assertAlmostEqual(lib.finger(eris.ooOO),-0.29993133610675266-0.30813660030875378j    , 12)
        self.assertAlmostEqual(lib.finger(eris.ooOV), 0.45509943469027042-0.26928937724216295j    , 12)
        self.assertAlmostEqual(lib.finger(eris.ovOO), 0.04290687995152876-0.089050147628659801j   , 12)
        self.assertAlmostEqual(lib.finger(eris.ooVV), -0.1807890256453191-1.0796936926021954j     , 12)
        self.assertAlmostEqual(lib.finger(eris.ovOV), 0.12678156024431853+0.2657704104428586j     , 12)
        self.assertAlmostEqual(lib.finger(eris.voOV), 0.16721872989676509-0.10351034481447512j    , 12)
        self.assertAlmostEqual(lib.finger(eris.voVV), 0.36164679793927601-0.27176490381380319j    , 12)
        self.assertAlmostEqual(lib.finger(eris.vvVV),-0.14030701121351946-0.26763712901678205j    , 12)

        self.assertAlmostEqual(lib.finger(eris.OOoo),-0.0082154271980681946-0.23061587416024174j  , 12)
        self.assertAlmostEqual(lib.finger(eris.OOov), 0.17345394148293944  +0.075909796863553688j , 12)
        self.assertAlmostEqual(lib.finger(eris.OVoo), 0.31370764062450857  -0.063959200362633334j , 12)
        self.assertAlmostEqual(lib.finger(eris.OOvv),-0.0060673232922933392-0.0062613404217953562j, 12)
        self.assertAlmostEqual(lib.finger(eris.OVov), 0.063146048356760257 +0.26635217117638132j  , 12)
        self.assertAlmostEqual(lib.finger(eris.VOov), 0.039870874777245872 -0.19883065522742366j  , 12)
        self.assertAlmostEqual(lib.finger(eris.VOvv),-0.0056875152302605656+0.036530474333401154j , 12)
        self.assertAlmostEqual(lib.finger(eris.VVvv), 0.024229643362609226 -0.029428136004862739j , 12)


if __name__ == '__main__':
    print("KUCCSD tests")
    unittest.main()


