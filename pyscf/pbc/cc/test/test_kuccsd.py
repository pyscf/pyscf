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
from pyscf.pbc.cc import eom_kccsd_ghf
from pyscf.pbc.cc import eom_kccsd_uhf
from pyscf.pbc.lib import kpts_helper

def setUpModule():
    global cell
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
    cell.mesh = [13,13,13]
    cell.build()

def tearDownModule():
    global cell
    del cell


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
    t2aa = t2aa - t2aa.transpose(1,0,2,4,3,5,6)
    tmp = t2aa.copy()
    for ki, kj, kk in kpts_helper.loop_kkk(nkpts):
        kl = kconserv[ki, kk, kj]
        t2aa[ki,kj,kk] = t2aa[ki,kj,kk] - tmp[ki,kj,kl].transpose(0,1,3,2)
    t2ab = (numpy.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb)) +
            numpy.random.random((nkpts,nkpts,nkpts,nocca,noccb,nvira,nvirb))*1j - .5-.5j)
    t2bb = (numpy.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb)) +
            numpy.random.random((nkpts,nkpts,nkpts,noccb,noccb,nvirb,nvirb))*1j - .5-.5j)
    t2bb = t2bb - t2bb.transpose(1,0,2,4,3,5,6)
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

        self.assertAlmostEqual(lib.fp(eris.fock[0]), 0.53719738596848132 +0.83031462049142502j, 11)
        self.assertAlmostEqual(lib.fp(eris.fock[1]), 0.043623927844025398+0.20815796288019522j, 11)

        self.assertAlmostEqual(lib.fp(eris.oooo), 0.10126616996580763    -0.89787173918481156j  , 11)
        self.assertAlmostEqual(lib.fp(eris.ooov), -0.035814310241888067  +0.20393025075274804j  , 11)
        self.assertAlmostEqual(lib.fp(eris.oovv), -0.032378345849800663  +0.015060519910448879j , 11)
        self.assertAlmostEqual(lib.fp(eris.ovov), 0.017919215232962762   -0.37180556037878837j  , 11)
        self.assertAlmostEqual(lib.fp(eris.voov), -0.33038865500581482   +0.18384096784449852j  , 11)
        self.assertAlmostEqual(lib.fp(eris.vovv), 0.078104278754223946   +0.0004014143354997223j, 11)
        self.assertAlmostEqual(lib.fp(eris.vvvv), -0.0199910973368542    -0.0019864189992825137j, 11)

        self.assertAlmostEqual(lib.fp(eris.OOOO), 0.022687859086726745   +0.0076542690105189095j, 11)
        self.assertAlmostEqual(lib.fp(eris.OOOV), -0.024119030579269278  -0.15249100640417029j  , 11)
        self.assertAlmostEqual(lib.fp(eris.OOVV), 0.085942751462484687   -0.27088394382044989j  , 11)
        self.assertAlmostEqual(lib.fp(eris.OVOV), 0.35291244981540776    +0.080119865729794376j , 11)
        self.assertAlmostEqual(lib.fp(eris.VOOV), 0.0045484393536995267  +0.0094123990059577414j, 11)
        self.assertAlmostEqual(lib.fp(eris.VOVV), -0.28341581692294759   +0.0022174023470048921j, 11)
        self.assertAlmostEqual(lib.fp(eris.VVVV), 0.96007536729340814    -0.019410945571596398j , 11)

        self.assertAlmostEqual(lib.fp(eris.ooOO), -0.32831508979976765   -0.32180378432620471j  , 11)
        self.assertAlmostEqual(lib.fp(eris.ooOV), 0.33617152217704632    -0.34130555912360216j  , 11)
        self.assertAlmostEqual(lib.fp(eris.ooVV), -0.00011230004797088339-1.2850251519380604j   , 11)
        self.assertAlmostEqual(lib.fp(eris.ovOV), 0.1365118156144336     +0.16999367231786541j  , 11)
        self.assertAlmostEqual(lib.fp(eris.voOV), 0.19736044623396901    -0.047060848969879901j , 11)
        self.assertAlmostEqual(lib.fp(eris.voVV), 0.44513499758124858    +0.06343379901453805j  , 11)
        self.assertAlmostEqual(lib.fp(eris.vvVV), -0.070971875998391304  -0.31253893124900545j  , 11)

        #self.assertAlmostEqual(lib.fp(eris.OOoo), 0.031140414688898856   -0.23913617484062258j  , 11)
        self.assertAlmostEqual(lib.fp(eris.OOov), 0.20355552926191381    +0.18712171841650935j  , 11)
        self.assertAlmostEqual(lib.fp(eris.OOvv), 0.070789122903945706   -0.013360818695166678j , 11)
        #self.assertAlmostEqual(lib.fp(eris.OVov), 0.38230103404493437    -0.019845885264560274j , 11)
        #self.assertAlmostEqual(lib.fp(eris.VOov), 0.081760186267865437   -0.052409714443657308j , 11)
        self.assertAlmostEqual(lib.fp(eris.VOvv), -0.036061642075282056  +0.019284185856121634j , 11)
        #self.assertAlmostEqual(lib.fp(eris.VVvv), 0.13458896578260207    -0.11322854172459119j  , 11)

    def test_spatial2spin_ip(self):
        numpy.random.seed(1)
        kpts = cell.make_kpts([1,2,3])
        nkpts = len(kpts)
        nmo = (8, 8)
        nocc = (3, 2)
        nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])

        orbspin = numpy.zeros((len(kpts),nmo[0]+nmo[1]), dtype=int)
        orbspin[:,1::2] = 1
        kconserv = kpts_helper.get_kconserv(cell, kpts)

        kshift = 0
        spin_r1_ip = (numpy.random.rand(nocc[0]+nocc[1])*1j +
                      numpy.random.rand(nocc[0]+nocc[1]) - 0.5 - 0.5*1j)
        spin_r2_ip = (numpy.random.rand(nkpts**2 * (nocc[0]+nocc[1])**2 * (nvir[0]+nvir[1])) +
                      numpy.random.rand(nkpts**2 * (nocc[0]+nocc[1])**2 * (nvir[0]+nvir[1]))*1j -
                      0.5 - 0.5*1j)
        spin_r2_ip = spin_r2_ip.reshape(nkpts, nkpts, (nocc[0]+nocc[1]),
                                        (nocc[0]+nocc[1]), (nvir[0]+nvir[1]))
        spin_r2_ip = eom_kccsd_ghf.enforce_2p_spin_ip_doublet(spin_r2_ip, kconserv, kshift, orbspin)

        [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb] = \
            eom_kccsd_ghf.spin2spatial_ip_doublet(spin_r1_ip, spin_r2_ip, kconserv, kshift, orbspin)

        r1, r2 = eom_kccsd_ghf.spatial2spin_ip_doublet([r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb],
                                                       kconserv, kshift, orbspin=orbspin)

        self.assertAlmostEqual(abs(r1-spin_r1_ip).max(), 0, 12)
        self.assertAlmostEqual(abs(r2-spin_r2_ip).max(), 0, 12)

    def test_spatial2spin_ea(self):
        numpy.random.seed(1)
        kpts = cell.make_kpts([1,2,3])
        nkpts = len(kpts)
        nmo = (8, 8)
        nocc = (3, 2)
        nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])

        orbspin = numpy.zeros((len(kpts),nmo[0]+nmo[1]), dtype=int)
        orbspin[:,1::2] = 1
        kconserv = kpts_helper.get_kconserv(cell, kpts)

        kshift = 0
        spin_r1_ea = (numpy.random.rand(nvir[0]+nvir[1])*1j +
                      numpy.random.rand(nvir[0]+nvir[1]) - 0.5 - 0.5*1j)
        spin_r2_ea = (numpy.random.rand(nkpts**2 * (nocc[0]+nocc[1])* (nvir[0]+nvir[1])**2) +
                      numpy.random.rand(nkpts**2 * (nocc[0]+nocc[1])* (nvir[0]+nvir[1])**2)*1j -
                      0.5 - 0.5*1j)
        spin_r2_ea = spin_r2_ea.reshape(nkpts, nkpts, (nocc[0]+nocc[1]),
                                        (nvir[0]+nvir[1]), (nvir[0]+nvir[1]))
        spin_r2_ea = eom_kccsd_ghf.enforce_2p_spin_ea_doublet(spin_r2_ea, kconserv, kshift, orbspin)

        [r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb] = \
            eom_kccsd_ghf.spin2spatial_ea_doublet(spin_r1_ea, spin_r2_ea, kconserv, kshift, orbspin)

        r1, r2 = eom_kccsd_ghf.spatial2spin_ea_doublet([r1a, r1b], [r2aaa, r2baa, r2abb, r2bbb],
                                                       kconserv, kshift, orbspin=orbspin)

        self.assertAlmostEqual(abs(r1-spin_r1_ea).max(), 0, 12)
        self.assertAlmostEqual(abs(r2-spin_r2_ea).max(), 0, 12)

if __name__ == '__main__':
    print("KUCCSD tests")
    unittest.main()
