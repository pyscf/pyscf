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

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf.cc import dfccsd, eom_rccsd

def make_mycc1():
    mf1 = mf.copy()
    no = mol.nelectron // 2
    n = mol.nao_nr()
    nv = n - no
    mf1.mo_occ = numpy.zeros(mol.nao_nr())
    mf1.mo_occ[:no] = 2
    numpy.random.seed(12)
    mf1.mo_coeff = numpy.random.random((n,n))
    dm = mf1.make_rdm1(mf1.mo_coeff, mf1.mo_occ)
    fockao = mf1.get_hcore() + mf1.get_veff(mol, dm)
    mf1.mo_energy = numpy.einsum('pi,pq,qi->i', mf1.mo_coeff, fockao, mf1.mo_coeff)
    idx = numpy.hstack([mf1.mo_energy[:no].argsort(), no+mf1.mo_energy[no:].argsort()])
    mf1.mo_coeff = mf1.mo_coeff[:,idx]
    mycc1 = dfccsd.RCCSD(mf1)
    eris1 = mycc1.ao2mo()
    numpy.random.seed(12)
    r1 = numpy.random.random((no,nv)) - .9
    r2 = numpy.random.random((no,no,nv,nv)) - .9
    r2 = r2 + r2.transpose(1,0,3,2)
    mycc1.t1 = r1*1e-5
    mycc1.t2 = r2*1e-5
    return mf1, mycc1, eris1

def setUpModule():
    global mol, mf, cc1, mycc, mf1, mycc1, eris1, no, nv
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    cc1 = dfccsd.RCCSD(mf).run(conv_tol=1e-10)
    mycc = cc.ccsd.CCSD(mf).density_fit().set(max_memory=0)
    mycc.__dict__.update(cc1.__dict__)

    mf1, mycc1, eris1 = make_mycc1()
    no, nv = mycc1.t1.shape

def tearDownModule():
    global mol, mf, cc1, mycc, mf1, mycc1, eris1
    mol.stdout.close()
    del mol, mf, cc1, mycc, mf1, mycc1, eris1

class KnownValues(unittest.TestCase):
    def test_with_df(self):
        self.assertAlmostEqual(cc1.e_tot, -76.118403942938741, 6)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.ccsd.CCSD(mf).ao2mo(mo_coeff)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.oooo)), 4.962033460861587 , 11)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.ovoo)),-1.3666078517246127, 11)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.oovv)), 55.122525571320871, 11)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.ovvo)), 133.48517302161068, 11)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.ovvv)), 59.418747028576142, 11)
        self.assertAlmostEqual(lib.fp(numpy.array(eris.vvvv)), 43.562457227975969, 11)


    def test_df_ipccsd(self):
        e,v = mycc.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.42788191082629801, 6)

        e,v = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.42788191082629801, 6)
        self.assertAlmostEqual(e[1], 0.50229582430658171, 6)
        self.assertAlmostEqual(e[2], 0.68557652412060088, 6)

        myeom = eom_rccsd.EOMIP(mycc)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43584093045349137, 6)
        self.assertAlmostEqual(e[1], 0.50959675100507518, 6)
        self.assertAlmostEqual(e[2], 0.69021193094404043, 6)

    def test_df_ipccsd_koopmans(self):
        e,v = mycc.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.42788191082629801, 6)
        self.assertAlmostEqual(e[1], 0.50229582430658171, 6)
        self.assertAlmostEqual(e[2], 0.68557652412060088, 6)

        e,v = mycc.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.42788191082629801, 6)
        self.assertAlmostEqual(e[1], 0.50229582430658171, 6)
        self.assertAlmostEqual(e[2], 0.68557652412060088, 6)

    def test_df_ipccsd_partition(self):
        e,v = mycc.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42183117410776649, 6)
        self.assertAlmostEqual(e[1], 0.49650713906402066, 6)
        self.assertAlmostEqual(e[2], 0.6808175428439881 , 6)

        e,v = mycc.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.41392302194803809, 6)
        self.assertAlmostEqual(e[1], 0.49046066205501643, 6)
        self.assertAlmostEqual(e[2], 0.67472905602747868, 6)


    def test_df_eaccsd(self):
        self.assertAlmostEqual(mycc.e_tot, -76.118403942938741, 6)
        e,v = mycc.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.1903885587959659, 6)

        e,v = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.1903885587959659, 6)
        self.assertAlmostEqual(e[1], 0.2833972143749155, 6)
        self.assertAlmostEqual(e[2], 0.5222497886685452, 6)

        myeom = eom_rccsd.EOMEA(mycc)
        lv = myeom.eaccsd(nroots=3, left=True)[1]
        e = myeom.eaccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.18931289565459147, 6)
        self.assertAlmostEqual(e[1], 0.28204643613789027, 6)
        self.assertAlmostEqual(e[2], 0.457836723621172  , 6)

    def test_df_eaccsd_koopmans(self):
        e,v = mycc.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19038860392603385, 6)
        self.assertAlmostEqual(e[1], 0.28339727115722535, 6)
        self.assertAlmostEqual(e[2], 1.0215547528836946 , 6)

        e,v = mycc.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.19038860392603385, 6)
        self.assertAlmostEqual(e[1], 0.28339727115722535, 6)
        self.assertAlmostEqual(e[2], 1.0215547528836946 , 6)

    def test_df_eaccsd_partition(self):
        e,v = mycc.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.19324341795558322, 6)
        self.assertAlmostEqual(e[1], 0.28716776030933833, 6)
        self.assertAlmostEqual(e[2], 0.90836050326011419, 6)

        e,v = mycc.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.18750981070399036, 6)
        self.assertAlmostEqual(e[1], 0.27959207345640869, 6)
        self.assertAlmostEqual(e[2], 0.57042043243953111, 6)


    def test_df_eeccsd(self):
        e,v = mycc.eeccsd(nroots=1)
        self.assertAlmostEqual(e, 0.28107576276117063, 6)

        e,v = mycc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[1], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[2], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[3], 0.30810935900155312, 6)

    def test_df_eeccsd_koopmans(self):
        e,v = mycc.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[1], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[2], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[3], 0.30810935900155312, 6)

        e,v = mycc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[1], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[2], 0.28107576276117063, 6)
        self.assertAlmostEqual(e[3], 0.30810935900155312, 6)

    def test_df_eomee_ccsd_matvec_singlet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEESinglet(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1), -29298.85295742222, 7)
        self.assertAlmostEqual(lib.fp(r2), 10145.543907318603, 7)

    def test_df_eomee_ccsd_matvec_triplet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        r2[0] = r2[0] - r2[0].transpose(0,1,3,2)
        r2[0] = r2[0] - r2[0].transpose(1,0,2,3)
        r2[1] = r2[1] - r2[1].transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEETriplet(mycc1)
        vec = myeom.amplitudes_to_vector(r1, r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), -6923.975705203359, 8)
        self.assertAlmostEqual(lib.fp(r2[0]), 37033.47500554715 , 7)
        self.assertAlmostEqual(lib.fp(r2[1]), 4164.221159944544, 7)

    def test_df_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), 133.93692998558186, 8)
        self.assertAlmostEqual(lib.fp(r2[0]), 15572.228624905263, 7)
        self.assertAlmostEqual(lib.fp(r2[1]),-12949.86945323115 , 7)

    def test_df_eomee_diag(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc1).get_diag()
        self.assertAlmostEqual(lib.fp(vec1S), 213.17116636400607, 9)
        self.assertAlmostEqual(lib.fp(vec1T),-857.2458742624522 , 9)
        self.assertAlmostEqual(lib.fp(vec2) , 14.357453812621733, 9)

    def test_ao2mo(self):
        numpy.random.seed(2)
        mo = numpy.random.random(mf.mo_coeff.shape)
        mycc = cc.CCSD(mf).density_fit(auxbasis='ccpvdz-ri')
        mycc.max_memory = 0
        eri_df = mycc.ao2mo(mo)
        self.assertAlmostEqual(lib.fp(eri_df.oooo), -493.98003157749906, 9)
        self.assertAlmostEqual(lib.fp(eri_df.oovv), -91.84858398271658 , 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovoo), -203.89515661847437, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvo), -14.883877359169205, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovov), -57.62195194777554 , 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvv), -24.359418953533535, 9)
        self.assertTrue(eri_df.vvvv is None)
        self.assertAlmostEqual(lib.fp(eri_df.vvL),  -0.5165177516806061, 9)


if __name__ == "__main__":
    print("Full Tests for DFCCSD")
    unittest.main()
