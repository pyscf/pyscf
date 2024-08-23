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

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import gccsd, eom_gccsd, gintermediates

def make_mycc1():
    mol = gto.M()
    nocc, nvir = 8, 14
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.GHF(mol)
    numpy.random.seed(12)
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = numpy.random.random((nmo,nmo))
    mf.mo_energy = numpy.arange(0., nmo)
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 1
    vhf = numpy.random.random((nmo,nmo)) + numpy.random.random((nmo,nmo))+1j
    vhf = vhf + vhf.conj().T
    mf.get_veff = lambda *args: vhf
    cinv = numpy.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(numpy.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    nmo_pair = nmo*(nmo//2+1)//4
    mf._eri = numpy.random.random(nmo_pair*(nmo_pair+1)//2)
    mycc1 = gccsd.GCCSD(mf)
    eris1 = mycc1.ao2mo()
    eris1.oooo = eris1.oooo + numpy.sin(eris1.oooo)*1j
    eris1.oooo = eris1.oooo + eris1.oooo.conj().transpose(2,3,0,1)
    eris1.ooov = eris1.ooov + numpy.sin(eris1.ooov)*1j
    eris1.oovv = eris1.oovv + numpy.sin(eris1.oovv)*1j
    eris1.ovov = eris1.ovov + numpy.sin(eris1.ovov)*1j
    eris1.ovvv = eris1.ovvv + numpy.sin(eris1.ovvv)*1j
    eris1.vvvv = eris1.vvvv + numpy.sin(eris1.vvvv)*1j
    eris1.vvvv = eris1.vvvv + eris1.vvvv.conj().transpose(2,3,0,1)
    a = numpy.random.random((nmo,nmo)) * .1
    eris1.fock += a + a.T
    t1 = numpy.random.random((nocc,nvir))*.1 + numpy.random.random((nocc,nvir))*.1j
    t2 = (numpy.random.random((nocc,nocc,nvir,nvir))*.1 +
          numpy.random.random((nocc,nocc,nvir,nvir))*.1j)
    t2 = t2 - t2.transpose(0,1,3,2)
    t2 = t2 - t2.transpose(1,0,2,3)
    mycc1.t1 = t1
    mycc1.t2 = t2
    return mycc1, eris1

def setUpModule():
    global mol, mf, mycc, eris1, mycc1, nocc, nvir
    mol = gto.Mole()
    mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '6-31g'
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = cc.GCCSD(mf).run()
    mycc1, eris1 = make_mycc1()
    nocc, nvir = mycc1.t1.shape

def tearDownModule():
    global mol, mf, mycc, eris1, mycc1
    mol.stdout.close()
    del mol, mf, mycc, eris1, mycc1

class KnownValues(unittest.TestCase):
    def test_ipccsd(self):
        e,v = mycc.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.42789089871467728, 5)

        myeom = eom_gccsd.EOMIP(mycc)
        e,v = myeom.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[1], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[2], 0.50226873136932748, 5)

        e,lv = myeom.ipccsd(nroots=3, left=True)
        self.assertAlmostEqual(e[0], 0.4278908208680458, 5)
        self.assertAlmostEqual(e[1], 0.4278908208680482, 5)
        self.assertAlmostEqual(e[2], 0.5022686041399118, 5)

        # Sometimes these tests can fail due to left and right evecs
        # having small (~zero) overlap, causing numerical error. FIXME
        e = myeom.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.4358615224789573, 5)
        self.assertAlmostEqual(e[1], 0.4358615224789594, 5)
        #self.assertAlmostEqual(e[2], 0.5095767839056080, 5)


    def test_ipccsd_koopmans(self):
        e,v = mycc.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[1], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[2], 0.50226873136932748, 5)

        e,v = mycc.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[1], 0.42789089871467728, 5)
        self.assertAlmostEqual(e[2], 0.50226873136932748, 5)


    def test_eaccsd(self):
        e,v = mycc.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.19050592141957523, 5)

        myeom = eom_gccsd.EOMEA(mycc)
        e,v = myeom.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[1], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[2], 0.28345228596676159, 5)

        e,lv = myeom.eaccsd(nroots=3, left=True)
        self.assertAlmostEqual(e[0], 0.1905059282334537, 5)
        self.assertAlmostEqual(e[1], 0.1905059282334538, 5)
        self.assertAlmostEqual(e[2], 0.2834522921515028, 5)

        e = myeom.eaccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.1894169322207168, 5)
        self.assertAlmostEqual(e[1], 0.1894169322207168, 5)
        self.assertAlmostEqual(e[2], 0.2820757599337823, 5)

    def test_eaccsd_koopmans(self):
        e,v = mycc.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[1], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[2], 0.28345228596676159, 5)

        e,v = mycc.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[1], 0.19050592141957523, 5)
        self.assertAlmostEqual(e[2], 0.28345228596676159, 5)


    def test_eeccsd_high_cost(self):
        e,v = mycc.eeccsd(nroots=2)
        self.assertAlmostEqual(e[0], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[1], 0.28114507364237717, 5)

        myeom = eom_gccsd.EOMEE(mycc)
        e,v = myeom.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[1], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[2], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[3], 0.30819729785603989, 5)

    def test_eeccsd_koopmans_high_cost(self):
        e,v = mycc.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[1], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[2], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[3], 0.30819729785603989, 5)

        e,v = mycc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[1], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[2], 0.28114507364237717, 5)
        self.assertAlmostEqual(e[3], 0.30819729785603989, 5)


    def test_vector_to_amplitudes(self):
        r1, r2 = mycc1.vector_to_amplitudes(mycc1.amplitudes_to_vector(mycc1.t1, mycc1.t2))
        self.assertAlmostEqual(abs(mycc1.t1-r1).max(), 0, 14)
        self.assertAlmostEqual(abs(mycc1.t2-r2).max(), 0, 14)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.GHF(mol).apply(cc.GCCSD)
        nelec = (3,3)
        nocc, nvir = nelec[0]*2, 4
        nmo = nocc + nvir
        mycc.nocc = nocc
        mycc.nmo = nmo
        def check_overwritten(method):
            vec = numpy.zeros(method.vector_size())
            vec_orig = vec.copy()
            t1, t2 = method.vector_to_amplitudes(vec)
            t1[:] = 1
            t2[:] = 1
            self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

        check_overwritten(mycc)
        check_overwritten(mycc.EOMIP())
        check_overwritten(mycc.EOMEA())
        check_overwritten(mycc.EOMEE())

    def test_ip_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nocc))-.9 + numpy.random.random((nocc))*.2j
        r2 = (numpy.random.random((nocc,nocc,nvir))-.9 +
              numpy.random.random((nocc,nocc,nvir))*.2j)
        myeom = eom_gccsd.EOMIP(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        self.assertAlmostEqual(lib.fp(r1), -0.27326789353857561+0.066555317833114358j, 12)
        self.assertAlmostEqual(lib.fp(r2), -4.6531653660668413+2.8855141740871528j, 12)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -3582.8270590959773+999.42924221648195j, 9)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 430.02008558030951+815.74607152866815j, 9)

    def test_ea_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nvir))-.9 + numpy.random.random((nvir))*.2j
        r2 = (numpy.random.random((nocc,nvir,nvir))-.9 +
              numpy.random.random((nocc,nvir,nvir))*.2j)
        myeom = eom_gccsd.EOMEA(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        self.assertAlmostEqual(lib.fp(r1), 0.92358852093801858+0.12450821337841513j, 12)
        self.assertAlmostEqual(lib.fp(r2), 12.871866219332823+2.72978281362834j, 12)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), 9955.7506642715725+1625.6301340218238j, 9)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), -23.784806916968591-975.86985392855195j, 9)

    def test_eomee_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((nocc,nvir)) - .9
        r2 = numpy.random.random((nocc,nocc,nvir,nvir)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_gccsd.EOMEE(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1), -6433.8165603568596-630.53684527676432j, 9)
        self.assertAlmostEqual(lib.fp(r2), 58591.683282553095+31543.960209750952j, 7)

    def test_eomee_diag(self):
        vec = eom_gccsd.EOMEE(mycc1).get_diag()
        self.assertAlmostEqual(lib.fp(vec), 1853.7201843910152+4488.8163311564713j, 9)

    def test_t3p2_intermediates_complex(self):
        '''Although this has not been tested strictly for complex values, it
        was written to be correct for complex values and differences in the complex
        values between versions should be taken into account and corrected.'''
        myt1 = mycc1.t1.copy()
        myt2 = mycc1.t2.copy()
        e, pt1, pt2, Wmcik, Wacek = gintermediates.get_t3p2_imds_slow(mycc1, myt1, myt2, eris=eris1)
        self.assertAlmostEqual(lib.fp(e), -13810450.064880637, 3)
        self.assertAlmostEqual(lib.fp(pt1), (14525.466845738476+1145.7490899866602j), 4)
        self.assertAlmostEqual(lib.fp(pt2), (-34943.95360794282+29728.34747703709j), 4)
        self.assertAlmostEqual(lib.fp(Wmcik), (271010.439304044-201703.96483952703j), 4)
        self.assertAlmostEqual(lib.fp(Wacek), (316710.3913587458+99554.48507036189j), 4)

    def test_t3p2_intermediates_real(self):
        myt1 = mycc1.t1.real.copy()
        myt2 = mycc1.t2.real.copy()
        new_eris = mycc1.ao2mo()
        new_eris.oooo = eris1.oooo.real
        new_eris.oooo = eris1.oooo.real
        new_eris.ooov = eris1.ooov.real
        new_eris.oovv = eris1.oovv.real
        new_eris.ovov = eris1.ovov.real
        new_eris.ovvv = eris1.ovvv.real
        new_eris.vvvv = eris1.vvvv.real
        new_eris.vvvv = eris1.vvvv.real
        new_eris.fock = eris1.fock.real
        new_eris.mo_energy = new_eris.fock.diagonal()
        e, pt1, pt2, Wmcik, Wacek = gintermediates.get_t3p2_imds_slow(mycc1, myt1, myt2, eris=new_eris)
        self.assertAlmostEqual(lib.fp(e), 8588977.565611511, 3)
        self.assertAlmostEqual(lib.fp(pt1), 14407.896043949795, 5)
        self.assertAlmostEqual(lib.fp(pt2), -34967.36031768824, 5)
        self.assertAlmostEqual(lib.fp(Wmcik), 271029.6000933048, 5)
        self.assertAlmostEqual(lib.fp(Wacek), 316110.05463216535, 5)

    def test_h2o_star(self):
        mol_h2o = gto.Mole()
        mol_h2o.atom = [
                [8, [0.000000000000000, -0.000000000000000, -0.124143731294022]],
                [1, [0.000000000000000, -1.430522735894536,  0.985125550040314]],
                [1, [0.000000000000000,  1.430522735894536,  0.985125550040314]]]
        mol_h2o.unit = 'B'
        mol_h2o.basis = {'H' : [[0,
                               [5.4471780, 0.156285],
                               [0.8245472, 0.904691]],
                               [0, [0.1831916, 1.0]]],
                        'O' : '3-21G'}
        mol_h2o.verbose = 9
        mol_h2o.output = '/dev/null'
        mol_h2o.build()
        mf_h2o = scf.RHF(mol_h2o)
        mf_h2o.conv_tol_grad = 1e-12
        mf_h2o.conv_tol = 1e-12
        mf_h2o.kernel()
        mycc_h2o = cc.GCCSD(mf_h2o).run()
        mycc_h2o.conv_tol_normt = 1e-12
        mycc_h2o.conv_tol = 1e-12
        mycc_h2o.kernel()

        myeom = eom_gccsd.EOMIP(mycc_h2o)
        e = myeom.ipccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.410661965883, 5)

        myeom = eom_gccsd.EOMIP_Ta(mycc_h2o)
        e = myeom.ipccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.411695647736, 5)

        myeom = eom_gccsd.EOMEA(mycc_h2o)
        e = myeom.eaccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.250589854185, 5)

        myeom = eom_gccsd.EOMEA_Ta(mycc_h2o)
        e = myeom.eaccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.250720295150, 5)

if __name__ == "__main__":
    print("Tests for EOM GCCSD")
    unittest.main()
