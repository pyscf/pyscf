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
import copy
import numpy
from functools import reduce

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf import ao2mo
from pyscf.cc import ccsd, rccsd, eom_rccsd, rintermediates, gintermediates

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

    mycc1 = rccsd.RCCSD(mf1)
    eris1 = mycc1.ao2mo()
    numpy.random.seed(12)
    r1 = numpy.random.random((no,nv)) - .9
    r2 = numpy.random.random((no,no,nv,nv)) - .9
    r2 = r2 + r2.transpose(1,0,3,2)
    mycc1.t1 = r1*1e-5
    mycc1.t2 = r2*1e-5
    return mf1, mycc1, eris1

def setUpModule():
    global mol, mf, mycc, mf1, eris1, mycc1, mycci, erisi, mycc2, mycc21, eris21, mycc3, mycc31, eris31, no, nv
    mol = gto.Mole()
    mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = rccsd.RCCSD(mf).run()

    mf1, mycc1, eris1 = make_mycc1()
    no, nv = mycc1.t1.shape

    mycci = mycc1.copy()
    erisi = copy.copy(eris1)
    erisi.oooo = eris1.oooo + numpy.sin(eris1.oooo)*1j
    erisi.oooo = erisi.oooo + erisi.oooo.conj().transpose(1,0,3,2)
    erisi.ovoo = eris1.ovoo + numpy.sin(eris1.ovoo)*1j
    erisi.ovvo = eris1.ovvo + numpy.sin(eris1.ovvo)*1j
    erisi.oovv = eris1.oovv + numpy.sin(eris1.oovv)*1j
    erisi.oovv = erisi.oovv + erisi.oovv.conj().transpose(1,0,3,2)
    erisi.ovov = eris1.ovov + numpy.sin(eris1.ovov)*1j
    erisi.ovvv = eris1.ovvv + numpy.sin(eris1.ovvv)*1j
    erisi.vvvv = eris1.vvvv + numpy.sin(eris1.vvvv)*1j
    erisi.vvvv = erisi.vvvv + erisi.vvvv.conj().transpose(1,0,3,2)

    mycc2 = ccsd.CCSD(mf)
    mycc21 = ccsd.CCSD(mf1)
    mycc2.__dict__.update(mycc.__dict__)
    mycc21.__dict__.update(mycc1.__dict__)
    eris21 = mycc21.ao2mo()

    mycc3 = ccsd.CCSD(mf)
    mycc31 = ccsd.CCSD(mf1)
    mycc3.__dict__.update(mycc.__dict__)
    mycc31.__dict__.update(mycc1.__dict__)
    mycc3 = mycc3.set(max_memory=0, direct=True)
    mycc31 = mycc31.set(max_memory=0, direct=True)
    eris31 = mycc31.ao2mo()


def tearDownModule():
    global mol, mf, mycc, mf1, eris1, mycc1, mycci, erisi, mycc2, mycc21, eris21, mycc3, mycc31, eris31
    del mol, mf, mycc, mf1, eris1, mycc1, mycci, erisi, mycc2, mycc21, eris21, mycc3, mycc31, eris31

class KnownValues(unittest.TestCase):
    def test_ipccsd(self):
        eom = mycc.eomip_method()
        e,v = eom.kernel(nroots=1, left=False, koopmans=False)
        e = eom.eip
        self.assertAlmostEqual(e, 0.4335604332073799, 5)

        e,v = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        myeom = eom_rccsd.EOMIP(mycc)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 5)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 5)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 5)

    def test_ipccsd_koopmans(self):
        e,v = mycc.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        e,v = mycc.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

    def test_ipccsd_partition(self):
        e,v = mycc.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 5)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 5)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 5)

        e,v = mycc.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 5)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 5)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 5)

        e,v = mycc.ipccsd(nroots=3, partition='mp', left=True)
        self.assertAlmostEqual(e[0], 0.42728862799879663, 5)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 5)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 5)

        e,v = mycc.ipccsd(nroots=3, partition='full', left=True)
        self.assertAlmostEqual(e[0], 0.42291981842588938, 5)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 5)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 5)


    def test_eaccsd(self):
        eom = mycc.eomea_method()
        e,v = eom.kernel(nroots=1, left=False, koopmans=False)
        e = eom.eea
        self.assertAlmostEqual(e, 0.16737886338859731, 5)

        e,v = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.51006797826488071, 5)

        myeom = eom_rccsd.EOMEA(mycc)
        lv = myeom.eaccsd(nroots=3, left=True)[1]
        e = myeom.eaccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.16656250872624662, 5)
        self.assertAlmostEqual(e[1], 0.2394414445283693 , 5)
        self.assertAlmostEqual(e[2], 0.41399434356202935, 5)

    def test_eaccsd_koopmans(self):
        e,v = mycc.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)

        e,v = mycc.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)

    def test_eaccsd_partition(self):
        e,v = mycc.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.16947311575051136, 5)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 5)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 5)

        e,v = mycc.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.16418276148493574, 5)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 5)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 5)

        e,v = mycc.eaccsd(nroots=3, partition='mp', left=True)
        self.assertAlmostEqual(e[0], 0.16947311575051136, 5)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 5)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 5)

        e,v = mycc.eaccsd(nroots=3, partition='full', left=True)
        self.assertAlmostEqual(e[0], 0.16418276148493574, 5)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 5)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 5)


    def test_eeccsd(self):
        eom = mycc.eomee_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        e = eom.eee
        self.assertAlmostEqual(e, 0.2757159395886167, 5)

        e,v = mycc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eeccsd_koopmans(self):
        e,v = mycc.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

        e,v = mycc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eomee_ccsd_singlet(self):
        e, v = mycc.eomee_ccsd_singlet(nroots=1)
        self.assertAlmostEqual(e, 0.3005716731825082, 5)

    def test_eomee_ccsd_triplet(self):
        e, v = mycc.eomee_ccsd_triplet(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 5)

    def test_eomsf_ccsd(self):
        e, v = mycc.eomsf_ccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 5)

    def test_vector_to_amplitudes(self):
        t1, t2 = mycc1.vector_to_amplitudes(mycc1.amplitudes_to_vector(mycc1.t1, mycc1.t2))
        self.assertAlmostEqual(abs(mycc1.t1-t1).sum(), 0, 9)
        self.assertAlmostEqual(abs(mycc1.t2-t2).sum(), 0, 9)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.RHF(mol).apply(cc.RCCSD)
        nelec = (3,3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        mycc.nocc = nocc
        mycc.nmo = nmo
        def check_overwritten(method):
            vec = numpy.zeros(method.vector_size())
            vec_orig = vec.copy()
            ts = method.vector_to_amplitudes(vec)
            for ti in ts:
                if isinstance(ti, numpy.ndarray):
                    ti[:] = 1
                else:
                    for t in ti:
                        t[:] = 1
            self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

        check_overwritten(mycc)
        check_overwritten(mycc.EOMIP())
        check_overwritten(mycc.EOMEA())
        check_overwritten(mycc.EOMEESinglet())
        check_overwritten(mycc.EOMEETriplet())

    def test_eomee_ccsd_matvec_singlet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEESinglet(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        vec1 = myeom.matvec(vec)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(v1), -145152.78511963107, 7)
        self.assertAlmostEqual(lib.fp(v2), -268196.0471308574, 7)

        gcc1 = cc.addons.convert_to_gccsd(mycc1)
        gee1 = gcc1.EOMEE()
        orbspin = gcc1._scf.mo_coeff.orbspin
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        self.assertAlmostEqual(abs(gr1-myeom.spatial2spin(v1, orbspin)).max(), 0, 8)
        self.assertAlmostEqual(abs(gr2-myeom.spatial2spin(v2, orbspin)).max(), 0, 8)

    def test_eomee_ccsd_matvec_triplet(self):
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
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(v1   ), -5616.12199728998, 8)
        self.assertAlmostEqual(lib.fp(v2[0]), -237430.14342594685,7)
        self.assertAlmostEqual(lib.fp(v2[1]), 127682.76151592708, 7)

        gcc1 = cc.addons.convert_to_uccsd(mycc1)
        gee1 = gcc1.EOMEESpinKeep()
        gr1 = (r1*.5**.5, -r1*.5**.5)
        raa = r2[0]*.5**.5
        rx = r2[1]*.5**.5
        gr2 = (raa, rx, -raa)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        self.assertAlmostEqual(abs(gr1[0]*2**.5- v1).max(), 0, 9)
        self.assertAlmostEqual(abs(gr2[0]*2**.5- v2[0]).max(), 0, 9)
        self.assertAlmostEqual(abs(gr2[1]*2**.5- v2[1]).max(), 0, 9)
        self.assertAlmostEqual(abs(-gr2[2]*2**.5- v2[0]).max(), 0, 9)

    def test_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        v1, v2 = myeom.vector_to_amplitudes(vec1)

        self.assertAlmostEqual(lib.fp(v1   ), -6213.095578988824, 7)
        self.assertAlmostEqual(lib.fp(v2[0]), 84329.40539995288 , 7)
        self.assertAlmostEqual(lib.fp(v2[1]), 6719.930458652292 , 7)

        numpy.random.seed(10)
        vec = numpy.random.random(myeom.vector_size()) - .9
        r1, r2 = myeom.vector_to_amplitudes(vec)
        vec1 = myeom.matvec(vec, imds)
        v1, v2 = myeom.vector_to_amplitudes(vec1)

        gcc1 = cc.addons.convert_to_gccsd(mycc1)
        gee1 = gcc1.EOMEE()
        orbspin = gcc1._scf.mo_coeff.orbspin
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        self.assertAlmostEqual(abs(gr1-myeom.spatial2spin(v1, orbspin)).max(), 0, 9)
        self.assertAlmostEqual(abs(gr2-myeom.spatial2spin(v2, orbspin)).max(), 0, 9)

    def test_eomee_diag(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc1).get_diag()
        self.assertAlmostEqual(lib.fp(vec1S), -4714.969920334639, 8)
        self.assertAlmostEqual(lib.fp(vec1T),  2221.322839866705, 8)
        self.assertAlmostEqual(lib.fp(vec2) , -5486.124838268124, 8)

    def test_ip_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = mycc1.EOMIP()
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)

        gcc1 = cc.addons.convert_to_gccsd(mycc1)
        gee1 = gcc1.EOMIP()
        orbspin = gcc1._scf.mo_coeff.orbspin
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        vec1 = myeom.matvec(vec)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(abs(gr1-myeom.spatial2spin(v1, orbspin)).max(), 0, 9)
        self.assertAlmostEqual(abs(gr2-myeom.spatial2spin(v2, orbspin)).max(), 0, 9)

        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.fp(r1), 0.37404344676857076, 11)
        self.assertAlmostEqual(lib.fp(r2), -1.1568913404570922, 11)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -14894.669606811192, 8)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 1182.3095479451745, 8)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris1)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.fp(vec1), -3795.9122245246967, 8)
        self.assertAlmostEqual(lib.fp(diag), 1106.260154202434, 8)

    def test_ea_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nv)) - .9
        r2 = numpy.random.random((no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEA(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)

        gcc1 = cc.addons.convert_to_gccsd(mycc1)
        gee1 = gcc1.EOMEA()
        orbspin = gcc1._scf.mo_coeff.orbspin
        gr1 = myeom.spatial2spin(r1, orbspin)
        gr2 = myeom.spatial2spin(r2, orbspin)
        gvec = gee1.amplitudes_to_vector(gr1, gr2)
        vecref = gee1.matvec(gvec)
        gr1, gr2 = gee1.vector_to_amplitudes(vecref)
        vec1 = myeom.matvec(vec)
        v1, v2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(abs(gr1-myeom.spatial2spin(v1, orbspin)).max(), 0, 9)
        self.assertAlmostEqual(abs(gr2-myeom.spatial2spin(v2, orbspin)).max(), 0, 9)

        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.fp(r1), 1.4488291275539353, 11)
        self.assertAlmostEqual(lib.fp(r2), 0.97080165032287469, 11)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -34426.363943760276, 8)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 2724.8239646679217, 8)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris1)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.fp(vec1), -17030.363405297598, 8)
        self.assertAlmostEqual(lib.fp(diag), 4688.9122122011922, 8)


########################################
# Complex integrals
    def test_ip_matvec1(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no))-.9 + numpy.random.random((no))*.2j
        r2 = (numpy.random.random((no,no,nv))-.9 +
              numpy.random.random((no,no,nv))*.2j)
        myeom = eom_rccsd.EOMIP(mycci)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        eris1
        imds = myeom.make_imds(erisi)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), 25176.428829164193-4955.5351324520125j, 7)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 1106.2601542024306, 8)

    def test_ea_matvec1(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nv))-.9 + numpy.random.random((nv))*.2j
        r2 = (numpy.random.random((no,nv,nv))-.9 +
              numpy.random.random((no,nv,nv))*.2j)
        myeom = eom_rccsd.EOMEA(mycci)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        imds = myeom.make_imds(erisi)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -105083.60825558871+25155.909195554908j, 6)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 4688.9122122011895, 8)


########################################
# With 4-fold symmetry in integrals
    def test_ipccsd2(self):
        e,v = mycc2.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 5)

        e,v = mycc2.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        myeom = eom_rccsd.EOMIP(mycc2)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 5)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 5)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 5)

    def test_ipccsd_koopmans2(self):
        e,v = mycc2.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        e,v = mycc2.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

    def test_ipccsd_partition2(self):
        e,v = mycc2.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 5)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 5)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 5)

        e,v = mycc2.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 5)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 5)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 5)


    def test_eaccsd2(self):
        e,v = mycc2.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.16737886338859731, 5)

        e,v = mycc2.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.51006797826488071, 5)

        myeom = eom_rccsd.EOMEA(mycc2)
        lv = myeom.eaccsd(nroots=3, left=True)[1]
        e = myeom.eaccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.16656250872624662, 5)
        self.assertAlmostEqual(e[1], 0.2394414445283693, 5)
        self.assertAlmostEqual(e[2], 0.41399434356202935, 5)

    def test_eaccsd_koopmans2(self):
        e,v = mycc2.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)

        e,v = mycc2.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)

    def test_eaccsd_partition2(self):
        e,v = mycc2.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.16947311575051136, 5)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 5)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 5)

        e,v = mycc2.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.16418276148493574, 5)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 5)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 5)


    def test_eeccsd2(self):
        e,v = mycc2.eeccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 5)

        e,v = mycc2.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eeccsd_koopmans2(self):
        e,v = mycc2.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

        e,v = mycc2.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eomee_ccsd_matvec_singlet2(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEESinglet(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1), -145152.7851196310, 7)
        self.assertAlmostEqual(lib.fp(r2), -268196.0471308578, 7)

    def test_eomee_ccsd_matvec_triplet2(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        r2[0] = r2[0] - r2[0].transpose(0,1,3,2)
        r2[0] = r2[0] - r2[0].transpose(1,0,2,3)
        r2[1] = r2[1] - r2[1].transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEETriplet(mycc21)
        vec = myeom.amplitudes_to_vector(r1, r2)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), -5616.12199728998, 8)
        self.assertAlmostEqual(lib.fp(r2[0]), -237430.14342594685,7)
        self.assertAlmostEqual(lib.fp(r2[1]), 127682.76151592708, 7)

    def test_eomsf_ccsd_matvec2(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), -6213.095578988824, 7)
        self.assertAlmostEqual(lib.fp(r2[0]), 84329.40539995288 , 7)
        self.assertAlmostEqual(lib.fp(r2[1]), 6719.930458652292 , 7)

    def test_eomee_diag2(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc21).get_diag()
        self.assertAlmostEqual(lib.fp(vec1S), -4714.969920334639, 8)
        self.assertAlmostEqual(lib.fp(vec1T),  2221.322839866705, 8)
        self.assertAlmostEqual(lib.fp(vec2) , -5486.124838268124, 8)


    def test_ip_matvec2(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = eom_rccsd.EOMIP(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.fp(r1), 0.37404344676857076, 11)
        self.assertAlmostEqual(lib.fp(r2), -1.1568913404570922, 11)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -14894.669606811192, 8)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 1182.3095479451745, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris21)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.fp(vec1), -3795.9122245246967, 8)
        self.assertAlmostEqual(lib.fp(diag), 1106.260154202434, 9)

    def test_ea_matvec2(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nv)) - .9
        r2 = numpy.random.random((no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEA(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.fp(r1), 1.4488291275539353, 11)
        self.assertAlmostEqual(lib.fp(r2), 0.97080165032287469, 11)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -34426.363943760276, 8)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 2724.8239646679217, 8)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris21)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.fp(vec1), -17030.363405297598, 8)
        self.assertAlmostEqual(lib.fp(diag), 4688.9122122011922, 8)


########################################
# With 4-fold symmetry in integrals
# max_memory = 0
# direct = True
    def test_ipccsd3(self):
        e,v = mycc3.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 5)

        e,v = mycc3.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        myeom = eom_rccsd.EOMIP(mycc3)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 5)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 5)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 5)

    def test_ipccsd_koopmans3(self):
        e,v = mycc3.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

        e,v = mycc3.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 5)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 5)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 5)

    def test_ipccsd_partition3(self):
        e,v = mycc3.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 5)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 5)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 5)

        e,v = mycc3.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 5)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 5)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 5)


#    def test_eaccsd3(self):
#        e,v = mycc3.eaccsd(nroots=1)
#        self.assertAlmostEqual(e, 0.16737886338859731, 5)
#
#        e,v = mycc3.eaccsd(nroots=3)
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
#        self.assertAlmostEqual(e[2], 0.51006797826488071, 5)
#
#        myeom = eom_rccsd.EOMEA(mycc3)
#        lv = myeom.eaccsd(nroots=3, left=True)[1]
#        e = myeom.eaccsd_star_contract(e, v, lv)
#        self.assertAlmostEqual(e[0], 0.16656250872624662, 5)
#        self.assertAlmostEqual(e[1], 0.2394414445283693, 5)
#        self.assertAlmostEqual(e[2], 0.41399434356202935, 5)
#
#    def test_eaccsd_koopmans3(self):
#        e,v = mycc3.eaccsd(nroots=3, koopmans=True)
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
#        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)
#
#        e,v = mycc3.eaccsd(nroots=3, guess=v[:3])
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 5)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 5)
#        self.assertAlmostEqual(e[2], 0.73443352557582653, 5)
#
#    def test_eaccsd_partition3(self):
#        e,v = mycc3.eaccsd(nroots=3, partition='mp')
#        self.assertAlmostEqual(e[0], 0.16947311575051136, 5)
#        self.assertAlmostEqual(e[1], 0.24234326468848749, 5)
#        self.assertAlmostEqual(e[2], 0.7434661346653969 , 5)
#
#        e,v = mycc3.eaccsd(nroots=3, partition='full')
#        self.assertAlmostEqual(e[0], 0.16418276148493574, 5)
#        self.assertAlmostEqual(e[1], 0.23683978491376495, 5)
#        self.assertAlmostEqual(e[2], 0.55640091560545624, 5)


    def test_eeccsd3(self):
        e,v = mycc3.eeccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 5)

        e,v = mycc3.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eeccsd_koopmans3(self):
        e,v = mycc3.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

        e,v = mycc3.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 5)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 5)

    def test_eomee_ccsd_matvec_singlet3(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEESinglet(mycc31)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1), -145152.7851196310, 7)
        self.assertAlmostEqual(lib.fp(r2), -268196.0471308578, 7)

    def test_eomee_ccsd_matvec_triplet3(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        r2[0] = r2[0] - r2[0].transpose(0,1,3,2)
        r2[0] = r2[0] - r2[0].transpose(1,0,2,3)
        r2[1] = r2[1] - r2[1].transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEETriplet(mycc31)
        vec = myeom.amplitudes_to_vector(r1, r2)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), -5616.12199728998, 8)
        self.assertAlmostEqual(lib.fp(r2[0]), -237430.14342594685,7)
        self.assertAlmostEqual(lib.fp(r2[1]), 127682.76151592708, 7)

    def test_eomsf_ccsd_matvec3(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc31)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.fp(r1   ), -6213.095578988824, 7)
        self.assertAlmostEqual(lib.fp(r2[0]), 84329.40539995288 , 7)
        self.assertAlmostEqual(lib.fp(r2[1]), 6719.930458652292 , 7)

    def test_eomee_diag3(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc31).get_diag()
        self.assertAlmostEqual(lib.fp(vec1S), -2881.664963714903, 8)
        self.assertAlmostEqual(lib.fp(vec1T),  2039.745909568253, 8)
        self.assertAlmostEqual(lib.fp(vec2) , -4271.586697637197, 8)


    def test_ip_matvec3(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = eom_rccsd.EOMIP(mycc31)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.fp(r1), 0.37404344676857076, 11)
        self.assertAlmostEqual(lib.fp(r2), -1.1568913404570922, 11)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.fp(vec1), -14894.669606811192, 8)
        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 1182.3095479451745, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris31)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.fp(vec1), -3795.9122245246967, 9)
        self.assertAlmostEqual(lib.fp(diag), 1106.260154202434, 9)

    def test_sort_left_right_eigensystem(self):
        myeom = eom_rccsd.EOMIP(mycc)
        right_evecs = [numpy.ones(10)] * 4
        left_evecs = [numpy.ones(10)] * 5
        right_evecs = [x*i for i, x in enumerate(right_evecs)]
        left_evecs = [x*i for i, x in enumerate(left_evecs)]
        revals, revecs, levecs = eom_rccsd._sort_left_right_eigensystem(
            myeom,
            [True, False, True, True], [-1.1, 0, 1.1, 2.2], right_evecs,
            [True, True, True, False, True], [-2.2, -1.1, 0, 1.1, 2.2], left_evecs)
        self.assertEqual(revals[0], -1.1)
        self.assertEqual(revals[1], 2.2)
        self.assertEqual(revecs[0][0], 0)
        self.assertEqual(revecs[1][0], 3)
        self.assertEqual(levecs[0][0], 1)
        self.assertEqual(levecs[1][0], 4)

        revals, revecs, levecs = eom_rccsd._sort_left_right_eigensystem(
            myeom,
            [True, False, True, True], [-1.1, 0, 1.1, 2.2], right_evecs,
            [True, True, False, True, True], [-2.2, -1.1, 0, 1.1, 2.2], left_evecs)
        self.assertEqual(revals[0], -1.1)
        self.assertEqual(revals[1], 1.1)
        self.assertEqual(revals[2], 2.2)
        self.assertEqual(revecs[0][0], 0)
        self.assertEqual(revecs[1][0], 2)
        self.assertEqual(revecs[2][0], 3)
        self.assertEqual(levecs[0][0], 1)
        self.assertEqual(levecs[1][0], 3)
        self.assertEqual(levecs[2][0], 4)

#    def test_ea_matvec3(self):
#        numpy.random.seed(12)
#        r1 = numpy.random.random((nv)) - .9
#        r2 = numpy.random.random((no,nv,nv)) - .9
#        myeom = eom_rccsd.EOMEA(mycc31)
#        vec = myeom.amplitudes_to_vector(r1,r2)
#        r1,r2 = myeom.vector_to_amplitudes(vec)
#        myeom.partition = 'mp'
#        self.assertAlmostEqual(lib.fp(r1), 1.4488291275539353, 12)
#        self.assertAlmostEqual(lib.fp(r2), 0.97080165032287469, 12)
#        imds = myeom.make_imds(eris31)
#        vec1 = myeom.matvec(vec, imds)
#        self.assertAlmostEqual(lib.fp(vec1), -34426.363943760276, 9)
#        self.assertAlmostEqual(lib.fp(myeom.get_diag()), 2724.8239646679217, 9)
#
#        myeom.partition = 'full'
#        imds = myeom.make_imds(eris31)
#        diag = myeom.get_diag(imds)
#        vec1 = myeom.matvec(vec, imds, diag=diag)
#        self.assertAlmostEqual(lib.fp(vec1), -17030.363405297598, 9)
#        self.assertAlmostEqual(lib.fp(diag), 4688.9122122011922, 9)

    def test_t3p2_intermediates_complex(self):
        '''Although this has not been tested strictly for complex values, it
        was written to be correct for complex values and differences in the complex
        values between versions should be taken into account and corrected.'''
        myt1 = mycc1.t1 + 1j * numpy.sin(mycc1.t1) * mycc1.t1
        myt2 = mycc1.t2 + 1j * numpy.sin(mycc1.t2) * mycc1.t2
        myt2 = myt2 + myt2.transpose(1,0,3,2)
        e, pt1, pt2, Wmcik, Wacek = rintermediates.get_t3p2_imds_slow(mycc1, myt1, myt2, eris=erisi)
        self.assertAlmostEqual(lib.fp(e), 23223.465490572264, 5)
        self.assertAlmostEqual(lib.fp(pt1), (-5.2202836452466705-0.09570164571057749j), 5)
        self.assertAlmostEqual(lib.fp(pt2), (46.188012063609506-1.303867687778909j), 5)
        self.assertAlmostEqual(lib.fp(Wmcik), (-18.438930654297778+1.5734161307568773j), 5)
        self.assertAlmostEqual(lib.fp(Wacek), (-7.187576764072701+0.7399185332889747j), 5)

    def test_t3p2_intermediates_real(self):
        myt1 = mycc1.t1.copy()
        myt2 = mycc1.t2.copy()
        myt2 = myt2 + myt2.transpose(1,0,3,2)
        e, pt1, pt2, Wmcik, Wacek = rintermediates.get_t3p2_imds_slow(mycc1, myt1, myt2)
        self.assertAlmostEqual(lib.fp(e), 23230.479350851536, 5)
        self.assertAlmostEqual(lib.fp(pt1), -5.218888542335442, 5)
        self.assertAlmostEqual(lib.fp(pt2), 46.19512409958347, 5)
        self.assertAlmostEqual(lib.fp(Wmcik), -18.47928005593598, 5)
        self.assertAlmostEqual(lib.fp(Wacek), -7.101360230151883, 5)

    def test_t3p2_intermediates_against_so(self):
        from pyscf.cc.addons import convert_to_gccsd
        myt1 = mycc1.t1.copy()
        myt2 = mycc1.t2.copy()
        e, pt1, pt2, Wmcik, Wacek = rintermediates.get_t3p2_imds_slow(mycc1, myt1, myt2)

        mygcc = convert_to_gccsd(mycc1)
        mygt1 = mygcc.t1.copy()
        mygt2 = mygcc.t2.copy()
        ge, gpt1, gpt2, gWmcik, gWacek = gintermediates.get_t3p2_imds_slow(mygcc, mygt1, mygt2)
        self.assertAlmostEqual(lib.fp(pt1), -2.6094405706617727, 5)
        self.assertAlmostEqual(lib.fp(pt2), 23.097562049844235, 5)
        self.assertAlmostEqual(lib.fp(pt1), lib.fp(gpt1[::2,::2]), 5)
        self.assertAlmostEqual(lib.fp(pt2), lib.fp(gpt2[::2,1::2,::2,1::2]), 5)

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
        mol_h2o.verbose = 7
        mol_h2o.output = '/dev/null'
        mol_h2o.build()
        mol.conv_tol = 1e-12
        mf_h2o = scf.RHF(mol_h2o)
        mf_h2o.conv_tol_grad = 1e-12
        mf_h2o.kernel()
        mycc_h2o = cc.RCCSD(mf_h2o).run()
        mycc_h2o.conv_tol_normt = 1e-12
        mycc_h2o.conv_tol = 1e-12
        mycc_h2o.kernel()

        myeom = eom_rccsd.EOMIP(mycc_h2o)
        e = myeom.ipccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.410661965883, 5)

        myeom = eom_rccsd.EOMIP_Ta(mycc_h2o)
        e = myeom.ipccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.411695647736, 5)

        myeom = eom_rccsd.EOMEA(mycc_h2o)
        e = myeom.eaccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.250589854185, 5)

        myeom = eom_rccsd.EOMEA_Ta(mycc_h2o)
        e = myeom.eaccsd_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.250720295150, 5)

if __name__ == "__main__":
    print("Tests for EOM RCCSD")
    unittest.main()
