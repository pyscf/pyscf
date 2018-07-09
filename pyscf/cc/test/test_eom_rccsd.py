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
from pyscf.cc import ccsd, rccsd, eom_rccsd

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

def make_mycc1():
    mf1 = copy.copy(mf)
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

mf1, mycc1, eris1 = make_mycc1()
no, nv = mycc1.t1.shape

mycci = copy.copy(mycc1)
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
        self.assertAlmostEqual(e, 0.4335604332073799, 6)

        e,v = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        myeom = eom_rccsd.EOMIP(mycc)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 6)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 6)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 6)

    def test_ipccsd_koopmans(self):
        e,v = mycc.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        e,v = mycc.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

    def test_ipccsd_partition(self):
        e,v = mycc.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 6)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 6)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 6)

        e,v = mycc.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 6)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 6)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 6)

        e,v = mycc.ipccsd(nroots=3, partition='mp', left=True)
        self.assertAlmostEqual(e[0], 0.42728862799879663, 6)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 6)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 6)

        e,v = mycc.ipccsd(nroots=3, partition='full', left=True)
        self.assertAlmostEqual(e[0], 0.42291981842588938, 6)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 6)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 6)


    def test_eaccsd(self):
        eom = mycc.eomea_method()
        e,v = eom.kernel(nroots=1, left=False, koopmans=False)
        e = eom.eea
        self.assertAlmostEqual(e, 0.16737886338859731, 6)

        e,v = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.51006797826488071, 6)

        myeom = eom_rccsd.EOMEA(mycc)
        lv = myeom.eaccsd(nroots=3, left=True)[1]
        e = myeom.eaccsd_star(e, v, lv)
        self.assertAlmostEqual(e[0], 0.16656250872624662, 6)
        self.assertAlmostEqual(e[1], 0.2394414445283693, 6)
        self.assertAlmostEqual(e[2], 0.41399434356202935, 6)

    def test_eaccsd_koopmans(self):
        e,v = mycc.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)

        e,v = mycc.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)

    def test_eaccsd_partition(self):
        e,v = mycc.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.16947311575051136, 6)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 6)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 6)

        e,v = mycc.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.16418276148493574, 6)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 6)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 6)

        e,v = mycc.eaccsd(nroots=3, partition='mp', left=True)
        self.assertAlmostEqual(e[0], 0.16947311575051136, 6)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 6)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 6)

        e,v = mycc.eaccsd(nroots=3, partition='full', left=True)
        self.assertAlmostEqual(e[0], 0.16418276148493574, 6)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 6)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 6)


    def test_eeccsd(self):
        eom = mycc.eomee_method()
        e,v = eom.kernel(nroots=1, koopmans=False)
        e = eom.eee
        self.assertAlmostEqual(e, 0.2757159395886167, 6)

        e,v = mycc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_eeccsd_koopmans(self):
        e,v = mycc.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = mycc.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_eomee_ccsd_singlet(self):
        e, v = mycc.eomee_ccsd_singlet(nroots=1)
        self.assertAlmostEqual(e, 0.3005716731825082, 6)

    def test_eomee_ccsd_triplet(self):
        e, v = mycc.eomee_ccsd_triplet(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 6)

    def test_eomsf_ccsd(self):
        e, v = mycc.eomsf_ccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 6)

    def test_vector_to_amplitudes(self):
        t1, t2 = mycc1.vector_to_amplitudes(mycc1.amplitudes_to_vector(mycc1.t1, mycc1.t2))
        self.assertAlmostEqual(abs(mycc1.t1-t1).sum(), 0, 9)
        self.assertAlmostEqual(abs(mycc1.t2-t2).sum(), 0, 9)

    def test_eomee_ccsd_matvec_singlet(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((no,no,nv,nv)) - .9
        r2 = r2 + r2.transpose(1,0,3,2)
        myeom = eom_rccsd.EOMEESinglet(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.finger(r1), -112883.3791497977, 8)
        self.assertAlmostEqual(lib.finger(r2), -268199.3475813322, 8)

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
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.finger(r1   ), 3550.5250670914056, 9)
        self.assertAlmostEqual(lib.finger(r2[0]), -237433.03756895234,8)
        self.assertAlmostEqual(lib.finger(r2[1]), 127680.0182437716 , 8)

    def test_eomsf_ccsd_matvec(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.finger(r1   ), -19368.729268465482, 8)
        self.assertAlmostEqual(lib.finger(r2[0]), 84325.863680611626 , 8)
        self.assertAlmostEqual(lib.finger(r2[1]), 6715.9574457836134 , 8)

    def test_eomee_diag(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc1).get_diag()
        self.assertAlmostEqual(lib.finger(vec1S),-4714.9854130015719, 9)
        self.assertAlmostEqual(lib.finger(vec1T), 2221.3155272953709, 9)
        self.assertAlmostEqual(lib.finger(vec2) ,-5486.1611871545592, 9)

    def test_ip_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = eom_rccsd.EOMIP(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.finger(r1), 0.37404344676857076, 12)
        self.assertAlmostEqual(lib.finger(r2), -1.1568913404570922, 12)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.finger(vec1), -14894.669606811192, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 1182.3095479451745, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris1)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.finger(vec1), -3795.9122245246967, 9)
        self.assertAlmostEqual(lib.finger(diag), 1106.260154202434, 9)

    def test_ea_matvec(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nv)) - .9
        r2 = numpy.random.random((no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEA(mycc1)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.finger(r1), 1.4488291275539353, 12)
        self.assertAlmostEqual(lib.finger(r2), 0.97080165032287469, 12)
        imds = myeom.make_imds(eris1)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.finger(vec1), -34426.363943760276, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 2724.8239646679217, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris1)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.finger(vec1), -17030.363405297598, 9)
        self.assertAlmostEqual(lib.finger(diag), 4688.9122122011922, 9)


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
        self.assertAlmostEqual(lib.finger(vec1), 25176.428829164193-4955.5351324520125j, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 1106.2601542024306, 9)

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
        self.assertAlmostEqual(lib.finger(vec1), -105083.60825558871+25155.909195554908j, 8)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 4688.9122122011895, 9)


########################################
# With 4-fold symmetry in integrals
    def test_ipccsd2(self):
        e,v = mycc2.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 6)

        e,v = mycc2.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        myeom = eom_rccsd.EOMIP(mycc2)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 6)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 6)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 6)

    def test_ipccsd_koopmans2(self):
        e,v = mycc2.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        e,v = mycc2.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

    def test_ipccsd_partition2(self):
        e,v = mycc2.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 6)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 6)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 6)

        e,v = mycc2.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 6)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 6)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 6)


    def test_eaccsd2(self):
        e,v = mycc2.eaccsd(nroots=1)
        self.assertAlmostEqual(e, 0.16737886338859731, 6)

        e,v = mycc2.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.51006797826488071, 6)

        myeom = eom_rccsd.EOMEA(mycc2)
        lv = myeom.eaccsd(nroots=3, left=True)[1]
        e = myeom.eaccsd_star(e, v, lv)
        self.assertAlmostEqual(e[0], 0.16656250872624662, 6)
        self.assertAlmostEqual(e[1], 0.2394414445283693, 6)
        self.assertAlmostEqual(e[2], 0.41399434356202935, 6)

    def test_eaccsd_koopmans2(self):
        e,v = mycc2.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)

        e,v = mycc2.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)

    def test_eaccsd_partition2(self):
        e,v = mycc2.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.16947311575051136, 6)
        self.assertAlmostEqual(e[1], 0.24234326468848749, 6)
        self.assertAlmostEqual(e[2], 0.7434661346653969 , 6)

        e,v = mycc2.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.16418276148493574, 6)
        self.assertAlmostEqual(e[1], 0.23683978491376495, 6)
        self.assertAlmostEqual(e[2], 0.55640091560545624, 6)


    def test_eeccsd2(self):
        e,v = mycc2.eeccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 6)

        e,v = mycc2.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_eeccsd_koopmans2(self):
        e,v = mycc2.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = mycc2.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

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
        self.assertAlmostEqual(lib.finger(r1), -112883.3791497977, 8)
        self.assertAlmostEqual(lib.finger(r2), -268199.3475813322, 8)

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
        self.assertAlmostEqual(lib.finger(r1   ), 3550.5250670914056, 9)
        self.assertAlmostEqual(lib.finger(r2[0]), -237433.03756895234,8)
        self.assertAlmostEqual(lib.finger(r2[1]), 127680.0182437716 , 8)

    def test_eomsf_ccsd_matvec2(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.finger(r1   ), -19368.729268465482, 8)
        self.assertAlmostEqual(lib.finger(r2[0]), 84325.863680611626 , 8)
        self.assertAlmostEqual(lib.finger(r2[1]), 6715.9574457836134 , 8)

    def test_eomee_diag2(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc21).get_diag()
        self.assertAlmostEqual(lib.finger(vec1S),-4714.9854130015719, 9)
        self.assertAlmostEqual(lib.finger(vec1T), 2221.3155272953709, 9)
        self.assertAlmostEqual(lib.finger(vec2) ,-5486.1611871545592, 9)


    def test_ip_matvec2(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = eom_rccsd.EOMIP(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.finger(r1), 0.37404344676857076, 12)
        self.assertAlmostEqual(lib.finger(r2), -1.1568913404570922, 12)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.finger(vec1), -14894.669606811192, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 1182.3095479451745, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris21)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.finger(vec1), -3795.9122245246967, 9)
        self.assertAlmostEqual(lib.finger(diag), 1106.260154202434, 9)

    def test_ea_matvec2(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((nv)) - .9
        r2 = numpy.random.random((no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEA(mycc21)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.finger(r1), 1.4488291275539353, 12)
        self.assertAlmostEqual(lib.finger(r2), 0.97080165032287469, 12)
        imds = myeom.make_imds(eris21)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.finger(vec1), -34426.363943760276, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 2724.8239646679217, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris21)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.finger(vec1), -17030.363405297598, 9)
        self.assertAlmostEqual(lib.finger(diag), 4688.9122122011922, 9)


########################################
# With 4-fold symmetry in integrals
# max_memory = 0
# direct = True
    def test_ipccsd3(self):
        e,v = mycc3.ipccsd(nroots=1)
        self.assertAlmostEqual(e, 0.4335604332073799, 6)

        e,v = mycc3.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        myeom = eom_rccsd.EOMIP(mycc3)
        lv = myeom.ipccsd(nroots=3, left=True)[1]
        e = myeom.ipccsd_star(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43793202122290747, 6)
        self.assertAlmostEqual(e[1], 0.52287073076243218, 6)
        self.assertAlmostEqual(e[2], 0.67994597799835099, 6)

    def test_ipccsd_koopmans3(self):
        e,v = mycc3.ipccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

        e,v = mycc3.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.4335604332073799, 6)
        self.assertAlmostEqual(e[1], 0.5187659896045407, 6)
        self.assertAlmostEqual(e[2], 0.6782876002229172, 6)

    def test_ipccsd_partition3(self):
        e,v = mycc3.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42728862799879663, 6)
        self.assertAlmostEqual(e[1], 0.51359478811505332, 6)
        self.assertAlmostEqual(e[2], 0.67382901297144682, 6)

        e,v = mycc3.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.42291981842588938, 6)
        self.assertAlmostEqual(e[1], 0.50992428154417802, 6)
        self.assertAlmostEqual(e[2], 0.67006510349161119, 6)


#    def test_eaccsd3(self):
#        e,v = mycc3.eaccsd(nroots=1)
#        self.assertAlmostEqual(e, 0.16737886338859731, 6)
#
#        e,v = mycc3.eaccsd(nroots=3)
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
#        self.assertAlmostEqual(e[2], 0.51006797826488071, 6)
#
#        myeom = eom_rccsd.EOMEA(mycc3)
#        lv = myeom.eaccsd(nroots=3, left=True)[1]
#        e = myeom.eaccsd_star(e, v, lv)
#        self.assertAlmostEqual(e[0], 0.16656250872624662, 6)
#        self.assertAlmostEqual(e[1], 0.2394414445283693, 6)
#        self.assertAlmostEqual(e[2], 0.41399434356202935, 6)
#
#    def test_eaccsd_koopmans3(self):
#        e,v = mycc3.eaccsd(nroots=3, koopmans=True)
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
#        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)
#
#        e,v = mycc3.eaccsd(nroots=3, guess=v[:3])
#        self.assertAlmostEqual(e[0], 0.16737886338859731, 6)
#        self.assertAlmostEqual(e[1], 0.24027613852009164, 6)
#        self.assertAlmostEqual(e[2], 0.73443352557582653, 6)
#
#    def test_eaccsd_partition3(self):
#        e,v = mycc3.eaccsd(nroots=3, partition='mp')
#        self.assertAlmostEqual(e[0], 0.16947311575051136, 6)
#        self.assertAlmostEqual(e[1], 0.24234326468848749, 6)
#        self.assertAlmostEqual(e[2], 0.7434661346653969 , 6)
#
#        e,v = mycc3.eaccsd(nroots=3, partition='full')
#        self.assertAlmostEqual(e[0], 0.16418276148493574, 6)
#        self.assertAlmostEqual(e[1], 0.23683978491376495, 6)
#        self.assertAlmostEqual(e[2], 0.55640091560545624, 6)


    def test_eeccsd3(self):
        e,v = mycc3.eeccsd(nroots=1)
        self.assertAlmostEqual(e, 0.2757159395886167, 6)

        e,v = mycc3.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

    def test_eeccsd_koopmans3(self):
        e,v = mycc3.eeccsd(nroots=4, koopmans=True)
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

        e,v = mycc3.eeccsd(nroots=4, guess=v[:4])
        self.assertAlmostEqual(e[0], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[1], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[2], 0.2757159395886167, 6)
        self.assertAlmostEqual(e[3], 0.3005716731825082, 6)

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
        self.assertAlmostEqual(lib.finger(r1), -112883.3791497977, 8)
        self.assertAlmostEqual(lib.finger(r2), -268199.3475813322, 8)

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
        self.assertAlmostEqual(lib.finger(r1   ), 3550.5250670914056, 9)
        self.assertAlmostEqual(lib.finger(r2[0]), -237433.03756895234,8)
        self.assertAlmostEqual(lib.finger(r2[1]), 127680.0182437716 , 8)

    def test_eomsf_ccsd_matvec3(self):
        numpy.random.seed(10)
        r1 = numpy.random.random((no,nv)) - .9
        r2 = numpy.random.random((2,no,no,nv,nv)) - .9
        myeom = eom_rccsd.EOMEESpinFlip(mycc31)
        vec = myeom.amplitudes_to_vector(r1,r2)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        r1, r2 = myeom.vector_to_amplitudes(vec1)
        self.assertAlmostEqual(lib.finger(r1   ), -19368.729268465482, 8)
        self.assertAlmostEqual(lib.finger(r2[0]), 84325.863680611626 , 8)
        self.assertAlmostEqual(lib.finger(r2[1]), 6715.9574457836134 , 8)

    def test_eomee_diag3(self):
        vec1S, vec1T, vec2 = eom_rccsd.EOMEE(mycc31).get_diag()
        self.assertAlmostEqual(lib.finger(vec1S),-2881.6804563818432, 9)
        self.assertAlmostEqual(lib.finger(vec1T), 2039.7385969969259, 9)
        self.assertAlmostEqual(lib.finger(vec2) ,-4271.6230465236358, 9)


    def test_ip_matvec3(self):
        numpy.random.seed(12)
        r1 = numpy.random.random((no)) - .9
        r2 = numpy.random.random((no,no,nv)) - .9
        myeom = eom_rccsd.EOMIP(mycc31)
        vec = myeom.amplitudes_to_vector(r1,r2)
        r1,r2 = myeom.vector_to_amplitudes(vec)
        myeom.partition = 'mp'
        self.assertAlmostEqual(lib.finger(r1), 0.37404344676857076, 12)
        self.assertAlmostEqual(lib.finger(r2), -1.1568913404570922, 12)
        imds = myeom.make_imds(eris31)
        vec1 = myeom.matvec(vec, imds)
        self.assertAlmostEqual(lib.finger(vec1), -14894.669606811192, 9)
        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 1182.3095479451745, 9)

        myeom.partition = 'full'
        imds = myeom.make_imds(eris31)
        diag = myeom.get_diag(imds)
        vec1 = myeom.matvec(vec, imds, diag=diag)
        self.assertAlmostEqual(lib.finger(vec1), -3795.9122245246967, 9)
        self.assertAlmostEqual(lib.finger(diag), 1106.260154202434, 9)


#    def test_ea_matvec3(self):
#        numpy.random.seed(12)
#        r1 = numpy.random.random((nv)) - .9
#        r2 = numpy.random.random((no,nv,nv)) - .9
#        myeom = eom_rccsd.EOMEA(mycc31)
#        vec = myeom.amplitudes_to_vector(r1,r2)
#        r1,r2 = myeom.vector_to_amplitudes(vec)
#        myeom.partition = 'mp'
#        self.assertAlmostEqual(lib.finger(r1), 1.4488291275539353, 12)
#        self.assertAlmostEqual(lib.finger(r2), 0.97080165032287469, 12)
#        imds = myeom.make_imds(eris31)
#        vec1 = myeom.matvec(vec, imds)
#        self.assertAlmostEqual(lib.finger(vec1), -34426.363943760276, 9)
#        self.assertAlmostEqual(lib.finger(myeom.get_diag()), 2724.8239646679217, 9)
#
#        myeom.partition = 'full'
#        imds = myeom.make_imds(eris31)
#        diag = myeom.get_diag(imds)
#        vec1 = myeom.matvec(vec, imds, diag=diag)
#        self.assertAlmostEqual(lib.finger(vec1), -17030.363405297598, 9)
#        self.assertAlmostEqual(lib.finger(diag), 4688.9122122011922, 9)

if __name__ == "__main__":
    print("Tests for EOM RCCSD")
    unittest.main()

