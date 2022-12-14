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
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.cc import uccsd
from pyscf.cc import addons
from pyscf.cc import uccsd_lambda
from pyscf.cc import gccsd, gccsd_lambda

def setUpModule():
    global mol, mf, mycc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mycc = uccsd.UCCSD(mf)

def tearDownModule():
    global mol, mf, mycc
    del mol, mf, mycc

class KnownValues(unittest.TestCase):
    def test_update_lambda_real(self):
        numpy.random.seed(21)
        eris = mycc.ao2mo()
        gcc1 = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        eri1 = gcc1.ao2mo()
        orbspin = eri1.orbspin

        nocc = mol.nelectron
        nvir = mol.nao_nr()*2 - nocc

        t1r = numpy.random.random((nocc,nvir))*.1
        t2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
        t2r = t2r - t2r.transpose(1,0,2,3)
        t2r = t2r - t2r.transpose(0,1,3,2)
        l1r = numpy.random.random((nocc,nvir))*.1
        l2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
        l2r = l2r - l2r.transpose(1,0,2,3)
        l2r = l2r - l2r.transpose(0,1,3,2)
        t1r = addons.spin2spatial(t1r, orbspin)
        t2r = addons.spin2spatial(t2r, orbspin)
        t1r = addons.spatial2spin(t1r, orbspin)
        t2r = addons.spatial2spin(t2r, orbspin)
        l1r = addons.spin2spatial(l1r, orbspin)
        l2r = addons.spin2spatial(l2r, orbspin)
        l1r = addons.spatial2spin(l1r, orbspin)
        l2r = addons.spatial2spin(l2r, orbspin)
        imds = gccsd_lambda.make_intermediates(gcc1, t1r, t2r, eri1)
        l1ref, l2ref = gccsd_lambda.update_lambda(gcc1, t1r, t2r, l1r, l2r, eri1, imds)

        t1 = addons.spin2spatial(t1r, orbspin)
        t2 = addons.spin2spatial(t2r, orbspin)
        l1 = addons.spin2spatial(l1r, orbspin)
        l2 = addons.spin2spatial(l2r, orbspin)
        imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1, l2 = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(float(abs(addons.spatial2spin(l1, orbspin)-l1ref).max()), 0, 8)
        self.assertAlmostEqual(float(abs(addons.spatial2spin(l2, orbspin)-l2ref).max()), 0, 8)

        l1ref = addons.spin2spatial(l1ref, orbspin)
        l2ref = addons.spin2spatial(l2ref, orbspin)
        self.assertAlmostEqual(abs(l1[0]-l1ref[0]).max(), 0, 8)
        self.assertAlmostEqual(abs(l1[1]-l1ref[1]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[0]-l2ref[0]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[1]-l2ref[1]).max(), 0, 8)
        self.assertAlmostEqual(abs(l2[2]-l2ref[2]).max(), 0, 8)

    def test_update_lambda_complex(self):
        nocca, noccb = mol.nelec
        nmo = mol.nao_nr()
        nvira,nvirb = nmo-nocca, nmo-noccb
        numpy.random.seed(9)
        t1 = [numpy.random.random((nocca,nvira))-.9,
              numpy.random.random((noccb,nvirb))-.9]
        l1 = [numpy.random.random((nocca,nvira))-.9,
              numpy.random.random((noccb,nvirb))-.9]
        t2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        t2[0] = t2[0] - t2[0].transpose(1,0,2,3)
        t2[0] = t2[0] - t2[0].transpose(0,1,3,2)
        t2[2] = t2[2] - t2[2].transpose(1,0,2,3)
        t2[2] = t2[2] - t2[2].transpose(0,1,3,2)
        l2 = [numpy.random.random((nocca,nocca,nvira,nvira))-.9,
              numpy.random.random((nocca,noccb,nvira,nvirb))-.9,
              numpy.random.random((noccb,noccb,nvirb,nvirb))-.9]
        l2[0] = l2[0] - l2[0].transpose(1,0,2,3)
        l2[0] = l2[0] - l2[0].transpose(0,1,3,2)
        l2[2] = l2[2] - l2[2].transpose(1,0,2,3)
        l2[2] = l2[2] - l2[2].transpose(0,1,3,2)

#        eris = mycc.ao2mo()
#        imds = make_intermediates(mycc, t1, t2, eris)
#        l1new, l2new = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
#        print(lib.fp(l1new[0]) --104.55975252585894)
#        print(lib.fp(l1new[1]) --241.12677819375281)
#        print(lib.fp(l2new[0]) --0.4957533529669417)
#        print(lib.fp(l2new[1]) - 15.46423057451851 )
#        print(lib.fp(l2new[2]) - 5.8430776663704407)

        nocca, noccb = mol.nelec
        mo_a = mf.mo_coeff[0] + numpy.sin(mf.mo_coeff[0]) * .01j
        mo_b = mf.mo_coeff[1] + numpy.sin(mf.mo_coeff[1]) * .01j
        nao = mo_a.shape[0]
        eri = ao2mo.restore(1, mf._eri, nao)
        eri0aa = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_a.conj(), mo_a, mo_a.conj(), mo_a)
        eri0ab = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_a.conj(), mo_a, mo_b.conj(), mo_b)
        eri0bb = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_b.conj(), mo_b, mo_b.conj(), mo_b)
        eri0ba = eri0ab.transpose(2,3,0,1)

        nvira = nao - nocca
        nvirb = nao - noccb
        eris = uccsd._ChemistsERIs(mol)
        eris.oooo = eri0aa[:nocca,:nocca,:nocca,:nocca].copy()
        eris.ovoo = eri0aa[:nocca,nocca:,:nocca,:nocca].copy()
        eris.oovv = eri0aa[:nocca,:nocca,nocca:,nocca:].copy()
        eris.ovvo = eri0aa[:nocca,nocca:,nocca:,:nocca].copy()
        eris.ovov = eri0aa[:nocca,nocca:,:nocca,nocca:].copy()
        eris.ovvv = eri0aa[:nocca,nocca:,nocca:,nocca:].copy()
        eris.vvvv = eri0aa[nocca:,nocca:,nocca:,nocca:].copy()

        eris.OOOO = eri0bb[:noccb,:noccb,:noccb,:noccb].copy()
        eris.OVOO = eri0bb[:noccb,noccb:,:noccb,:noccb].copy()
        eris.OOVV = eri0bb[:noccb,:noccb,noccb:,noccb:].copy()
        eris.OVVO = eri0bb[:noccb,noccb:,noccb:,:noccb].copy()
        eris.OVOV = eri0bb[:noccb,noccb:,:noccb,noccb:].copy()
        eris.OVVV = eri0bb[:noccb,noccb:,noccb:,noccb:].copy()
        eris.VVVV = eri0bb[noccb:,noccb:,noccb:,noccb:].copy()

        eris.ooOO = eri0ab[:nocca,:nocca,:noccb,:noccb].copy()
        eris.ovOO = eri0ab[:nocca,nocca:,:noccb,:noccb].copy()
        eris.ooVV = eri0ab[:nocca,:nocca,noccb:,noccb:].copy()
        eris.ovVO = eri0ab[:nocca,nocca:,noccb:,:noccb].copy()
        eris.ovOV = eri0ab[:nocca,nocca:,:noccb,noccb:].copy()
        eris.ovVV = eri0ab[:nocca,nocca:,noccb:,noccb:].copy()
        eris.vvVV = eri0ab[nocca:,nocca:,noccb:,noccb:].copy()

        eris.OOoo = eri0ba[:noccb,:noccb,:nocca,:nocca].copy()
        eris.OVoo = eri0ba[:noccb,noccb:,:nocca,:nocca].copy()
        eris.OOvv = eri0ba[:noccb,:noccb,nocca:,nocca:].copy()
        eris.OVvo = eri0ba[:noccb,noccb:,nocca:,:nocca].copy()
        eris.OVov = eri0ba[:noccb,noccb:,:nocca,nocca:].copy()
        eris.OVvv = eri0ba[:noccb,noccb:,nocca:,nocca:].copy()
        eris.VVvv = eri0ba[noccb:,noccb:,nocca:,nocca:].copy()

        eris.focka = numpy.diag(mf.mo_energy[0])
        eris.fockb = numpy.diag(mf.mo_energy[1])
        eris.mo_energy = mf.mo_energy

        t1[0] = t1[0] + numpy.sin(t1[0]) * .05j
        t1[1] = t1[1] + numpy.sin(t1[1]) * .05j
        t2[0] = t2[0] + numpy.sin(t2[0]) * .05j
        t2[1] = t2[1] + numpy.sin(t2[1]) * .05j
        t2[2] = t2[2] + numpy.sin(t2[2]) * .05j
        l1[0] = l1[0] + numpy.sin(l1[0]) * .05j
        l1[1] = l1[1] + numpy.sin(l1[1]) * .05j
        l2[0] = l2[0] + numpy.sin(l2[0]) * .05j
        l2[1] = l2[1] + numpy.sin(l2[1]) * .05j
        l2[2] = l2[2] + numpy.sin(l2[2]) * .05j
        imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new_ref, l2new_ref = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)

        nocc = nocca + noccb
        orbspin = numpy.zeros(nao*2, dtype=int)
        orbspin[1::2] = 1
        orbspin[nocc-1] = 0
        orbspin[nocc  ] = 1
        eri1 = numpy.zeros([nao*2]*4, dtype=numpy.complex128)
        idxa = numpy.where(orbspin == 0)[0]
        idxb = numpy.where(orbspin == 1)[0]
        eri1[idxa[:,None,None,None],idxa[:,None,None],idxa[:,None],idxa] = eri0aa
        eri1[idxa[:,None,None,None],idxa[:,None,None],idxb[:,None],idxb] = eri0ab
        eri1[idxb[:,None,None,None],idxb[:,None,None],idxa[:,None],idxa] = eri0ba
        eri1[idxb[:,None,None,None],idxb[:,None,None],idxb[:,None],idxb] = eri0bb
        eri1 = eri1.transpose(0,2,1,3) - eri1.transpose(0,2,3,1)
        erig = gccsd._PhysicistsERIs()
        erig.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        erig.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        erig.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        erig.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        erig.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        erig.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        erig.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        mo_e = numpy.empty(nao*2)
        mo_e[orbspin==0] = mf.mo_energy[0]
        mo_e[orbspin==1] = mf.mo_energy[1]
        erig.fock = numpy.diag(mo_e)
        erig.mo_energy = mo_e.real

        myccg = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        t1 = myccg.spatial2spin(t1, orbspin)
        t2 = myccg.spatial2spin(t2, orbspin)
        l1 = myccg.spatial2spin(l1, orbspin)
        l2 = myccg.spatial2spin(l2, orbspin)
        imds = gccsd_lambda.make_intermediates(myccg, t1, t2, erig)
        l1new, l2new = gccsd_lambda.update_lambda(myccg, t1, t2, l1, l2, erig, imds)
        l1new = myccg.spin2spatial(l1new, orbspin)
        l2new = myccg.spin2spatial(l2new, orbspin)
        self.assertAlmostEqual(abs(l1new[0] - l1new_ref[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(l1new[1] - l1new_ref[1]).max(), 0, 11)
        self.assertAlmostEqual(abs(l2new[0] - l2new_ref[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(l2new[1] - l2new_ref[1]).max(), 0, 11)
        self.assertAlmostEqual(abs(l2new[2] - l2new_ref[2]).max(), 0, 11)


if __name__ == "__main__":
    print("Full Tests for UCCSD lambda")
    unittest.main()
