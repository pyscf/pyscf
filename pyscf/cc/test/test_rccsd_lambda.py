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
from pyscf.cc import rccsd
from pyscf.cc import addons
from pyscf.cc import rccsd_lambda
from pyscf.cc import ccsd_rdm
from pyscf.cc import gccsd, gccsd_lambda

def setUpModule():
    global mol, mf, mycc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.build()
    mf = scf.RHF(mol).run()
    mycc = rccsd.RCCSD(mf)

def tearDownModule():
    global mol, mf, mycc
    mol.stdout.close()
    del mol, mf, mycc

class KnownValues(unittest.TestCase):
    def test_update_lambda_real(self):
        mycc = rccsd.RCCSD(mf)
        np.random.seed(12)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = np.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = np.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + np.diag(range(nmo))*2
        t1 = np.random.random((nocc,nvir))
        t2 = np.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = np.random.random((nocc,nvir))
        l2 = np.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)

        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        idx = np.tril_indices(nvir)
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = fock0

        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new, l2new = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(lib.fp(l1new), -6699.5335665027187, 9)
        self.assertAlmostEqual(lib.fp(l2new), -514.7001243502192 , 9)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).max(), 0, 12)

        mycc.max_memory = 0
        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new, l2new = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
        self.assertAlmostEqual(lib.fp(l1new), -6699.5335665027187, 9)
        self.assertAlmostEqual(lib.fp(l2new), -514.7001243502192 , 9)
        self.assertAlmostEqual(abs(l2new-l2new.transpose(1,0,3,2)).max(), 0, 12)

    def test_update_lambda_complex(self):
        mo_coeff = mf.mo_coeff + np.sin(mf.mo_coeff) * .01j
        nao = mo_coeff.shape[0]
        eri = ao2mo.restore(1, mf._eri, nao)
        eri0 = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo_coeff.conj(), mo_coeff,
                          mo_coeff.conj(), mo_coeff)

        nocc, nvir = 5, nao-5
        eris = rccsd._ChemistsERIs(mol)
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = np.diag(mf.mo_energy)

        np.random.seed(1)
        t1 = np.random.random((nocc,nvir)) + np.random.random((nocc,nvir))*.1j - .5
        t2 = np.random.random((nocc,nocc,nvir,nvir)) - .5
        t2 = t2 + np.sin(t2) * .1j
        t2 = t2 + t2.transpose(1,0,3,2)

        l1 = np.random.random((nocc,nvir)) + np.random.random((nocc,nvir))*.1j - .5
        l2 = np.random.random((nocc,nocc,nvir,nvir)) - .5
        l2 = l2 + np.sin(l2) * .1j
        l2 = l2 + l2.transpose(1,0,3,2)
        mycc = rccsd.RCCSD(mf)
        imds = rccsd_lambda.make_intermediates(mycc, t1, t2, eris)
        l1new_ref, l2new_ref = rccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)

        orbspin = np.zeros(nao*2, dtype=int)
        orbspin[1::2] = 1
        eri1 = np.zeros([nao*2]*4, dtype=np.complex128)
        eri1[0::2,0::2,0::2,0::2] = \
        eri1[0::2,0::2,1::2,1::2] = \
        eri1[1::2,1::2,0::2,0::2] = \
        eri1[1::2,1::2,1::2,1::2] = eri0
        eri1 = eri1.transpose(0,2,1,3) - eri1.transpose(0,2,3,1)
        erig = gccsd._PhysicistsERIs(mol)
        nocc *= 2
        nvir *= 2
        erig.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        erig.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        erig.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        erig.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        erig.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        erig.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        erig.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        mo_e = np.array([mf.mo_energy]*2)
        erig.fock = np.diag(mo_e.T.ravel())
        erig.mo_energy = erig.fock.diagonal()

        myccg = gccsd.GCCSD(scf.addons.convert_to_ghf(mf))
        t1, t2 = myccg.amplitudes_from_ccsd(t1, t2)
        l1, l2 = myccg.amplitudes_from_ccsd(l1, l2)
        imds = gccsd_lambda.make_intermediates(myccg, t1, t2, erig)
        l1new, l2new = gccsd_lambda.update_lambda(myccg, t1, t2, l1, l2, erig, imds)
        self.assertAlmostEqual(float(abs(l1new[0::2,0::2]-l1new_ref).max()), 0, 9)
        l2aa = l2new[0::2,0::2,0::2,0::2]
        l2ab = l2new[0::2,1::2,0::2,1::2]
        self.assertAlmostEqual(float(abs(l2ab-l2new_ref).max()), 0, 9)
        self.assertAlmostEqual(float(abs(l2ab-l2ab.transpose(1,0,2,3) - l2aa).max()), 0, 9)

    def test_rdm(self):
        mycc = rccsd.RCCSD(mf)
        mycc.frozen = 1
        mycc.kernel()
        dm1 = mycc.make_rdm1()
        dm2 = mycc.make_rdm2()
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        nmo = mf.mo_coeff.shape[1]
        eri = ao2mo.restore(1, ao2mo.kernel(mf._eri, mf.mo_coeff), nmo)
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 6)

        d1 = ccsd_rdm._gamma1_intermediates(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2)
        mycc1 = mycc.copy()
        mycc1.max_memory = 0
        d2 = ccsd_rdm._gamma2_intermediates(mycc1, mycc.t1, mycc.t2, mycc.l1, mycc.l2, True)
        dm2 = ccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True)
        e1 = numpy.einsum('ij,ji', h1, dm1)
        e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
        e1+= mol.energy_nuc()
        self.assertAlmostEqual(e1, mycc.e_tot, 6)

    def test_rdm_trace(self):
        mycc = rccsd.RCCSD(mf)
        numpy.random.seed(2)
        nocc = 5
        nmo = 12
        nvir = nmo - nocc
        eri0 = numpy.random.random((nmo,nmo,nmo,nmo))
        eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
        fock0 = numpy.random.random((nmo,nmo))
        fock0 = fock0 + fock0.T + numpy.diag(range(nmo))*2
        t1 = numpy.random.random((nocc,nvir))
        t2 = numpy.random.random((nocc,nocc,nvir,nvir))
        t2 = t2 + t2.transpose(1,0,3,2)
        l1 = numpy.random.random((nocc,nvir))
        l2 = numpy.random.random((nocc,nocc,nvir,nvir))
        l2 = l2 + l2.transpose(1,0,3,2)
        h1 = fock0 - (numpy.einsum('kkpq->pq', eri0[:nocc,:nocc])*2
                    - numpy.einsum('pkkq->pq', eri0[:,:nocc,:nocc]))

        eris = lambda:None
        eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        eris.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
        eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        eris.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        eris.fock = fock0

        doo, dov, dvo, dvv = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
        self.assertAlmostEqual((numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc]))*2, -20166.329861034799, 8)
        self.assertAlmostEqual((numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:]))*2,  58078.964019246778, 8)
        self.assertAlmostEqual((numpy.einsum('ai,ia', dvo, fock0[:nocc,nocc:]))*2, -74994.356886784764, 8)
        self.assertAlmostEqual((numpy.einsum('ia,ai', dov, fock0[nocc:,:nocc]))*2,  34.010188025702391, 9)

        fdm2 = lib.H5TmpFile()
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
                ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)
        self.assertAlmostEqual(lib.fp(numpy.array(dovov)), -14384.907042073517, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dvvvv)), -25.374007033024839, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(doooo)),  60.114594698129963, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(doovv)), -79.176348067958401, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dovvo)),   9.864134457251815, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dovvv)), -421.90333700061342, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dooov)), -592.66863759586136, 9)
        fdm2 = None

        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = \
                ccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
        self.assertAlmostEqual(lib.fp(numpy.array(dovov)), -14384.907042073517, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dvvvv)),  45.872344902116758, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(doooo)),  60.114594698129963, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(doovv)), -79.176348067958401, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dovvo)),   9.864134457251815, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dovvv)), -421.90333700061342, 9)
        self.assertAlmostEqual(lib.fp(numpy.array(dooov)), -592.66863759586136, 9)

        self.assertAlmostEqual(numpy.einsum('kilj,kilj', doooo, eris.oooo)*2, 15939.9007625418, 7)
        self.assertAlmostEqual(numpy.einsum('acbd,acbd', dvvvv, eris.vvvv)*2, 37581.823919588 , 7)
        self.assertAlmostEqual(numpy.einsum('jkia,jkia', dooov, eris.ooov)*2, 128470.009687716, 7)
        self.assertAlmostEqual(numpy.einsum('icba,icba', dovvv, eris.ovvv)*2,-166794.225195056, 7)
        self.assertAlmostEqual(numpy.einsum('iajb,iajb', dovov, eris.ovov)*2,-719279.812916893, 7)
        self.assertAlmostEqual(numpy.einsum('jbai,jbia', dovvo, eris.ovov)*2+
                               numpy.einsum('jiab,jiba', doovv, eris.oovv)*2,-53634.0012286654, 7)

        dm1 = ccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2)
        dm2 = ccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2)
        e2 =(numpy.einsum('ijkl,ijkl', doooo, eris.oooo)*2
            +numpy.einsum('acbd,acbd', dvvvv, eris.vvvv)*2
            +numpy.einsum('jkia,jkia', dooov, eris.ooov)*2
            +numpy.einsum('icba,icba', dovvv, eris.ovvv)*2
            +numpy.einsum('iajb,iajb', dovov, eris.ovov)*2
            +numpy.einsum('jbai,jbia', dovvo, eris.ovov)*2
            +numpy.einsum('ijab,ijab', doovv, eris.oovv)*2
            +numpy.einsum('ij,ij', doo, fock0[:nocc,:nocc])*2
            +numpy.einsum('ai,ia', dvo, fock0[:nocc,nocc:])*2
            +numpy.einsum('ia,ai', dov, fock0[nocc:,:nocc])*2
            +numpy.einsum('ab,ab', dvv, fock0[nocc:,nocc:])*2
            +fock0[:nocc].trace()*2
            -numpy.einsum('kkpq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace()*2
            +numpy.einsum('pkkq->pq', eri0[:nocc,:nocc,:nocc,:nocc]).trace())
        self.assertAlmostEqual(e2, -794721.197459942, 8)
        self.assertAlmostEqual(numpy.einsum('pqrs,pqrs', dm2, eri0)*.5 +
                               numpy.einsum('pq,qp', dm1, h1), e2, 9)

        self.assertAlmostEqual(abs(dm2-dm2.transpose(1,0,3,2)).max(), 0, 9)
        self.assertAlmostEqual(abs(dm2-dm2.transpose(2,3,0,1)).max(), 0, 9)

        d1 = numpy.einsum('kkpq->qp', dm2) / 9
        self.assertAlmostEqual(abs(d1-dm1).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for RCCSD lambda")
    unittest.main()
