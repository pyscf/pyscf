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
from pyscf import ao2mo
from pyscf.cc import ccsd_t
from pyscf.cc import gccsd, gccsd_t
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm

def setUpModule():
    global mol, rhf, mcc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -.757 , .587)],
        [1 , (0. ,  .757 , .587)]]
    mol.symmetry = True
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.basis = 'ccpvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-14
    rhf.scf()

    mcc = cc.CCSD(rhf)
    mcc.conv_tol = 1e-14
    mcc.ccsd()

def tearDownModule():
    global mol, rhf, mcc
    mol.stdout.close()
    del mol, rhf, mcc

class KnownValues(unittest.TestCase):
    def test_ccsd_t(self):
        mol = gto.M()
        numpy.random.seed(12)
        nocc, nvir = 5, 12
        nmo = nocc + nvir
        eris = cc.rccsd._ChemistsERIs()
        eri1 = numpy.random.random((nmo,nmo,nmo,nmo)) - .5
        eri1 = eri1 + eri1.transpose(1,0,2,3)
        eri1 = eri1 + eri1.transpose(0,1,3,2)
        eri1 = eri1 + eri1.transpose(2,3,0,1)
        eri1 *= .1
        eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]
        eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc]
        eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:]
        t1 = numpy.random.random((nocc,nvir)) * .1
        t2 = numpy.random.random((nocc,nocc,nvir,nvir)) * .1
        t2 = t2 + t2.transpose(1,0,3,2)
        mf = scf.RHF(mol)
        mycc = cc.CCSD(mf)
        mycc.incore_complete = True
        mycc.mo_energy = mycc._scf.mo_energy = numpy.arange(0., nocc+nvir)
        f = numpy.random.random((nmo,nmo)) * .1
        eris.fock = f+f.T + numpy.diag(numpy.arange(nmo))
        eris.mo_energy = eris.fock.diagonal()
        e = ccsd_t.kernel(mycc, eris, t1, t2)
        self.assertAlmostEqual(e, -45.96028705175308, 9)

        mycc.max_memory = 0
        e = ccsd_t.kernel(mycc, eris, t1, t2)
        self.assertAlmostEqual(e, -45.96028705175308, 9)

    def test_ccsd_t_symm(self):
        e3a = ccsd_t.kernel(mcc, mcc.ao2mo())
        self.assertAlmostEqual(e3a, -0.003060022611584471, 9)

        mcc.mol.symmetry = False
        e3a = ccsd_t.kernel(mcc, mcc.ao2mo())
        self.assertAlmostEqual(e3a, -0.003060022611584471, 9)
        mcc.mol.symmetry = True

    def test_sort_eri(self):
        eris = mcc.ao2mo()
        nocc, nvir = mcc.t1.shape
        nmo = nocc + nvir
        vvop = numpy.empty((nvir,nvir,nocc,nmo))
        log = lib.logger.Logger(mcc.stdout, mcc.verbose)
        orbsym = ccsd_t._sort_eri(mcc, eris, nocc, nvir, vvop, log)

        o_sorted = numpy.hstack([numpy.where(orbsym[:nocc] == i)[0] for i in range(8)])
        v_sorted = numpy.hstack([numpy.where(orbsym[nocc:] == i)[0] for i in range(8)])
        eris_vvop = numpy.empty((nvir,nvir,nocc,nmo))
        eris_voov = numpy.asarray(eris.ovvo).transpose(1,0,3,2)
        eris_voov = eris_voov[v_sorted][:,o_sorted][:,:,o_sorted][:,:,:,v_sorted]
        eris_vvop[:,:,:,:nocc] = eris_voov.transpose(0,3,1,2)
        eris_vovv = lib.unpack_tril(numpy.asarray(eris.ovvv).transpose(1,0,2).reshape(nocc*nvir,-1))
        eris_vovv = eris_vovv.reshape(nvir,nocc,nvir,nvir)
        eris_vovv = eris_vovv[v_sorted][:,o_sorted][:,:,v_sorted][:,:,:,v_sorted]
        eris_vvop[:,:,:,nocc:] = eris_vovv.transpose(0,2,1,3)
        self.assertAlmostEqual(abs(eris_vvop-vvop).max(), 0, 9)

    def test_sort_t2_vooo(self):
        t1 = mcc.t1
        t2 = mcc.t2
        eris = mcc.ao2mo()
        nocc, nvir = t1.shape
        nmo = nocc + nvir
        mol = mcc.mol
        orbsym = symm.addons.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                            mcc.mo_coeff)
        orbsym = numpy.asarray(orbsym, dtype=numpy.int32)
        t2ref = t2.copy()
        mo_energy, t1T, t2T, vooo, foVT, restore_t2_inplace = \
                ccsd_t._sort_t2_vooo_(mcc, orbsym, t1, t2.copy(), eris)
        self.assertAlmostEqual(abs(t2ref-restore_t2_inplace(t2T.copy())).max(), 0, 12)

        o_sorted = numpy.hstack([numpy.where(orbsym[:nocc] == i)[0] for i in range(8)])
        v_sorted = numpy.hstack([numpy.where(orbsym[nocc:] == i)[0] for i in range(8)])
        o_sym = orbsym[o_sorted]
        oo_sym = (o_sym[:,None] ^ o_sym).ravel()
        oo_sorted = numpy.hstack([numpy.where(oo_sym == i)[0] for i in range(8)])

        ref_t2T = t2.transpose(2,3,1,0)
        ref_t2T = ref_t2T[v_sorted][:,v_sorted][:,:,o_sorted][:,:,:,o_sorted]
        ref_t2T = ref_t2T.reshape(nvir,nvir,-1)[:,:,oo_sorted].reshape(nvir,nvir,nocc,nocc)
        ref_vooo = numpy.asarray(eris.ovoo).transpose(1,0,2,3)
        ref_vooo = ref_vooo[v_sorted][:,o_sorted][:,:,o_sorted][:,:,:,o_sorted]
        ref_vooo = ref_vooo.reshape(nvir,-1,nocc)[:,oo_sorted].reshape(nvir,nocc,nocc,nocc)

        self.assertAlmostEqual(abs(ref_vooo-vooo).sum(), 0, 9)
        self.assertAlmostEqual(abs(ref_t2T-t2T).sum(), 0, 9)

    def test_ccsd_t_complex(self):
        mol = gto.M()
        numpy.random.seed(12)
        nocc, nvir = 3, 4
        nmo = nocc + nvir
        eris = cc.rccsd._ChemistsERIs()
        eri1 = (numpy.random.random((nmo,nmo,nmo,nmo)) +
                numpy.random.random((nmo,nmo,nmo,nmo)) * .8j - .5-.4j)
        eri1 = eri1 + eri1.transpose(1,0,2,3)
        eri1 = eri1 + eri1.transpose(0,1,3,2)
        eri1 = eri1 + eri1.transpose(2,3,0,1)
        eri1 *= .1
        eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:]
        eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc]
        eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:]
        t1 = (numpy.random.random((nocc,nvir)) * .1 +
              numpy.random.random((nocc,nvir)) * .1j)
        t2 = (numpy.random.random((nocc,nocc,nvir,nvir)) * .1 +
              numpy.random.random((nocc,nocc,nvir,nvir)) * .1j)
        t2 = t2 + t2.transpose(1,0,3,2)
        mf = scf.RHF(mol)
        mcc = cc.CCSD(mf)
        f = (numpy.random.random((nmo,nmo)) * .1 +
             numpy.random.random((nmo,nmo)) * .1j)
        eris.fock = f+f.T.conj() + numpy.diag(numpy.arange(nmo))
        eris.mo_energy = eris.fock.diagonal().real
        e0 = ccsd_t.kernel(mcc, eris, t1, t2)

        eri2 = numpy.zeros((nmo*2,nmo*2,nmo*2,nmo*2), dtype=numpy.complex128)
        orbspin = numpy.zeros(nmo*2,dtype=int)
        orbspin[1::2] = 1
        eri2[0::2,0::2,0::2,0::2] = eri1
        eri2[1::2,1::2,0::2,0::2] = eri1
        eri2[0::2,0::2,1::2,1::2] = eri1
        eri2[1::2,1::2,1::2,1::2] = eri1
        eri2 = eri2.transpose(0,2,1,3) - eri2.transpose(0,2,3,1)
        fock = numpy.zeros((nmo*2,nmo*2), dtype=numpy.complex128)
        fock[0::2,0::2] = eris.fock
        fock[1::2,1::2] = eris.fock
        eris1 = gccsd._PhysicistsERIs()
        eris1.ovvv = eri2[:nocc*2,nocc*2:,nocc*2:,nocc*2:]
        eris1.oovv = eri2[:nocc*2,:nocc*2,nocc*2:,nocc*2:]
        eris1.ooov = eri2[:nocc*2,:nocc*2,:nocc*2,nocc*2:]
        eris1.fock = fock
        eris1.mo_energy = fock.diagonal().real
        t1 = gccsd.spatial2spin(t1, orbspin)
        t2 = gccsd.spatial2spin(t2, orbspin)
        gcc = gccsd.GCCSD(scf.GHF(gto.M()))
        e1 = gccsd_t.kernel(gcc, eris1, t1, t2)
        self.assertAlmostEqual(e0, e1.real, 9)
        self.assertAlmostEqual(e1, -0.98756910139720788-0.0019567929592079489j, 9)

    def test_ccsd_t_rdm(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -.957 , .587)],
            [1 , (0.2,  .757 , .487)]]

        mol.basis = '631g'
        mol.build()
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-1
        mf.scf()
        mcc = mf.CCSD()
        ecc, t1, t2 = mcc.kernel()
        eris = mcc.ao2mo()
        e3ref = ccsd_t.kernel(mcc, eris, t1, t2)
        l1, l2 = ccsd_t_lambda.kernel(mcc, eris, t1, t2)[1:]

        eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
        nmo = mf.mo_coeff.shape[1]
        eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
        dm1 = ccsd_t_rdm.make_rdm1(mcc, t1, t2, l1, l2, eris=eris)
        dm2 = ccsd_t_rdm.make_rdm2(mcc, t1, t2, l1, l2, eris=eris)
        h1 = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
        e3 =(numpy.einsum('ij,ji->', h1, dm1) +
             numpy.einsum('ijkl,ijkl->', eri_mo, dm2)*.5 + mf.mol.energy_nuc())
        self.assertAlmostEqual(e3ref, e3-(mf.e_tot+ecc), 7)


if __name__ == "__main__":
    print("Full Tests for CCSD(T)")
    unittest.main()
