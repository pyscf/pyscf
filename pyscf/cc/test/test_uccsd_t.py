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
from pyscf import ao2mo
from pyscf import cc
from pyscf.cc import uccsd_t
from pyscf.cc import gccsd, gccsd_t
from pyscf.cc import uccsd_t_slow
from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_rdm
from pyscf.cc import uccsd_t_blocked
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm


def setUpModule():
    global mol, mol1, mf, mcc
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
    mcc = cc.UCCSD(mf)
    mcc.conv_tol = 1e-14
    mcc.kernel()

def tearDownModule():
    global mol, mol1, mf, mcc
    mol.stdout.close()
    del mol, mol1, mf, mcc

class KnownValues(unittest.TestCase):
    def test_uccsd_t(self):
        mf1 = mf.copy()
        nao, nmo = mf.mo_coeff[0].shape
        numpy.random.seed(10)
        mf1.mo_coeff = numpy.random.random((2,nao,nmo)) - .5
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
        t1 = t1a, t1b
        t2 = t2aa, t2ab, t2bb
        mycc = cc.UCCSD(mf1)
        eris = mycc.ao2mo(mf1.mo_coeff)
        mycc.incore_complete = True
        e3a = mycc.ccsd_t([t1a,t1b], [t2aa, t2ab, t2bb], eris)
        self.assertAlmostEqual(e3a, 15.582860941071505, 8)

        mycc.incore_complete = False
        mycc.max_memory = 0
        e3a = uccsd_t.kernel(mycc, eris, [t1a,t1b], [t2aa, t2ab, t2bb])
        self.assertAlmostEqual(e3a, 15.582860941071505, 8)

        e3a = mcc.ccsd_t()
        self.assertAlmostEqual(e3a, -0.0009857042572475674, 11)

    #def test_uccsd_t_symm(self):
    #    mf = scf.UHF(mol).run(conv_tol=1e-14)
    #    mcc = cc.UCCSD(mf)
    #    mcc.conv_tol = 1e-14
    #    e3a = mcc.run().ccsd_t()
    #    self.assertAlmostEqual(e3a, -0.0030600226107389866, 11)

    def test_uccsd_t_complex(self):
        mol = gto.M()
        numpy.random.seed(12)
        nocca, noccb, nvira, nvirb = 3, 2, 4, 5
        nmo = nocca + nvira
        eris = cc.uccsd._ChemistsERIs()
        eris.nocca = nocca
        eris.noccb = noccb
        eris.nocc = (nocca, noccb)
        eri1 = (numpy.random.random((3,nmo,nmo,nmo,nmo)) +
                numpy.random.random((3,nmo,nmo,nmo,nmo)) * .8j - .5-.4j)
        eri1 = eri1 + eri1.transpose(0,2,1,4,3).conj()
        eri1[0] = eri1[0] + eri1[0].transpose(2,3,0,1)
        eri1[2] = eri1[2] + eri1[2].transpose(2,3,0,1)
        eri1 *= .1
        eris.ovvv = eri1[0,:nocca,nocca:,nocca:,nocca:]
        eris.ovov = eri1[0,:nocca,nocca:,:nocca,nocca:]
        eris.ovoo = eri1[0,:nocca,nocca:,:nocca,:nocca]
        eris.OVVV = eri1[2,:noccb,noccb:,noccb:,noccb:]
        eris.OVOV = eri1[2,:noccb,noccb:,:noccb,noccb:]
        eris.OVOO = eri1[2,:noccb,noccb:,:noccb,:noccb]
        eris.voVP = eri1[1,nocca:,:nocca,noccb:,:     ]
        eris.ovVV = eri1[1,:nocca,nocca:,noccb:,noccb:]
        eris.ovOV = eri1[1,:nocca,nocca:,:noccb,noccb:]
        eris.ovOO = eri1[1,:nocca,nocca:,:noccb,:noccb]
        eris.OVvv = eri1[1,nocca:,nocca:,:noccb,noccb:].transpose(2,3,0,1)
        eris.OVoo = eri1[1,:nocca,:nocca,:noccb,noccb:].transpose(2,3,0,1)
        t1a  = .1 * numpy.random.random((nocca,nvira)) + numpy.random.random((nocca,nvira))*.1j
        t1b  = .1 * numpy.random.random((noccb,nvirb)) + numpy.random.random((noccb,nvirb))*.1j
        t2aa = .1 * numpy.random.random((nocca,nocca,nvira,nvira)) + numpy.random.random((nocca,nocca,nvira,nvira))*.1j
        t2aa = t2aa - t2aa.transpose(0,1,3,2)
        t2aa = t2aa - t2aa.transpose(1,0,2,3)
        t2bb = .1 * numpy.random.random((noccb,noccb,nvirb,nvirb)) + numpy.random.random((noccb,noccb,nvirb,nvirb))*.1j
        t2bb = t2bb - t2bb.transpose(0,1,3,2)
        t2bb = t2bb - t2bb.transpose(1,0,2,3)
        t2ab = .1 * numpy.random.random((nocca,noccb,nvira,nvirb)) + numpy.random.random((nocca,noccb,nvira,nvirb))*.1j
        f = (numpy.random.random((2,nmo,nmo)) * .4 +
             numpy.random.random((2,nmo,nmo)) * .4j)
        eris.focka = f[0]+f[0].T.conj() + numpy.diag(numpy.arange(nmo))
        eris.fockb = f[1]+f[1].T.conj() + numpy.diag(numpy.arange(nmo))
        eris.mo_energy = (eris.focka.diagonal().real,
                          eris.fockb.diagonal().real)
        t1 = t1a, t1b
        t2 = t2aa, t2ab, t2bb
        mcc = cc.UCCSD(scf.UHF(mol))
        mcc.nocc = eris.nocc
        e0 = uccsd_t.kernel(mcc, eris, t1, t2)

        eri2 = numpy.zeros((nmo*2,nmo*2,nmo*2,nmo*2), dtype=eri1.dtype)
        orbspin = numpy.zeros(nmo*2,dtype=int)
        orbspin[1::2] = 1
        eri2[0::2,0::2,0::2,0::2] = eri1[0]
        eri2[1::2,1::2,0::2,0::2] = eri1[1].transpose(2,3,0,1)
        eri2[0::2,0::2,1::2,1::2] = eri1[1]
        eri2[1::2,1::2,1::2,1::2] = eri1[2]
        eri2 = eri2.transpose(0,2,1,3) - eri2.transpose(0,2,3,1)
        fock = numpy.zeros((nmo*2,nmo*2), dtype=eris.focka.dtype)
        fock[0::2,0::2] = eris.focka
        fock[1::2,1::2] = eris.fockb
        eris1 = gccsd._PhysicistsERIs()
        nocc = nocca + noccb
        eris1.ovvv = eri2[:nocc,nocc:,nocc:,nocc:]
        eris1.oovv = eri2[:nocc,:nocc,nocc:,nocc:]
        eris1.ooov = eri2[:nocc,:nocc,:nocc,nocc:]
        eris1.fock = fock
        eris1.mo_energy = fock.diagonal().real
        t1 = gccsd.spatial2spin(t1, orbspin)
        t2 = gccsd.spatial2spin(t2, orbspin)
        gcc = gccsd.GCCSD(scf.GHF(gto.M()))
        e1 = gccsd_t.kernel(gcc, eris1, t1, t2)
        self.assertAlmostEqual(e0, e1.real, 9)
        self.assertAlmostEqual(e1, -0.056092415718338388-0.011390417704868244j, 9)

    def test_uccsd_t_rdm(self):
        mol = gto.Mole()
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (0. , -.957 , .587)],
            [1 , (0.2,  .757 , .487)]]
        mol.basis = '631g'
        mol.build()
        mf0 = mf = scf.RHF(mol).run(conv_tol=1.)
        mf = scf.addons.convert_to_uhf(mf)

        mycc0 = cc.CCSD(mf0)
        eris0 = mycc0.ao2mo()
        mycc0.kernel(eris=eris0)
        t1 = mycc0.t1
        t2 = mycc0.t2
        imds = ccsd_t_lambda.make_intermediates(mycc0, t1, t2, eris0)
        l1, l2 = ccsd_t_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
        dm1ref = ccsd_t_rdm.make_rdm1(mycc0, t1, t2, l1, l2, eris0)
        dm2ref = ccsd_t_rdm.make_rdm2(mycc0, t1, t2, l1, l2, eris0)

        t1 = (t1, t1)
        t2aa = t2 - t2.transpose(1,0,2,3)
        t2 = (t2aa, t2, t2aa)
        l1 = (l1, l1)
        l2aa = l2 - l2.transpose(1,0,2,3)
        l2 = (l2aa, l2, l2aa)
        mycc = cc.UCCSD(mf)
        eris = mycc.ao2mo()
        dm1 = uccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris)
        dm2 = uccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris)
        trdm1 = dm1[0] + dm1[1]
        trdm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
        self.assertAlmostEqual(abs(trdm1 - dm1ref).max(), 0, 12)
        self.assertAlmostEqual(abs(trdm2 - dm2ref).max(), 0, 12)

        ecc = mycc.kernel(eris=eris)[0]
        l1, l2 = mycc.solve_lambda(eris=eris)
        e3ref = mycc.e_tot + mycc.ccsd_t()

        nmoa, nmob = mycc.nmo
        eri_aa = ao2mo.kernel(mf._eri, mf.mo_coeff[0], compact=False).reshape([nmoa]*4)
        eri_bb = ao2mo.kernel(mf._eri, mf.mo_coeff[1], compact=False).reshape([nmob]*4)
        eri_ab = ao2mo.kernel(mf._eri, [mf.mo_coeff[k] for k in (0,0,1,1)],
                              compact=False).reshape(nmoa,nmoa,nmob,nmob)
        dm1 = uccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris)
        dm2 = uccsd_t_rdm.make_rdm2(mycc, t1, t2, l1, l2, eris=eris)
        h1a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
        h1b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
        e3 =(numpy.einsum('ij,ji->', h1a, dm1[0]) +
             numpy.einsum('ij,ji->', h1b, dm1[1]) +
             numpy.einsum('ijkl,ijkl->', eri_aa, dm2[0])*.5 +
             numpy.einsum('ijkl,ijkl->', eri_bb, dm2[2])*.5 +
             numpy.einsum('ijkl,ijkl->', eri_ab, dm2[1])    +
             mf.mol.energy_nuc())
        self.assertAlmostEqual(e3, e3ref, 9)

    def test_uccsd_t_gamma2_outcore(self):
        # the vvvv blocks in h5fobj, the (T) terms added; every block equals
        # _gamma2_intermediates
        mycc = cc.UCCSD(mf).run(conv_tol=1e-10)
        eris = mycc.ao2mo()
        conv, l1, l2 = uccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2, tol=1e-10)
        mycc.max_memory = 1
        for compress_vvvv in (False, True):
            ref = uccsd_t_rdm._gamma2_intermediates(mycc, mycc.t1, mycc.t2, l1, l2, eris,
                                                   compress_vvvv=compress_vvvv)
            f = lib.H5TmpFile()
            d2 = uccsd_t_rdm._gamma2_outcore(mycc, mycc.t1, mycc.t2, l1, l2, eris, f,
                                             compress_vvvv=compress_vvvv)
            for i, j in numpy.ndindex(len(ref), 4):
                with self.subTest(compress_vvvv=compress_vvvv, block=(i, j)):
                    if ref[i][j] is None:
                        self.assertIsNone(d2[i][j])
                    else:
                        self.assertAlmostEqual(abs(numpy.asarray(d2[i][j]) - ref[i][j]).max(), 0, 12)

    # The blocked (T) terms are compared with the whole-array code at one
    # virtual orbital per block, at three, and with every virtual in one block.
    def test_blocked_lambda_intermediates(self):
        mycc = cc.UCCSD(mf).run(conv_tol=1e-10, conv_tol_normt=1e-8)
        eris = mycc.ao2mo()
        imds = uccsd_t_lambda.make_intermediates(mycc, mycc.t1, mycc.t2, eris)
        ref = {'l1a': imds.l1a_t, 'l1b': imds.l1b_t, 'l2aa': imds.l2aa_t, 'l2ab': imds.l2ab_t, 'l2bb': imds.l2bb_t}
        for blksize in (1, 3, 100):
            got = uccsd_t_blocked.lambda_intermediates(mycc, mycc.t1, mycc.t2, eris, blksize)
            for name, x in zip(ref, got):
                with self.subTest(blksize=blksize, intermediate=name):
                    self.assertAlmostEqual(abs(x - ref[name]).max(), 0, 12)

    def test_blocked_gamma1_intermediates(self):
        mycc = cc.UCCSD(mf).run(conv_tol=1e-10, conv_tol_normt=1e-8)
        eris = mycc.ao2mo()
        ref = uccsd_t_rdm._t3_gamma1(mycc.t1, mycc.t2, eris)
        for blksize in (1, 3, 100):
            got = uccsd_t_blocked.gamma1_intermediates(mycc, mycc.t1, mycc.t2, eris, blksize)
            for name, r, x in zip(('goo', 'gOO', 'gvv', 'gVV', 'gvo', 'gVO'), ref, got):
                with self.subTest(blksize=blksize, intermediate=name):
                    self.assertAlmostEqual(abs(x - r).max(), 0, 12)

    def test_blocked_gamma2_intermediates(self):
        mycc = cc.UCCSD(mf).run(conv_tol=1e-10, conv_tol_normt=1e-8)
        eris = mycc.ao2mo()
        l1, l2 = mycc.solve_lambda(eris=eris)
        ref = uccsd_t_rdm._gamma2_intermediates(mycc, mycc.t1, mycc.t2, l1, l2, eris)
        for blksize in (1, 3, 100):
            d2 = uccsd_rdm._gamma2_intermediates(mycc, mycc.t1, mycc.t2, l1, l2)
            d2 = uccsd_t_blocked.add_gamma2(mycc, mycc.t1, mycc.t2, eris, d2, blksize)
            for i, j in numpy.ndindex(len(ref), 4):
                with self.subTest(blksize=blksize, block=(i, j)):
                    if ref[i][j] is None:
                        self.assertIsNone(d2[i][j])
                    else:
                        self.assertAlmostEqual(abs(d2[i][j] - ref[i][j]).max(), 0, 12)

    def test_blocked_path_with_little_memory(self):
        # with max_memory = 1 make_intermediates, _gamma1_intermediates and
        # _gamma2_intermediates take the blocked path; their results do not change
        mycc = cc.UCCSD(mf).run(conv_tol=1e-10, conv_tol_normt=1e-8)
        eris = mycc.ao2mo()
        t1, t2 = mycc.t1, mycc.t2
        l1, l2 = mycc.solve_lambda(eris=eris)
        imds = uccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
        ref_l = (imds.l1a_t, imds.l1b_t, imds.l2aa_t, imds.l2ab_t, imds.l2bb_t)
        ref_d1 = uccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        ref_d2 = uccsd_t_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2, eris)

        mycc.max_memory = 1
        imds = uccsd_t_lambda.make_intermediates(mycc, t1, t2, eris)
        d1 = uccsd_t_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2, eris, for_grad=True)
        d2 = uccsd_t_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        for k, x in enumerate((imds.l1a_t, imds.l1b_t, imds.l2aa_t, imds.l2ab_t, imds.l2bb_t)):
            with self.subTest(lambda_intermediate=k):
                self.assertAlmostEqual(abs(x - ref_l[k]).max(), 0, 12)
        for i, j in numpy.ndindex(4, 2):
            with self.subTest(d1_block=(i, j)):
                self.assertAlmostEqual(abs(d1[i][j] - ref_d1[i][j]).max(), 0, 12)
        for i, j in numpy.ndindex(len(ref_d2), 4):
            with self.subTest(d2_block=(i, j)):
                if ref_d2[i][j] is None:
                    self.assertIsNone(d2[i][j])
                else:
                    self.assertAlmostEqual(abs(d2[i][j] - ref_d2[i][j]).max(), 0, 12)

    def test_blocked_t3_intermediates_complex(self):
        # complex amplitudes and integrals: the conjugations of the blocked
        # gamma1 and gamma2 terms are those of the whole-array code. Any
        # complex input will do; only the two implementations are compared.
        numpy.random.seed(3)
        nocca, noccb, nvira, nvirb = 3, 2, 4, 5
        dims = {'o': nocca, 'O': noccb, 'v': nvira, 'V': nvirb}

        def rand(key):
            shape = [dims[x] for x in key]
            return (numpy.random.random(shape) - .5 + (numpy.random.random(shape) - .5) * 1j) * .1

        eris = cc.uccsd._ChemistsERIs()
        for key in ('ovvv', 'OVVV', 'ovVV', 'OVvv', 'ovoo', 'OVOO', 'ovOO', 'OVoo', 'ovov', 'OVOV', 'ovOV'):
            setattr(eris, key, rand(key))
        eris.focka = numpy.diag(numpy.arange(nocca + nvira, dtype=float)).astype(complex)
        eris.focka[nocca:, :nocca] = rand('vo')
        eris.fockb = numpy.diag(numpy.arange(noccb + nvirb, dtype=float)).astype(complex)
        eris.fockb[noccb:, :noccb] = rand('VO')
        eris.mo_energy = (eris.focka.diagonal().real, eris.fockb.diagonal().real)
        t1 = rand('ov'), rand('OV')
        t2 = rand('oovv'), rand('oOvV'), rand('OOVV')
        mycc = cc.UCCSD(scf.UHF(gto.M()))

        ref = uccsd_t_rdm._t3_gamma1(t1, t2, eris)
        got = uccsd_t_blocked.gamma1_intermediates(mycc, t1, t2, eris, 2)
        for name, r, x in zip(('goo', 'gOO', 'gvv', 'gVV', 'gvo', 'gVO'), ref, got):
            with self.subTest(intermediate=name):
                self.assertAlmostEqual(abs(x - r).max(), 0, 12)

        def zero_d2():
            def group(*keys):
                return tuple(numpy.zeros([dims[x] for x in key], dtype=complex) for key in keys)
            return ((group('ovov', 'ovOV', 'OVov', 'OVOV'),) + (None,) * 5
                    + (group('ovvv', 'ovVV', 'OVvv', 'OVVV'), group('ooov', 'ooOV', 'OOov', 'OOOV')))

        ref = uccsd_t_rdm._add_t3_gamma2(mycc, t1, t2, eris, zero_d2())
        got = uccsd_t_blocked.add_gamma2(mycc, t1, t2, eris, zero_d2(), 2)
        for i, j in ((0, 0), (0, 1), (0, 3), (6, 0), (6, 1), (6, 2), (6, 3), (7, 0), (7, 1), (7, 2), (7, 3)):
            with self.subTest(d2_block=(i, j)):
                self.assertAlmostEqual(abs(got[i][j] - ref[i][j]).max(), 0, 12)

if __name__ == "__main__":
    print("Full Tests for UCCSD(T)")
    unittest.main()
