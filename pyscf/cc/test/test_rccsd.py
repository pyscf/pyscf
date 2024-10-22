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

import tempfile
from functools import reduce
import unittest
import copy
import numpy
import numpy as np
import h5py

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import df
from pyscf import cc
from pyscf import ao2mo
from pyscf import mp
from pyscf.cc import ccsd
from pyscf.cc import rccsd

def setUpModule():
    global mol, mf, eris, mycc
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    mycc = rccsd.RCCSD(mf)
    mycc.conv_tol = 1e-10
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)

def tearDownModule():
    global mol, mf, eris, mycc
    mol.stdout.close()
    del mol, mf, eris, mycc


class KnownValues(unittest.TestCase):
    def test_roccsd(self):
        mf = scf.ROHF(mol).run()
        mycc = cc.RCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -76.119346385357446, 6)

    def test_density_fit_interface(self):
        mydf = df.DF(mol)
        mycc1 = ccsd.CCSD(mf).density_fit(auxbasis='ccpvdz-ri', with_df=mydf).run()
        self.assertAlmostEqual(mycc1.e_tot, -76.119348934346789, 6)

    def test_ERIS(self):
        mycc = rccsd.RCCSD(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = rccsd._make_eris_incore(mycc, mo_coeff)

        self.assertAlmostEqual(lib.fp(eris.oooo),  4.963884938282539, 11)
        self.assertAlmostEqual(lib.fp(eris.ovoo), -1.362368189698315, 11)
        self.assertAlmostEqual(lib.fp(eris.ovov),125.815506844421580, 11)
        self.assertAlmostEqual(lib.fp(eris.oovv), 55.123681017639463, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvo),133.480835278982620, 11)
        self.assertAlmostEqual(lib.fp(eris.ovvv), 95.756230114113222, 11)
        self.assertAlmostEqual(lib.fp(eris.vvvv),-10.450387490987071, 11)

        ccsd.MEMORYMIN, bak = 0, ccsd.MEMORYMIN
        mycc.max_memory = 0
        eris1 = mycc.ao2mo(mo_coeff)
        ccsd.MEMORYMIN = bak
        self.assertAlmostEqual(abs(numpy.array(eris1.oooo)-eris.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovoo)-eris.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovov)-eris.ovov).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.oovv)-eris.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovvo)-eris.ovvo).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.ovvv)-eris.ovvv).max(), 0, 11)
        self.assertAlmostEqual(abs(numpy.array(eris1.vvvv)-eris.vvvv).max(), 0, 11)

        # Testing the complex MO integrals
        def ao2mofn(mos):
            if isinstance(mos, numpy.ndarray) and mos.ndim == 2:
                mos = [mos]*4
            nmos = [mo.shape[1] for mo in mos]
            eri_mo = ao2mo.kernel(mf._eri, mos, compact=False).reshape(nmos)
            return eri_mo * 1j
        eris1 = rccsd._make_eris_incore(mycc, mo_coeff, ao2mofn=ao2mofn)
        self.assertAlmostEqual(abs(eris1.oooo.imag-eris.oooo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovoo.imag-eris.ovoo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovov.imag-eris.ovov).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.oovv.imag-eris.oovv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovvo.imag-eris.ovvo).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.ovvv.imag-eris.ovvv).max(), 0, 11)
        self.assertAlmostEqual(abs(eris1.vvvv.imag-eris.vvvv).max(), 0, 11)

    def test_dump_chk(self):
        cc1 = mycc.copy()
        cc1.nmo = mf.mo_energy.size
        cc1.nocc = mol.nelectron // 2
        cc1.dump_chk()
        cc1 = cc.CCSD(mf)
        cc1.__dict__.update(lib.chkfile.load(cc1.chkfile, 'ccsd'))
        e = cc1.energy(cc1.t1, cc1.t2, eris)
        self.assertAlmostEqual(e, -0.13539788638119823, 7)

    def test_ccsd_t(self):
        e = mycc.ccsd_t()
        self.assertAlmostEqual(e, -0.0009964234049929792, 8)

    def test_mbpt2(self):
        e = mycc.kernel(mbpt2=True)[0]
        #emp2 = mp.MP2(mf).kernel()[0]
        self.assertAlmostEqual(e, -0.12886859466216125, 8)

    def test_ao_direct(self):
        cc1 = cc.CCSD(mf)
        cc1.direct = True
        cc1.conv_tol = 1e-10
        cc1.kernel(t1=numpy.zeros_like(mycc.t1))
        self.assertAlmostEqual(cc1.e_corr, -0.13539788638119823, 7)

    def test_incore_complete(self):
        cc1 = cc.CCSD(mf)
        cc1.incore_complete = True
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13539788638119823, 7)

    def test_no_diis(self):
        cc1 = cc.CCSD(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13516622806104395, 7)

    def test_restart(self):
        ftmp = tempfile.NamedTemporaryFile()
        cc1 = cc.CCSD(mf)
        cc1.max_cycle = 5
        cc1.kernel()
        ref = cc1.e_corr

        adiis = lib.diis.DIIS(mol)
        adiis.filename = ftmp.name
        cc1.diis = adiis
        cc1.max_cycle = 3
        cc1.kernel(t1=None, t2=None)
        self.assertAlmostEqual(cc1.e_corr, -0.13529291367331436, 7)

        t1, t2 = cc1.vector_to_amplitudes(adiis.extrapolate())
        self.assertAlmostEqual(abs(t1-cc1.t1).max(), 0, 9)
        self.assertAlmostEqual(abs(t2-cc1.t2).max(), 0, 9)
        cc1.diis = None
        cc1.max_cycle = 1
        cc1.kernel(t1, t2)
        self.assertAlmostEqual(cc1.e_corr, -0.13535690694539226, 7)

        cc1.diis = adiis
        cc1.max_cycle = 2
        cc1.kernel(t1, t2)
        self.assertAlmostEqual(cc1.e_corr, ref, 8)

        cc2 = cc.CCSD(mf)
        cc2.restore_from_diis_(ftmp.name)
        self.assertAlmostEqual(abs(cc1.t1 - cc2.t1).max(), 0, 9)
        self.assertAlmostEqual(abs(cc1.t2 - cc2.t2).max(), 0, 9)

    def test_iterative_dampling(self):
        cc1 = cc.CCSD(mf)
        cc1.max_cycle = 3
        cc1.iterative_damping = 0.7
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13508743605375528, 7)

    def test_amplitudes_to_vector(self):
        vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
        #self.assertAlmostEqual(lib.fp(vec), -0.056992042448099592, 6)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        self.assertAlmostEqual(abs(r1-mycc.t1).max(), 0, 14)
        self.assertAlmostEqual(abs(r2-mycc.t2).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        vec1 = mycc.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(vec-vec1).max(), 0, 14)

    def test_vector_to_amplitudes_overwritten(self):
        mol = gto.M()
        mycc = scf.RHF(mol).apply(cc.CCSD)
        nelec = (3,3)
        nocc, nvir = nelec[0], 4
        nmo = nocc + nvir
        mycc.nocc = nocc
        mycc.nmo = nmo
        vec = numpy.zeros(mycc.vector_size())
        vec_orig = vec.copy()
        t1, t2 = mycc.vector_to_amplitudes(vec)
        t1[:] = 1
        t2[:] = 1
        self.assertAlmostEqual(abs(vec - vec_orig).max(), 0, 15)

    def test_vector_size(self):
        self.assertEqual(mycc.vector_size(), 860)

    def test_rccsd_frozen(self):
        cc1 = mycc.copy()
        cc1.frozen = 1
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.set_frozen()
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [0,1]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 3)
        cc1.frozen = [1,9]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [9,10,12]
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 5)
        cc1.nmo = 10
        cc1.nocc = 6
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 6)

    def test_update_amps(self):
        mol = gto.M()
        nocc, nvir = 5, 12
        nmo = nocc + nvir
        nmo_pair = nmo*(nmo+1)//2
        mf = scf.RHF(mol)
        np.random.seed(12)
        mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
        mf.mo_coeff = np.random.random((nmo,nmo))
        mf.mo_energy = np.arange(0., nmo)
        mf.mo_occ = np.zeros(nmo)
        mf.mo_occ[:nocc] = 2
        vhf = mf.get_veff(mol, mf.make_rdm1())
        cinv = np.linalg.inv(mf.mo_coeff)
        mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)

        mycc1 = rccsd.RCCSD(mf)
        eris1 = mycc1.ao2mo()
        mycc2 = ccsd.CCSD(mf)
        eris2 = mycc2.ao2mo()
        a = np.random.random((nmo,nmo)) * .1
        eris1.fock += a + a.T.conj()
        eris2.fock += a + a.T
        t1 = np.random.random((nocc,nvir)) * .1
        t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
        t2 = t2 + t2.transpose(1,0,3,2)

        t1b, t2b = ccsd.update_amps(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.fp(t1b), -106360.5276951083, 6)
        self.assertAlmostEqual(lib.fp(t2b), 66540.100267798145, 6)

        mycc2.max_memory = 0
        t1a, t2a = ccsd.update_amps(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(abs(t1a-t1b).max(), 0, 9)
        self.assertAlmostEqual(abs(t2a-t2b).max(), 0, 9)

        t2tril = ccsd._add_vvvv_tril(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.fp(t2tril), 13306.139402693696, 8)

        Ht2 = ccsd._add_vvvv_full(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.fp(Ht2), 760.50164232208408, 9)

        mycc1.cc2 = False
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.fp(t1a), -106360.5276951083, 7)
        self.assertAlmostEqual(lib.fp(t2a), 66540.100267798145, 6)
        self.assertAlmostEqual(abs(t1a-t1b).max(), 0, 6)
        self.assertAlmostEqual(abs(t2a-t2b).max(), 0, 6)
        mycc1.cc2 = True
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.fp(t1a), -106360.5276951083, 7)
        self.assertAlmostEqual(lib.fp(t2a), -1517.9391800662809, 7)

        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [
            [8 , (0. , 0.     , 0.)],
            [1 , (1. , 0.     , 0.)],
            [1 , (0. , -0.757 , 0.587)],
            [1 , (0. , 0.757  , 0.587)]]
        mol.basis = {'H': 'sto3g',
                     'O': 'cc-pvdz',}
        mol.charge = 1
        mol.build(0, 0)
        mycc2.direct = True
        eris2.vvvv = None
        eris2.mol = mol
        mycc2.mo_coeff, eris2.mo_coeff = eris2.mo_coeff, None
        t2tril = ccsd._add_vvvv_tril(mycc2, t1, t2, eris2, with_ovvv=True)
        self.assertAlmostEqual(lib.fp(t2tril), 680.07199094501584, 9)
        t2tril = ccsd._add_vvvv_tril(mycc2, t1, t2, eris2, with_ovvv=False)
        self.assertAlmostEqual(lib.fp(t2tril), 446.56702664171348, 9)
        Ht2 = ccsd._add_vvvv_full(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.fp(Ht2), 48.122317842230686, 9)

        eri1 = np.random.random((nmo,nmo,nmo,nmo)) + np.random.random((nmo,nmo,nmo,nmo))*1j
        eri1 = eri1.transpose(0,2,1,3)
        eri1 = eri1 + eri1.transpose(1,0,3,2).conj()
        eri1 = eri1 + eri1.transpose(2,3,0,1)
        eri1 *= .1
        eris1.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        eris1.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        eris1.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        eris1.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        eris1.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        eris1.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        eris1.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        a = np.random.random((nmo,nmo)) * .1j
        eris1.fock = eris1.fock + a + a.T.conj()

        t1 = t1 + np.random.random((nocc,nvir)) * .1j
        t2 = t2 + np.random.random((nocc,nocc,nvir,nvir)) * .1j
        t2 = t2 + t2.transpose(1,0,3,2)
        mycc1.cc2 = False
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.fp(t1a), -13.32050019680894-1.8825765910430254j, 9)
        self.assertAlmostEqual(lib.fp(t2a), 9.2521062044785189+29.999480274811873j, 9)
        mycc1.cc2 = True
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.fp(t1a), -13.32050019680894-1.8825765910430254j, 9)
        self.assertAlmostEqual(lib.fp(t2a), -0.056223856104895858+0.025472249329733986j, 9)

    def test_eris_contract_vvvv_t2(self):
        mol = gto.Mole()
        nocc, nvir = 5, 12
        nvir_pair = nvir*(nvir+1)//2
        numpy.random.seed(9)
        t2 = numpy.random.random((nocc,nocc,nvir,nvir)) - .5
        t2 = t2 + t2.transpose(1,0,3,2)
        eris = ccsd._ChemistsERIs()
        vvvv = numpy.random.random((nvir_pair,nvir_pair)) - .5
        eris.vvvv = vvvv + vvvv.T
        eris.mol = mol
        mycc.max_memory, bak = 0, mycc.max_memory
        vt2 = eris._contract_vvvv_t2(mycc, t2, eris.vvvv)
        mycc.max_memory = bak
        self.assertAlmostEqual(lib.fp(vt2), -39.572579908080087, 11)
        vvvv = ao2mo.restore(1, eris.vvvv, nvir)
        ref = lib.einsum('acbd,ijcd->ijab', vvvv, t2)
        self.assertAlmostEqual(abs(vt2 - ref).max(), 0, 11)

        # _contract_s1vvvv_t2, testing complex and real mixed contraction
        vvvv =(numpy.random.random((nvir,nvir,nvir,nvir)) +
               numpy.random.random((nvir,nvir,nvir,nvir))*1j - (.5+.5j))
        vvvv = vvvv + vvvv.transpose(1,0,3,2).conj()
        vvvv = vvvv + vvvv.transpose(2,3,0,1)
        eris.vvvv = vvvv
        eris.mol = mol
        mycc.max_memory, bak = 0, mycc.max_memory
        vt2 = eris._contract_vvvv_t2(mycc, t2, eris.vvvv)
        mycc.max_memory = bak
        self.assertAlmostEqual(lib.fp(vt2), 23.502736435296871+113.90422480013488j, 11)
        ref = lib.einsum('acbd,ijcd->ijab', eris.vvvv, t2)
        self.assertAlmostEqual(abs(vt2 - ref).max(), 0, 11)

    def test_add_vvvv(self):
        t1 = mycc.t1
        t2 = mycc.t2
        nocc, nvir = t1.shape
        tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        eris1 = copy.copy(eris)
        mycc1 = mycc.copy()
        ovvv = eris1.get_ovvv()
        tmp = -numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, ovvv)
        t2a = tmp + tmp.transpose(1,0,3,2)
        t2a += mycc1._add_vvvv(t1, t2, eris1)
        mycc1.direct = True
        eris1.vvvv = None  # == with_ovvv=True in the call below
        t2b = mycc1._add_vvvv(t1, t2, eris1, t2sym='jiba')
        self.assertAlmostEqual(abs(t2a-t2b).max(), 0, 12)

    def test_diagnostic(self):
        t1_diag = mycc.get_t1_diagnostic()
        d1_diag = mycc.get_d1_diagnostic()
        d2_diag = mycc.get_d2_diagnostic()
        self.assertAlmostEqual(t1_diag, 0.006002754773812036, 6)
        self.assertAlmostEqual(d1_diag, 0.012738043220198926, 6)
        self.assertAlmostEqual(d2_diag, 0.1169239107130769, 6)

    def test_ao2mo(self):
        mycc = ccsd.CCSD(mf)
        numpy.random.seed(2)
        mo = numpy.random.random(mf.mo_coeff.shape)
        mycc.max_memory = 2000
        eri_incore = mycc.ao2mo(mo)
        mycc.max_memory = 0
        eri_outcore = mycc.ao2mo(mo)
        self.assertTrue(isinstance(eri_outcore.oovv, h5py.Dataset))
        self.assertAlmostEqual(abs(eri_incore.oooo - eri_outcore.oooo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.oovv - eri_outcore.oovv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovoo - eri_outcore.ovoo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovvo - eri_outcore.ovvo).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovov - eri_outcore.ovov).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.ovvv - eri_outcore.ovvv).max(), 0, 12)
        self.assertAlmostEqual(abs(eri_incore.vvvv - eri_outcore.vvvv).max(), 0, 12)

        mycc1 = ccsd.CCSD(mf.density_fit(auxbasis='ccpvdz-ri'))
        mycc1.max_memory = 0
        eri_df = mycc1.ao2mo(mo)
        self.assertAlmostEqual(lib.fp(eri_df.oooo), -493.98003157749906, 9)
        self.assertAlmostEqual(lib.fp(eri_df.oovv), -91.84858398271658 , 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovoo), -203.89515661847437, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvo), -14.883877359169205, 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovov), -57.62195194777554 , 9)
        self.assertAlmostEqual(lib.fp(eri_df.ovvv), -24.359418953533535, 9)
        self.assertAlmostEqual(lib.fp(eri_df.vvvv),  76.9017539373456  , 9)

if __name__ == "__main__":
    print("Full Tests for RCCSD")
    unittest.main()
