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

from functools import reduce
import unittest
import copy
import numpy
import numpy as np

from pyscf import gto, lib
from pyscf import scf, dft
from pyscf import cc
from pyscf import ao2mo
from pyscf import mp
from pyscf.cc import rccsd

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
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = rccsd.RCCSD(mf).run(conv_tol=1e-10)


class KnownValues(unittest.TestCase):
    def test_roccsd(self):
        mf = scf.ROHF(mol).run()
        mycc = cc.RCCSD(mf).run()
        self.assertAlmostEqual(mycc.e_tot, -76.119346385357446, 7)

    def test_dump_chk(self):
        cc1 = copy.copy(mycc)
        cc1.nmo = mf.mo_energy.size
        cc1.nocc = mol.nelectron // 2
        cc1.dump_chk()
        cc1 = cc.CCSD(mf)
        cc1.__dict__.update(lib.chkfile.load(cc1._scf.chkfile, 'ccsd'))
        eris = cc1.ao2mo()
        e = cc1.energy(cc1.t1, cc1.t2, eris)
        self.assertAlmostEqual(e, -0.13539788638119823, 8)

    def test_ccsd_t(self):
        e = mycc.ccsd_t()
        self.assertAlmostEqual(e, -0.0009964234049929792, 10)

    def test_mbpt2(self):
        e = mycc.kernel(mbpt2=True)[0]
        #emp2 = mp.MP2(mf).kernel()[0]
        self.assertAlmostEqual(e, -0.12886859466216125, 10)

    def test_ao_direct(self):
        cc1 = cc.CCSD(mf)
        cc1.direct = True
        cc1.conv_tol = 1e-10
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13539788638119823, 8)

    def test_diis(self):
        cc1 = cc.CCSD(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13516622806104395, 8)

    def test_ERIS(self):
        cc1 = cc.RCCSD(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = cc.rccsd._make_eris_outcore(cc1, mo_coeff)

        self.assertAlmostEqual(lib.finger(numpy.array(eris.oooo)), 4.9638849382825754, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovoo)),-1.3623681896984081, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvo)), 133.4808352789826 , 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.oovv)), 55.123681017639655, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovov)), 125.81550684442149, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.ovvv)), 95.756230114113322, 12)
        self.assertAlmostEqual(lib.finger(numpy.array(eris.vvvv)),-10.450387490987545, 12)

    def test_amplitudes_to_vector(self):
        vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
        #self.assertAlmostEqual(lib.finger(vec), -0.056992042448099592, 6)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        self.assertAlmostEqual(abs(r1-mycc.t1).max(), 0, 14)
        self.assertAlmostEqual(abs(r2-mycc.t2).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        vec1 = mycc.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(vec-vec1).max(), 0, 14)

    def test_rccsd_frozen(self):
        cc1 = copy.copy(mycc)
        cc1.frozen = 1
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
        mycc2 = cc.ccsd.CCSD(mf)
        eris2 = mycc2.ao2mo()
        a = np.random.random((nmo,nmo)) * .1
        eris1.fock += a + a.T.conj()
        eris2.fock += a + a.T
        t1 = np.random.random((nocc,nvir)) * .1
        t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
        t2 = t2 + t2.transpose(1,0,3,2)

        t1b, t2b = cc.ccsd.update_amps(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.finger(t1b), -106360.5276951083, 6)
        self.assertAlmostEqual(lib.finger(t2b), 66540.100267798145, 6)

        mycc2.max_memory = 0
        t1a, t2a = cc.ccsd.update_amps(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(abs(t1a-t1b).max(), 0, 9)
        self.assertAlmostEqual(abs(t2a-t2b).max(), 0, 9)

        t2tril = cc.ccsd._add_vvvv_tril(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.finger(t2tril), 13306.139402693696, 8)

        Ht2 = cc.ccsd._add_vvvv_full(mycc2, t1, t2, eris2)
        self.assertAlmostEqual(lib.finger(Ht2), 760.50164232208408, 9)

        mycc1.cc2 = False
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.finger(t1a), -106360.5276951083, 7)
        self.assertAlmostEqual(lib.finger(t2a), 66540.100267798145, 6)
        self.assertAlmostEqual(abs(t1a-t1b).max(), 0, 6)
        self.assertAlmostEqual(abs(t2a-t2b).max(), 0, 6)
        mycc1.cc2 = True
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.finger(t1a), -106360.5276951083, 7)
        self.assertAlmostEqual(lib.finger(t2a), -1517.9391800662809, 7)

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
        mycc.direct = True
        eris2.vvvv = None
        eris2.mol = mol
        t2tril = cc.ccsd._add_vvvv_tril(mycc, t1, t2, eris2, with_ovvv=True)
        self.assertAlmostEqual(lib.finger(t2tril), 680.07199094501584, 9)
        t2tril = cc.ccsd._add_vvvv_tril(mycc, t1, t2, eris2, with_ovvv=False)
        self.assertAlmostEqual(lib.finger(t2tril), 446.56702664171348, 9)
        Ht2 = cc.ccsd._add_vvvv_full(mycc, t1, t2, eris2)
        self.assertAlmostEqual(lib.finger(Ht2), 48.122317842230686, 9)

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
        self.assertAlmostEqual(lib.finger(t1a), -13.32050019680894-1.8825765910430254j, 9)
        self.assertAlmostEqual(lib.finger(t2a), 9.2521062044785189+29.999480274811873j, 9)
        mycc1.cc2 = True
        t1a, t2a = rccsd.update_amps(mycc1, t1, t2, eris1)
        self.assertAlmostEqual(lib.finger(t1a), -13.32050019680894-1.8825765910430254j, 9)
        self.assertAlmostEqual(lib.finger(t2a), -0.056223856104895858+0.025472249329733986j, 9)


if __name__ == "__main__":
    print("Full Tests for RCCSD")
    unittest.main()

