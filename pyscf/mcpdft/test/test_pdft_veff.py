#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
#
import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf, ao2mo
from pyscf.lib import temporary_env
from pyscf.mcscf import newton_casscf, mc_ao2mo
from pyscf import mcpdft
from pyscf.mcpdft import pdft_veff
import unittest


def get_veff_ref(mc):
    nao, nmo = mc.mo_coeff.shape
    dm1s = np.asarray(mc.make_rdm1s())
    casdm1s = np.asarray(mc.fcisolver.make_rdm1s(mc.ci, mc.ncas, mc.nelecas))
    casdm1 = casdm1s.sum(0)
    casdm1s = mc.fcisolver.make_rdm1s(mc.ci, mc.ncas, mc.nelecas)
    casdm2 = mc.fcisolver.make_rdm2(mc.ci, mc.ncas, mc.nelecas)
    cascm2 = casdm2 - np.multiply.outer(casdm1, casdm1)
    cascm2 += np.einsum('sij,skl->ilkj', casdm1s, casdm1s)
    mo_cas = mc.mo_coeff[:, mc.ncore:][:, :mc.ncas]
    v1, v2_ao = pdft_veff.lazy_kernel(mc.otfnal, dm1s, cascm2, mo_cas)
    with temporary_env(mc._scf, _eri=ao2mo.restore(4, v2_ao, nao)):
        with temporary_env(mc.mol, incore_anyway=True):
            v2 = mc_ao2mo._ERIS(mc, mc.mo_coeff, method='incore')
    return v1, v2


def case(kv, mc):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nao, nmo = mc.mo_coeff.shape
    nocc, nvir = ncore + ncas, nmo - ncore - ncas
    ngorb = ncore * ncas + nocc * nvir
    fcasscf = mcscf.CASSCF(mc._scf, ncas, nelecas)
    fcasscf.__dict__.update(mc.__dict__)
    veff1, veff2 = mc.get_pdft_veff(mc.mo_coeff, mc.ci, incl_coul=True, paaa_only=True)
    with lib.temporary_env(fcasscf, get_hcore=lambda: mc.get_hcore() + veff1):
        g_all, _, _, hdiag_all = newton_casscf.gen_g_hop(
            fcasscf, mc.mo_coeff, mc.ci, veff2)
    g_numzero = np.abs(g_all) < 1e-8
    hdiag_all[g_numzero] = 1
    x0 = -g_all / hdiag_all
    xorb_norm = linalg.norm(x0[:ngorb])
    xci_norm = linalg.norm(x0[ngorb:])
    x0 = g_all * np.random.rand(*x0.shape) - 0.5
    x0[g_numzero] = 0
    x0[:ngorb] *= xorb_norm / linalg.norm(x0[:ngorb])
    x0[ngorb:] *= xci_norm / (linalg.norm(x0[ngorb:]) or 1)
    err_tab = np.zeros((0, 2))

    def seminum(x):
        uorb, ci1 = newton_casscf.extract_rotation(fcasscf, x, 1, mc.ci)
        mo1 = mc.rotate_mo(mc.mo_coeff, uorb)
        e1 = mc.energy_tot(mo_coeff=mo1, ci=ci1)[0]
        return e1 - mc.e_tot

    for ix, p in enumerate(range(20)):
        # For numerically unstable (i.e., translated) fnals,
        # it is somewhat difficult to find the convergence plateau
        # However, repeated calculations should show that
        # failure is rare and due only to numerical instability
        # and chance.
        x1 = x0 / (2 ** p)
        x1_norm = linalg.norm(x1)
        de_test = np.dot(g_all, x1)
        de_ref = seminum(x1)
        de_err = abs((de_test - de_ref) / de_ref)
        err_tab = np.append(err_tab, [[x1_norm, de_err]], axis=0)
        if ix > 0:
            conv_tab = err_tab[1:ix + 1, :] / err_tab[:ix, :]
        if ix > 1 and np.all(np.abs(conv_tab[-3:, -1] - 0.5) < 0.01) and abs(err_tab[-1, 1]) < 1e-3:
            break

    with kv.subTest(q='x'):
        kv.assertAlmostEqual(conv_tab[-1, 0], 0.5, 9)
    with kv.subTest(q='de'):
        kv.assertLess(abs(err_tab[-1, 1]), 1e-3)
        kv.assertAlmostEqual(conv_tab[-1, 1], 0.5, delta=0.01)


def setUpModule():
    global h2, lih
    h2 = scf.RHF(gto.M(atom='H 0 0 0; H 1.2 0 0', basis='6-31g',
                       output='/dev/null', verbose=0)).run()
    lih = scf.RHF(gto.M(atom='Li 0 0 0; H 1.2 0 0', basis='sto-3g',
                        output='/dev/null', verbose=0)).run()


def tearDownModule():
    global h2, lih
    h2.mol.stdout.close()
    lih.mol.stdout.close()
    del h2, lih


class KnownValues(unittest.TestCase):

    def test_de(self):
        np.random.seed(1)
        for mol, mf in zip(('H2', 'LiH'), (h2, lih)):
            for state, nel in zip(('Singlet', 'Triplet'), (2, (2, 0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE', 'tN12', 'ftN12', 'tM06L'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).run()
                    with self.subTest(mol=mol, state=state, fnal=fnal):
                        case(self, mc)


    def test_veff_ao2mo(self):
        for mol, mf in zip(('H2', 'LiH'), (h2, lih)):
            for state, nel in zip(('Singlet', 'Triplet'), (2, (2, 0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE', 'tN12', 'ftN12', 'tM06L'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).run()
                    v1_test, v2_test = mc.get_pdft_veff(jk_pc=True)
                    v1_ref, v2_ref = get_veff_ref(mc)
                    v_test = [v1_test, v2_test.vhf_c, v2_test.papa,
                              v2_test.ppaa, v2_test.j_pc, v2_test.k_pc]
                    v_ref = [v1_ref, v2_ref.vhf_c, v2_ref.papa, v2_ref.ppaa,
                             v2_ref.j_pc, v2_ref.k_pc]
                    terms = ['v1', 'v2.vhf_c', 'v2.papa', 'v2.ppaa', 'v2.j_pc',
                             'v2.k_pc']
                    for test, ref, term in zip(v_test, v_ref, terms):
                        with self.subTest(mol=mol, state=state, fnal=fnal,
                                          term=term):
                            self.assertAlmostEqual(lib.fp(test),
                                                   lib.fp(ref), delta=1e-4)


if __name__ == "__main__":
    print("Full Tests for MC-PDFT first fnal derivatives")
    unittest.main()
