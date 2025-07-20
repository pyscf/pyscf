#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import numpy as np
from pyscf import gto, scf, ao2mo, lib, mcscf
from pyscf.lib import temporary_env
from pyscf.mcscf import mc_ao2mo, newton_casscf
from pyscf.mcpdft import pdft_feff, _dms
from pyscf import mcpdft
import unittest


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


def get_feff_ref(mc, state=0, dm1s=None, cascm2=None, c_dm1s=None, c_cascm2=None):
    nao = mc.mo_coeff.shape[0]
    if dm1s is None or cascm2 is None:
        t_casdm1s = mc.make_one_casdm1s(mc.ci, state=state)
        t_dm1s = _dms.casdm1s_to_dm1s(mc, t_casdm1s)
        t_casdm2 = mc.make_one_casdm2(mc.ci, state=state)
        t_cascm2 = _dms.dm2_cumulant(t_casdm2, t_casdm1s)
        if dm1s is None:
            dm1s = t_dm1s

        if cascm2 is None:
            cascm2 = t_cascm2

    mo_cas = mc.mo_coeff[:, mc.ncore:][:, :mc.ncas]
    if c_dm1s is None:
        c_dm1s = dm1s

    if c_cascm2 is None:
        c_cascm2 = cascm2

    v1, v2_ao = pdft_feff.lazy_kernel(mc.otfnal, dm1s, cascm2, c_dm1s,
                                      c_cascm2, mo_cas)
    with temporary_env(mc._scf, _eri=ao2mo.restore(4, v2_ao, nao)):
        with temporary_env(mc.mol, incore_anyway=True):
            v2 = mc_ao2mo._ERIS(mc, mc.mo_coeff, method='incore')

    return v1, v2


def contract_veff(mc, mo_coeff, ci, veff1, veff2, ncore=None, ncas=None):
    if ncore is None:
        ncore = mc.ncore
    if ncas is None:
        ncas = mc.ncas

    nocc = ncore + ncas

    casdm1s = mc.make_one_casdm1s(ci)
    casdm1 = casdm1s[0] + casdm1s[1]
    casdm2 = mc.make_one_casdm2(ci)

    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s, mo_coeff=mo_coeff)
    dm1 = dm1s[0] + dm1s[1]

    ref_e = np.tensordot(veff1, dm1)
    ref_e += veff2.energy_core
    ref_e += np.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1)
    ref_e += 0.5 * np.tensordot(veff2.papa[ncore:nocc, :, ncore:nocc, :],
                                casdm2, axes=4)
    return ref_e


def case(kv, mc):
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nmo = mc.mo_coeff.shape[1]
    nocc, nvir = ncore + ncas, nmo - ncore - ncas
    ngorb = ncore * ncas + nocc * nvir
    fcasscf = mcscf.CASSCF(mc._scf, ncas, nelecas)
    fcasscf.__dict__.update(mc.__dict__)

    # What we will do is numerically check the differentiation of V_pq D_pq +
    # 1/2 v_pqrs d_pqrs with respect to MO and CI rotations. This manifests
    # itself as the generalized Fock matrix elements generated from V_pq and
    # v_pqrs as well as then the generalized fock matrix elements from [rho
    # F]_pq and [rho F]_pqrs. We are then numerically check how the contract is
    # by evaluating V_pq D_pq + ... at the reference and the slightly modified
    # CI/MO parameters

    feff1, feff2 = mc.get_pdft_feff(mc.mo_coeff, mc.ci, incl_coul=False, paaa_only=True, jk_pc=True)
    veff1, veff2 = mc.get_pdft_veff(mc.mo_coeff, mc.ci, incl_coul=False, paaa_only=False)
    ref_c_veff = contract_veff(mc, mc.mo_coeff, mc.ci, veff1, veff2)

    with lib.temporary_env(fcasscf, get_hcore=lambda: feff1):
        g_feff, _, _, hdiag_feff = newton_casscf.gen_g_hop(fcasscf,
                                                           mc.mo_coeff, mc.ci,
                                                           feff2)

    with lib.temporary_env(fcasscf, get_hcore=lambda: veff1):
        g_veff, _, _, hdiag_veff = newton_casscf.gen_g_hop(fcasscf,
                                                           mc.mo_coeff, mc.ci,
                                                           veff2)

    g_all = g_feff + g_veff
    hdiag_all = hdiag_feff + hdiag_veff
    g_numzero = np.abs(g_all) < 1e-8
    hdiag_all[g_numzero] = 1
    x0 = -g_all / hdiag_all
    xorb_norm = np.linalg.norm(x0[:ngorb])
    xci_norm = np.linalg.norm(x0[ngorb:])
    x0 = g_all * np.random.rand(*x0.shape) - 0.5
    x0[g_numzero] = 0
    x0[:ngorb] *= xorb_norm / np.linalg.norm(x0[:ngorb])
    x0[ngorb:] *= xci_norm / (np.linalg.norm(x0[ngorb:]) or 1)
    err_tab = np.zeros((0, 2))

    def seminum(x):
        uorb, ci1 = newton_casscf.extract_rotation(fcasscf, x, 1, mc.ci)
        mo1 = mc.rotate_mo(mc.mo_coeff, uorb)
        veff1_1, veff2_1 = mc.get_pdft_veff(mo=mo1, ci=ci1, incl_coul=False)
        semi_num_c_veff = contract_veff(mc, mo1, ci1, veff1_1, veff2_1)
        return semi_num_c_veff - ref_c_veff

    for ix, p in enumerate(range(30)):
        x1 = x0 / (2 ** p)
        x1_norm = np.linalg.norm(x1)
        dg_test = np.dot(g_all, x1)
        dg_ref = seminum(x1)
        dg_err = abs((dg_test - dg_ref) / dg_ref)
        err_tab = np.append(err_tab, [[x1_norm, dg_err]], axis=0)
        if ix > 0:
            conv_tab = err_tab[1:ix + 1, :] / err_tab[:ix, :]

        if ix > 1 and np.all(np.abs(conv_tab[-3:, -1] - 0.5) < 0.01) and abs(err_tab[-1, 1]) < 1e-3:
            break

    with kv.subTest(q='x'):
        kv.assertAlmostEqual(conv_tab[-1, 0], 0.5, 9)

    with kv.subTest(q='de'):
        kv.assertLess(abs(err_tab[-1, 1]), 1e-3)
        kv.assertAlmostEqual(conv_tab[-1, 1], 0.5, delta=0.05)


class KnownValues(unittest.TestCase):

    def test_dvot(self):
        np.random.seed(1)
        for mol, mf in zip(("H2", "LiH"), (h2, lih)):
            for state, nel in zip(('Singlet', 'Triplet'), (2, (2, 0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).run()
                    with self.subTest(mol=mol, state=state, fnal=fnal):
                        case(self, mc)

    def test_feff_ao2mo(self):
        for mol, mf in zip(("H2", "LiH"), (h2, lih)):
            for state, nel in zip(('Singlet', 'Triplet'), (2, (2, 0))):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).run()
                    f1_test, f2_test = mc.get_pdft_feff(jk_pc=True)
                    f1_ref, f2_ref = get_feff_ref(mc)
                    f_test = [f1_test, f2_test.vhf_c, f2_test.papa,
                              f2_test.ppaa, f2_test.j_pc, f2_test.k_pc]
                    f_ref = [f1_ref, f2_ref.vhf_c, f2_ref.papa, f2_ref.ppaa,
                             f2_ref.j_pc, f2_ref.k_pc]
                    terms = ['f1', 'f2.vhf_c', 'f2.papa', 'f2.ppaa', 'f2.j_pc',
                             'f2.k_pc']
                    for test, ref, term in zip(f_test, f_ref, terms):
                        with self.subTest(mol=mol, state=state, fnal=fnal,
                                          term=term):
                            self.assertAlmostEqual(lib.fp(test),
                                                   lib.fp(ref), delta=1e-4)

    def test_sa_contract_feff_ao2mo(self):
        for mol, mf in zip(("H2", "LiH"), (h2, lih)):
            for state, nel in zip(['Singlet'], [2]):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel,
                                       grids_level=1).state_average_([0.5,
                                                                      0.5]).run()

                    sa_casdm1s = _dms.make_weighted_casdm1s(mc)
                    sa_casdm2 = _dms.make_weighted_casdm2(mc)
                    sa_dm1s = _dms.casdm1s_to_dm1s(mc, sa_casdm1s)
                    sa_cascm2 = _dms.dm2_cumulant(sa_casdm2, sa_casdm1s)

                    f1_test, f2_test = mc.get_pdft_feff(jk_pc=True,
                                                        c_dm1s=sa_dm1s,
                                                        c_cascm2=sa_cascm2)
                    f1_ref, f2_ref = get_feff_ref(mc, c_dm1s=sa_dm1s,
                                                  c_cascm2=sa_cascm2)
                    f_test = [f1_test, f2_test.vhf_c, f2_test.papa,
                              f2_test.ppaa, f2_test.j_pc, f2_test.k_pc]
                    f_ref = [f1_ref, f2_ref.vhf_c, f2_ref.papa, f2_ref.ppaa,
                             f2_ref.j_pc, f2_ref.k_pc]
                    terms = ['f1', 'f2.vhf_c', 'f2.papa', 'f2.ppaa', 'f2.j_pc',
                             'f2.k_pc']
                    for test, ref, term in zip(f_test, f_ref, terms):
                        with self.subTest(mol=mol, state=state, fnal=fnal,
                                          term=term):
                            self.assertAlmostEqual(lib.fp(test), lib.fp(ref),
                                                   delta=1e-6)

    def test_delta_contract_feff_ao2mo(self):
        for mol, mf in zip(("H2", "LiH"), (h2, lih)):
            for state, nel in zip(['Singlet'], [2]):
                for fnal in ('tLDA,VWN3', 'ftLDA,VWN3', 'tPBE', 'ftPBE'):
                    mc = mcpdft.CASSCF(mf, fnal, 2, nel, grids_level=1).state_average_([0.5, 0.5]).run()

                    sa_casdm1s = _dms.make_weighted_casdm1s(mc)
                    sa_casdm2 = _dms.make_weighted_casdm2(mc)

                    casdm1s = mc.make_one_casdm1s(ci=mc.ci)
                    casdm2 = mc.make_one_casdm2(ci=mc.ci)
                    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s)
                    cascm2 = _dms.dm2_cumulant(casdm2, casdm1s)

                    f1_test, f2_test = mc.get_pdft_feff(jk_pc=True, casdm1s=sa_casdm1s, casdm2=sa_casdm2, c_dm1s=dm1s,
                                                        c_cascm2=cascm2, delta=True)

                    f1_ref, f2_ref = mc.get_pdft_feff(jk_pc=True, casdm1s=sa_casdm1s, casdm2=sa_casdm2, c_dm1s=dm1s,
                                                      c_cascm2=cascm2)
                    f1_sa, f2_sa = mc.get_pdft_feff(jk_pc=True, casdm1s=sa_casdm1s, casdm2=sa_casdm2)

                    f1_ref -= f1_sa
                    f2_ref.vhf_c -= f2_sa.vhf_c
                    f2_ref.papa -= f2_sa.papa
                    f2_ref.ppaa -= f2_sa.ppaa
                    f2_ref.j_pc -= f2_sa.j_pc
                    f2_ref.k_pc -= f2_sa.k_pc

                    f_test = [f1_test, f2_test.vhf_c, f2_test.papa,
                              f2_test.ppaa, f2_test.j_pc, f2_test.k_pc]
                    f_ref = [f1_ref, f2_ref.vhf_c, f2_ref.papa, f2_ref.ppaa,
                             f2_ref.j_pc, f2_ref.k_pc]
                    terms = ['f1', 'f2.vhf_c', 'f2.papa', 'f2.ppaa', 'f2.j_pc',
                             'f2.k_pc']
                    for test, ref, term in zip(f_test, f_ref, terms):
                        with self.subTest(mol=mol, state=state, fnal=fnal,
                                          term=term):
                            self.assertAlmostEqual(lib.fp(test), lib.fp(ref),
                                                   delta=1e-6)


if __name__ == "__main__":
    print("Full Tests for pdft_feff")
    unittest.main()
