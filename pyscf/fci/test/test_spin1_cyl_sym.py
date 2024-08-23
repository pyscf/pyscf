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
import numpy as np
from pyscf import gto, lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring, direct_spin1, direct_spin1_symm
from pyscf.fci import direct_nosym
from pyscf.fci import direct_spin1_cyl_sym
from pyscf import mcscf
from pyscf.symm.basis import linearmole_irrep2momentum

class KnownValues(unittest.TestCase):
    def test_contract_2e(self):
        mol = gto.M(
            atom = 'Li 0 0 0; Li 0 0 2.',
            basis = {'Li': [[0, [4.5, 1]], [2, [0.5, 1]]]},
            spin=0,
            symmetry = True,
        )
        mf = mol.RHF().run()
        norb = mf.mo_coeff.shape[1]
        nelec = mol.nelec
        h1e = mf.mo_coeff.T.dot(scf.hf.get_hcore(mol)).dot(mf.mo_coeff)
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
        orbsym = mf.orbsym

        degen_mapping = direct_spin1_cyl_sym.map_degeneracy(h1e.diagonal(), orbsym)
        orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)
        u = direct_spin1_cyl_sym._cyl_sym_orbital_rotation(orbsym, degen_mapping)

        mo = mf.mo_coeff.dot(u.conj().T)
        h1e = mo.conj().T.dot(mf.get_hcore()).dot(mo)
        eri = mol.intor('int2e_sph').reshape([norb]*4)
        eri = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo.conj(), mo, mo.conj(), mo)
        h1e = h1e.real.copy()
        g2e = eri.real.copy()

        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
        airreps_d2h = birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
        a_ls = b_ls = direct_spin1_cyl_sym._strs_angular_momentum(strsa, orbsym)
        if neleca != nelecb:
            strsb = cistring.gen_strings4orblist(range(norb), nelecb)
            birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsb, orbsym)
            b_ls = direct_spin1_cyl_sym._strs_angular_momentum(strsb, orbsym)
        a_ungerade = airreps_d2h >= 4
        b_ungerade = birreps_d2h >= 4
        na = len(strsa)
        nb = len(strsb)

        def check(wfnsym):
            wfn_momentum = linearmole_irrep2momentum(wfnsym)
            wfn_ungerade = (wfnsym % 10) >= 4
            np.random.seed(15)
            ci0 = np.random.random((na,nb))
            sym_allowed = a_ungerade[:,None] == b_ungerade ^ wfn_ungerade
            # total angular momentum == wfn_momentum
            sym_allowed &= a_ls[:,None] == wfn_momentum - b_ls
            ci0[~sym_allowed] = 0
            ci1ref = direct_nosym.contract_2e(g2e, ci0, norb, nelec)
            ci1 = direct_spin1_cyl_sym.contract_2e(g2e, ci0, norb, nelec,
                                                   orbsym=orbsym, wfnsym=wfnsym)
            self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)

        check(0)
        check(10)
        check(15)
        check(14)
        check(7)
        check(3)
        check(5)

    def test_contract_2e_1(self):
        mol = gto.M(
            atom = 'Li 0 0 0; Li 0 0 2.',
            basis = {'Li': [[0, [4.5, 1]], [1, [0.5, 1]]]},
            spin=0,
            symmetry = True,
        )
        mf = mol.RHF().run()
        norb = mf.mo_coeff.shape[1]
        nelec = mol.nelec
        h1e = mf.mo_coeff.T.dot(scf.hf.get_hcore(mol)).dot(mf.mo_coeff)
        eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
        orbsym = mf.orbsym

        degen_mapping = direct_spin1_cyl_sym.map_degeneracy(h1e.diagonal(), orbsym)
        orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)
        u = direct_spin1_cyl_sym._cyl_sym_orbital_rotation(orbsym, degen_mapping)

        mo = mf.mo_coeff.dot(u.conj().T)
        h1e = mo.conj().T.dot(mf.get_hcore()).dot(mo)
        eri = mol.intor('int2e_sph').reshape([norb]*4)
        eri = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, mo.conj(), mo, mo.conj(), mo)
        h1e = h1e.real.copy()
        g2e = eri.real.copy()

        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
        airreps_d2h = birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
        a_ls = b_ls = direct_spin1_cyl_sym._strs_angular_momentum(strsa, orbsym)
        if neleca != nelecb:
            strsb = cistring.gen_strings4orblist(range(norb), nelecb)
            birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsb, orbsym)
            b_ls = direct_spin1_cyl_sym._strs_angular_momentum(strsb, orbsym)
        a_ungerade = airreps_d2h >= 4
        b_ungerade = birreps_d2h >= 4
        na = len(strsa)
        nb = len(strsb)

        def check(wfnsym):
            wfn_momentum = linearmole_irrep2momentum(wfnsym)
            wfn_ungerade = (wfnsym % 10) >= 4
            np.random.seed(15)
            ci0 = np.random.random((na,nb))
            sym_allowed = a_ungerade[:,None] == b_ungerade ^ wfn_ungerade
            # total angular momentum == wfn_momentum
            sym_allowed &= a_ls[:,None] == wfn_momentum - b_ls
            ci0[~sym_allowed] = 0
            ci1ref = direct_nosym.contract_2e(g2e, ci0, norb, nelec)
            ci1 = direct_spin1_cyl_sym.contract_2e(g2e, ci0, norb, nelec,
                                                   orbsym=orbsym, wfnsym=wfnsym)
            self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)

        check(0)
        check(11)
        check(6)
        check(3)
        check(5)

    def test_spin1_cyl_sym(self):
        mol = gto.M(
            atom = 'N 0 0 0; N 0 0 1.5',
            basis = 'cc-pVDZ',
            spin = 0,
            symmetry = True,
        )
        mc = mol.RHF().run().CASCI(12, 6)
        mc.fcisolver.wfnsym = 'E1ux'
        mc.run()
        e1 = mc.e_tot
        ci1 = mc.ci
        self.assertAlmostEqual(e1, -108.683383569227, 7)

        mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
        mc.fcisolver.wfnsym = 'E1ux'
        mc.fcisolver.davidson_only = True
        mc.run()
        e2 = mc.e_tot
        self.assertAlmostEqual(e2, -108.683383569227, 7)
        orbsym = mc.fcisolver.orbsym
        degen_mapping = orbsym.degen_mapping
        u = direct_spin1_symm._cyl_sym_orbital_rotation(orbsym, degen_mapping)
        ci2 = fci.addons.transform_ci(mc.ci, (3,3), u)
        ci2 = ci2.real / np.linalg.norm(ci2.real)
        self.assertAlmostEqual(abs(ci1.ravel().dot(ci2.ravel())), 1, 6)

    def test_wrong_initial_guess(self):
        mol = gto.M(
            atom = 'H 0 0 0; H 0 0 1.2',
            basis = [[0, [3, 1]], [1, [1, 1]]],
            spin = 1,
            charge = 1,
            symmetry = True)
        mf = mol.RHF().run()
        mc = mcscf.CASCI(mf, mf.mo_energy.size, mol.nelec)
        mc.fcisolver.wfnsym = 'A2g'
        self.assertRaises(RuntimeError, mc.run)

        mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
        mc.fcisolver.wfnsym = 'A2g'
        self.assertRaises(RuntimeError, mc.run)

    def test_linearmole_a2(self):
        mol = gto.M(
            atom = 'H 0 0 0; H 0 0 1.2',
            basis = [[0, [3, 1]], [1, [1, 1]]],
            symmetry = True)
        mf = mol.RHF().run()

        mc = mcscf.CASCI(mf, mf.mo_energy.size, mol.nelec)
        mc.fcisolver.wfnsym = 'A2g'
        mc.run()
        self.assertAlmostEqual(mc.e_tot, 2.6561956585409616, 8)
        mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
        mc.fcisolver.wfnsym = 'A2g'
        mc.run()
        self.assertAlmostEqual(mc.e_tot, 2.6561956585409616, 8)

        mc = mcscf.CASCI(mf, mf.mo_energy.size, mol.nelec)
        mc.fcisolver.wfnsym = 'A2u'
        mc.run()
        self.assertAlmostEqual(mc.e_tot, 2.8999951068356475, 8)
        mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
        mc.fcisolver.wfnsym = 'A2u'
        mc.run()
        self.assertAlmostEqual(mc.e_tot, 2.8999951068356475, 8)

    def test_incomplete_orbsym(self):
        sol = direct_spin1_cyl_sym.FCI(gto.Mole())
        no, ne = 2, 2
        h1 = np.ones((no,no))
        h2 = np.ones((no,no,no,no))
        orbsym = lib.tag_array(np.array([0,3]), degen_mapping=[0,2])
        with self.assertRaises(lib.exceptions.PointGroupSymmetryError):
            sol.kernel(h1, h2, no, ne, orbsym=orbsym)

    # issue 2291
    def test_triplet_degeneracy(self):
        mol = gto.M(atom='O; O 1 1.2', basis='631g', spin=2, symmetry=1)
        mf = mol.RHF().run()

        def casci(nelec):
            norb = 4
            mc = mcscf.CASCI(mf, norb, nelec)
            mc.fcisolver = direct_spin1_cyl_sym.FCI(mol)
            mc.fcisolver.wfnsym = 'A2g'
            mc.kernel()
            return mc.e_tot
        self.assertAlmostEqual(casci((4, 2)), -149.56827649707, 9)
        self.assertAlmostEqual(casci((3, 3)), -149.56827649707, 9)
        self.assertAlmostEqual(casci((2, 4)), -149.56827649707, 9)

if __name__ == "__main__":
    print("Full Tests for spin1-symm")
    unittest.main()
