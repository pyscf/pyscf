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

import math
import unittest
import numpy
from pyscf import gto, lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring, direct_spin1, direct_spin1_symm
from pyscf import mcscf

def setUpModule():
    global mol, m, h1e, g2e, ci0, cis
    global norb, nelec, orbsym
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
        O    0.  0.      0.
        H    0.  -0.757  0.587
        H    0.  0.757   0.587'''
    mol.basis = 'sto-3g'
    mol.symmetry = 1
    mol.build()
    m = scf.RHF(mol)
    m.conv_tol = 1e-15
    ehf = m.scf()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelectron
    h1e = m.mo_coeff.T.dot(scf.hf.get_hcore(mol)).dot(m.mo_coeff)
    g2e = ao2mo.incore.full(m._eri, m.mo_coeff)
    orbsym = m.orbsym
    cis = direct_spin1_symm.FCISolver(mol)
    cis.orbsym = orbsym

    numpy.random.seed(15)
    na = cistring.num_strings(norb, nelec//2)
    ci0 = numpy.random.random((na,na))

def tearDownModule():
    global mol, m, h1e, g2e, ci0, cis
    del mol, m, h1e, g2e, ci0, cis

class KnownValues(unittest.TestCase):
    def test_contract(self):
        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=0)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=0)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 83.016780379400785, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=1)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=1)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.295069645213317, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=3)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=3)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 82.256692620435118, 9)

        ci1 = fci.addons.symmetrize_wfn(ci0, norb, nelec, orbsym, wfnsym=2)
        ci1ref = direct_spin1.contract_2e(g2e, ci1, norb, nelec)
        ci1 = cis.contract_2e(g2e, ci1, norb, nelec, wfnsym=2)
        self.assertAlmostEqual(abs(ci1ref - ci1).max(), 0, 9)
        self.assertAlmostEqual(numpy.linalg.norm(ci1), 81.343382883053323, 9)

    def test_kernel(self):
        e, c = fci.direct_spin1_symm.kernel(h1e, g2e, norb, nelec, orbsym=orbsym)
        self.assertAlmostEqual(e, -84.200905534209554, 8)
        e = fci.direct_spin1_symm.energy(h1e, g2e, c, norb, nelec)
        self.assertAlmostEqual(e, -84.200905534209554, 8)

    def test_fci_spin_square_nroots(self):
        mol = gto.M(
            verbose = 0,
            atom = '''
            O    0.  0.      0.
            H    0.  -0.757  0.587
            H    0.  0.757   0.587''',
            basis = '631g',
            symmetry = 1)
        m = scf.RHF(mol).set(conv_tol=1e-15).run()
        mc = mcscf.casci_symm.CASCI(m, 4, (2, 0))
        mc.fcisolver.nroots = 2
        mc.kernel()[0]
        self.assertTrue(len(mc.e_tot) == 1)
        ss = mc.fcisolver.spin_square(mc.ci[0], mc.ncas, mc.nelecas)
        self.assertAlmostEqual(ss[0], 2, 9)

        mc = mcscf.casci.CASCI(m, 4, (2, 0))
        mc.fcisolver.nroots = 2
        mc.kernel()[0]
        ss = mc.fcisolver.spin_square(mc.ci[1], mc.ncas, mc.nelecas)
        self.assertAlmostEqual(ss[0], 2, 9)

    def test_guess_wfnsym(self):
        self.assertEqual(cis.guess_wfnsym(norb, nelec), 0)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0), 3)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0, wfnsym=0), 0)
        self.assertEqual(cis.guess_wfnsym(norb, nelec, ci0, wfnsym='B2'), 3)
        self.assertRaises(RuntimeError, cis.guess_wfnsym, norb, nelec, numpy.zeros_like(ci0), wfnsym=1)

    def test_csf2civec(self):
        def check(ci_str, orbsym, degen_mapping):
            norb = orbsym.size
            nelec = bin(ci_str).count('1')
            strs = numpy.asarray(cistring.make_strings(range(norb), nelec))
            addr = numpy.where(strs == ci_str)[0][0]
            ci1 = direct_spin1_symm._cyl_sym_csf2civec(strs, addr, orbsym, degen_mapping)

            u = direct_spin1_symm._cyl_sym_orbital_rotation(orbsym, degen_mapping)
            ref = numpy.zeros((1, strs.size))
            ref[0, addr] = 1.
            ref = fci.addons.transform_ci(ref, (0, nelec), u).ravel()
            self.assertAlmostEqual(abs(ref-ci1).max(), 0, 14)

        orbsym = numpy.array([6, 7, 0, 2, 3, 11, 10, 16, 17, 5, 22, 23])
        degen_mapping = numpy.array([1, 0, 2, 4, 3, 6, 5, 8, 7, 9, 11, 10])

        check(0b101000111, orbsym, degen_mapping)
        check(0b110100001000, orbsym, degen_mapping)
        check(0b100110000010, orbsym, degen_mapping)
        check(0b100011010001, orbsym, degen_mapping)
        check(0b100100101010, orbsym, degen_mapping)
        check(0b10110100110, orbsym, degen_mapping)

        numpy.random.seed(2)
        idx = numpy.arange(orbsym.size)
        numpy.random.shuffle(idx)
        rank = idx.argsort()
        orbsym = orbsym[idx]
        degen_mapping[rank] = rank[degen_mapping]
        check(0b101000111, orbsym, degen_mapping)
        check(0b110100001000, orbsym, degen_mapping)
        check(0b100110000010, orbsym, degen_mapping)
        check(0b100011010001, orbsym, degen_mapping)
        check(0b100100101010, orbsym, degen_mapping)
        check(0b10110100110, orbsym, degen_mapping)

    def test_linearmole(self):
        mol = gto.M(
            atom = 'Li 0 0 0; Li 0 0 2.913',
            basis = '''
#BASIS SET: (9s,4p,1d) -> [3s,2p,1d]
Li    S
   1469.0000000              0.0007660             -0.0001200        
    220.5000000              0.0058920             -0.0009230        
     50.2600000              0.0296710             -0.0046890        
     14.2400000              0.1091800             -0.0176820        
      4.5810000              0.2827890             -0.0489020        
      1.5800000              0.4531230             -0.0960090        
      0.5640000              0.2747740             -0.1363800        
      0.0734500              0.0097510              0.5751020        
Li    P
      1.5340000              0.0227840        
      0.2749000              0.1391070        
      0.0736200              0.5003750        
Li    P
      0.0240300              1.0000000        
''',
            symmetry = True,
        )
        mf = mol.RHF().run()
        mci = fci.FCI(mol, mf.mo_coeff, singlet=False)
        ex, ci_x = mci.kernel(wfnsym='E2ux')
        ey, ci_y = mci.kernel(wfnsym='E2uy')
        self.assertAlmostEqual(ex - ey, 0, 7)
        self.assertAlmostEqual(ex - -14.70061197088, 0, 7)
        ss, sz = mci.spin_square(ci_x, mf.mo_energy.size, mol.nelec)
        self.assertAlmostEqual(ss, 2, 6)

        swap_xy = numpy.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        mo_swapxy = mol.ao_rotation_matrix(swap_xy).dot(mf.mo_coeff)
        u = mf.mo_coeff.T.dot(mf.get_ovlp()).dot(mo_swapxy)
        ci1 = fci.addons.transform_ci(ci_x, (3,3), u.T)
        self.assertAlmostEqual(ci_x.ravel().dot(ci_y.ravel()), 0, 9)
        self.assertAlmostEqual(abs(ci1.ravel().dot(ci_x.ravel())), 1, 9)
        ci1 = fci.addons.transform_ci(ci_y, (3,3), u.T)
        self.assertAlmostEqual(abs(ci1.ravel().dot(ci_y.ravel())), 1, 9)

    def test_incomplete_orbsym(self):
        mol = gto.Mole()
        mol.groupname = 'Dooh'
        sol = direct_spin1_symm.FCI(mol)
        no, ne = 2, 2
        h1 = numpy.ones((no,no))
        h2 = numpy.ones((no,no,no,no))
        orbsym = lib.tag_array(numpy.array([0,3]), degen_mapping=[0,2])
        with self.assertRaises(lib.exceptions.PointGroupSymmetryError):
            sol.kernel(h1, h2, no, ne, orbsym=orbsym)

    def test_many_roots(self):
        norb = 4
        nelec = (2, 2)
        nroots = 36
        h1 = numpy.eye(norb) * -.5
        h2 = numpy.zeros((norb, norb, norb, norb))
        orbsym = numpy.array([0, 5, 3, 2])
        for i in range(norb):
            h2[i,i,i,i] = .1
        obj = direct_spin1_symm.FCI()
        e, fcivec = obj.kernel(h1, h2, norb, nelec, nroots=nroots,
                               davidson_only=True, orbsym=orbsym)
        self.assertAlmostEqual(e[0], -1.8, 9)

    def test_guess_wfnsym_cyl_sym(self):
        mol = gto.M(atom='C 0 0 0; C 0 0 1.5', basis='6-31g', symmetry=True)
        mf = mol.RHF().run()
        mc = mcscf.CASCI(mf, 8, 4)
        mc.fcisolver.wfnsym = 'A1g'
        ncas = {'A1g':2, 'A1u':2, 'E1gx':1, 'E1gy':1, 'E1ux':1, 'E1uy':1}
        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas)
        na = math.comb(8, 2)
        ci0 = numpy.zeros((na, na))
        ci0[1,1] = ci0[2,2] = .5**.5 # corresponding to (E+)(E-') + (E-)(E+') => A1
        mc.kernel(mo, ci0=ci0)
        self.assertAlmostEqual(mc.e_cas, -4.205889578214524, 9)

if __name__ == "__main__":
    print("Full Tests for spin1-symm")
    unittest.main()
