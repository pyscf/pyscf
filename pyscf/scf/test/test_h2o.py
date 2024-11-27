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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.scf import dhf

def setUpModule():
    global mol, molsym
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = '/dev/null',
        atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis = {"H": '6-31g',
                 "O": '6-31g',}
    )
    molsym = mol.copy()
    molsym.symmetry = True
    molsym.build(0, 0)

def tearDownModule():
    global mol, molsym
    mol.stdout.close()
    del mol, molsym


class KnownValues(unittest.TestCase):
    def test_nr_rhf(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_rohf(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = {"H": '6-31g',
                     "O": '6-31g',},
            charge = 1,
            spin = 1,
        )
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.578396379589748, 9)
        pop_chg, dip = mf.analyze()
        self.assertAlmostEqual(lib.finger(pop_chg[0]), 1.0036241405313113, 6)
        self.assertAlmostEqual(lib.finger(dip), -1.4000447020842097, 6)

    def test_nr_uhf(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_nr_df_rhf(self):
        rhf = scf.density_fit(scf.RHF(mol), 'weigend')
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -75.983210886950, 9)

    def test_nr_df_rohf(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = {"H": '6-31g',
                     "O": '6-31g',},
            charge = 1,
            spin = 1,
        )
        mf = scf.density_fit(scf.ROHF(mol), 'weigend')
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.5775921401438, 9)

    def test_nr_df_uhf(self):
        uhf = scf.density_fit(scf.UHF(mol), 'weigend')
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -75.983210886950, 9)

    def test_nr_df_ghf(self):
        mf = mol.GHF().density_fit(auxbasis='weigend')
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.983210886950, 9)

    def test_nr_rhf_no_mem(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        rhf.max_memory = 0
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf_no_mem(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        uhf.max_memory = 0
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_nr_rhf_no_direct(self):
        rhf = scf.RHF(mol)
        rhf.conv_tol = 1e-11
        rhf.max_memory = 0
        rhf.direct_scf = False
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_uhf_no_direct(self):
        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-11
        uhf.max_memory = 0
        uhf.direct_scf = False
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_r_uhf(self):
        uhf = dhf.UHF(mol)
        uhf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(uhf.scf(), -76.038520463270061, 7)

    def test_r_rhf(self):
        uhf = scf.DHF(mol)
        uhf.conv_tol_grad = 1e-5
        self.assertAlmostEqual(uhf.scf(), -76.038520463270061, 7)

    def test_level_shift_uhf(self):
        uhf = scf.UHF(mol)
        uhf.level_shift = .2
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_energy_nuc(self):
        self.assertAlmostEqual(mol.energy_nuc(), 9.18825841775, 10)

    def test_nr_rhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        rhf = scf.hf.RHF(mol1)
        rhf.conv_tol = 1e-11
        self.assertAlmostEqual(rhf.scf(), -75.98394849812, 9)

    def test_nr_rohf_symm(self):
        mol = gto.Mole()
        mol.build(
            verbose = 0,
            atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ],
            basis = {"H": '6-31g',
                     "O": '6-31g',},
            charge = 1,
            spin = 1,
            symmetry = True,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.578396379589748, 9)

    def test_nr_uhf_symm(self):
        mol1 = mol.copy()
        mol1.symmetry = 1
        mol1.build()
        uhf = scf.UHF(mol1)
        uhf.conv_tol = 1e-11
        self.assertAlmostEqual(uhf.scf(), -75.98394849812, 9)

    def test_init_guess_minao(self):
        dm = scf.hf.init_guess_by_minao(mol)
        self.assertEqual(dm.mo_coeff.shape[0], mol.nao)
        self.assertEqual(dm.mo_occ.size, dm.mo_coeff.shape[1])
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.0334714065913508, 9)

        mf = scf.hf.RHF(mol)
        dm0 = scf.hf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.get_init_guess(key='minao')
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.DHF(mol)
        dm0 = scf.dhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.get_init_guess(key='minao')
        self.assertTrue(numpy.allclose(dm0, dm1))

        pmol = gto.M(atom='ghost-O 0 0 0; H 0 0 0.5; H 0 0.5 0', basis='ccpvdz')
        dm1 = mf.get_init_guess(key='minao')
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.0334714065913482, 8)

    def test_init_guess_atom(self):
        dm = scf.hf.init_guess_by_atom(mol)
        self.assertEqual(dm.mo_coeff.shape[0], mol.nao)
        self.assertEqual(dm.mo_occ.size, dm.mo_coeff.shape[1])
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.041411845876416, 8)

        mf = scf.hf.RHF(mol)
        dm0 = scf.rhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_atom(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.DHF(mol)
        dm0 = scf.dhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_atom(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        pmol = gto.M(atom=mol.atom, basis='ccpvdz')
        pmol.cart = True
        dm = scf.hf.init_guess_by_atom(pmol)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 2.9173248538892547, 8)

        pmol = gto.M(atom='ghost-O 0 0 0; H 0 0 0.5; H 0 0.5 0', basis='ccpvdz')
        dm = scf.hf.init_guess_by_atom(pmol)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 0.8436562326772896, 8)

    def test_init_guess_1e(self):
        dm = scf.hf.init_guess_by_1e(mol)
        self.assertEqual(dm.mo_coeff.shape[0], mol.nao)
        self.assertEqual(dm.mo_occ.size, dm.mo_coeff.shape[1])
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ,
                             overwrite_mol=False)  # dump_scf twice to test overwrite_mol
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 5.3700828975288122, 9)

        mf = scf.hf.RHF(mol)
        dm0 = scf.rhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.rohf.ROHF(mol)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertAlmostEqual(numpy.linalg.norm(dm1),
                               5.3700828975288122/numpy.sqrt(2), 9)

        mf = scf.rohf.ROHF(molsym)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertAlmostEqual(numpy.linalg.norm(dm1),
                               5.3700828975288122/numpy.sqrt(2), 9)

        mf = scf.DHF(mol)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 7.5925205205065422, 9)

    def test_init_guess_chkfile(self):
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        def save(HFclass):
            mf0 = HFclass(mol)
            mf0.chkfile = ftmp.name
            h = mf0.get_hcore(mol)
            s = mf0.get_ovlp(mol)
            f = mf0.get_fock(h, s, numpy.zeros_like(h), numpy.zeros_like(h))
            mo_energy, mo_coeff = mf0.eig(f, s)
            mo_occ = mf0.get_occ(mo_energy, mo_coeff)
            e_tot = 0
            mf0.dump_chk(locals())
        def check(HFclass, ref):
            mol1 = mol.copy()
            mol1.basis = 'cc-pvdz'
            mol1.build()
            mf1 = HFclass(mol1)
            mf1.chkfile = ftmp.name
            dm1 = mf1.init_guess_by_chkfile(project=True)
            self.assertAlmostEqual(numpy.linalg.norm(dm1), ref, 9)

        save(scf.hf.RHF)
        check(scf.hf.RHF, 5.2653611000274259)
        check(scf.rohf.ROHF, 3.7231725392252519)
        check(scf.uhf.UHF, 3.7231725392252519)
        check(scf.dhf.UHF, 3.7225488248743241)

        save(scf.uhf.UHF)
        check(scf.hf.RHF, 5.2653611000274259)
        check(scf.rohf.ROHF, 3.7231725392252519)
        check(scf.uhf.UHF, 3.7231725392252519)
        check(scf.dhf.UHF, 3.7225488248743241)

        save(scf.dhf.UHF)
        check(scf.dhf.UHF, 7.3552498540235245)

    def test_scanner(self):
        from pyscf import dft
        mol1 = molsym.copy()
        mol1.set_geom_('''
        O   0.   0.       .1
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''')
        mf_scanner = scf.UHF(molsym).density_fit('weigend').as_scanner()
        self.assertAlmostEqual(mf_scanner(molsym), -75.98321088694874, 8)
        self.assertAlmostEqual(mf_scanner(mol1), -75.97901175977492, 8)

        mf_scanner = scf.fast_newton(scf.ROHF(molsym)).as_scanner()
        self.assertAlmostEqual(mf_scanner(molsym), -75.983948498066198, 8)
        self.assertAlmostEqual(mf_scanner(mol1), -75.97974371226907, 8)

        with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
            mf_scanner = dft.RKS(molsym).set(xc='bp86').as_scanner()
            self.assertAlmostEqual(mf_scanner(molsym), -76.385043416002361, 8)
            eref = dft.RKS(mol1).set(xc='bp86').kernel()
            e1 = mf_scanner(mol1)
            self.assertAlmostEqual(e1, -76.372784697245777, 8)
            self.assertAlmostEqual(e1, eref, 8)

        # Test init_guess_by_chkfile for stretched geometry and different basis set
        mol1.atom = '''
        O   0.   0.      -.5
        H   0.  -0.957   0.587
        H   0.   0.957   0.587'''
        mol1.basis = 'ccpvdz'
        mol1.build(0,0)
        self.assertAlmostEqual(mf_scanner(mol1), -76.273052274103648, 5)

        mf = mf_scanner.undo_scanner()
        mf.run()
        self.assertAlmostEqual(mf.e_tot, -76.273052274103648, 5)

    def test_init(self):
        from pyscf import dft
        from pyscf import x2c
        mol_r = mol
        mol_u = gto.M(atom='Li', spin=1, verbose=0)
        mol_r1 = gto.M(atom='H', spin=1, verbose=0)
        sym_mol_r = molsym
        sym_mol_u = gto.M(atom='Li', spin=1, symmetry=1, verbose=0)
        sym_mol_r1 = gto.M(atom='H', spin=1, symmetry=1, verbose=0)
        self.assertTrue(isinstance(scf.RKS(mol_r), dft.rks.RKS))
        self.assertTrue(isinstance(scf.RKS(mol_u), dft.roks.ROKS))
        self.assertTrue(isinstance(scf.UKS(mol_r), dft.uks.UKS))
        self.assertTrue(isinstance(scf.ROKS(mol_r), dft.roks.ROKS))
        self.assertTrue(isinstance(scf.GKS(mol_r), dft.gks.GKS))
        self.assertTrue(isinstance(scf.KS(mol_r), dft.rks.RKS))
        self.assertTrue(isinstance(scf.KS(mol_u), dft.uks.UKS))

        self.assertTrue(isinstance(scf.RHF(mol_r), scf.hf.RHF))
        self.assertTrue(isinstance(scf.RHF(mol_u), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.RHF(mol_r1), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.UHF(mol_r), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.UHF(mol_u), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.UHF(mol_r1), scf.uhf.HF1e))
        self.assertTrue(isinstance(scf.ROHF(mol_r), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.ROHF(mol_u), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.ROHF(mol_r1), scf.rohf.HF1e))
        self.assertTrue(isinstance(scf.HF(mol_r), scf.hf.RHF))
        self.assertTrue(isinstance(scf.HF(mol_u), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.HF(mol_r1), scf.rohf.HF1e))
        self.assertTrue(isinstance(scf.GHF(mol_r), scf.ghf.GHF))
        self.assertTrue(isinstance(scf.GHF(mol_u), scf.ghf.GHF))
        self.assertTrue(isinstance(scf.GHF(mol_r1), scf.ghf.HF1e))
        #TODO: self.assertTrue(isinstance(scf.DHF(mol_r), scf.dhf.RHF))
        self.assertTrue(isinstance(scf.DHF(mol_u), scf.dhf.UHF))
        self.assertTrue(isinstance(scf.DHF(mol_r1), scf.dhf.HF1e))

        self.assertTrue(isinstance(scf.RHF(sym_mol_r), scf.hf_symm.RHF))
        self.assertTrue(isinstance(scf.RHF(sym_mol_u), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.RHF(sym_mol_r1), scf.hf_symm.HF1e))
        self.assertTrue(isinstance(scf.UHF(sym_mol_r), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.UHF(sym_mol_u), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.UHF(sym_mol_r1), scf.uhf_symm.HF1e))
        self.assertTrue(isinstance(scf.ROHF(sym_mol_r), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.ROHF(sym_mol_u), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.ROHF(sym_mol_r1), scf.hf_symm.HF1e))
        self.assertTrue(isinstance(scf.HF(sym_mol_r), scf.hf_symm.RHF))
        self.assertTrue(isinstance(scf.HF(sym_mol_u), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.HF(sym_mol_r1), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.GHF(sym_mol_r), scf.ghf_symm.GHF))
        self.assertTrue(isinstance(scf.GHF(sym_mol_u), scf.ghf_symm.GHF))
        self.assertTrue(isinstance(scf.GHF(sym_mol_r1), scf.ghf_symm.HF1e))

        #self.assertTrue(isinstance(scf.X2C(mol_r), x2c.x2c.RHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(mol_r)), scf.rhf.RHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(mol_u)), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(mol_r1)), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(sym_mol_r)), scf.rhf_symm.RHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(sym_mol_u)), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.sfx2c1e(scf.HF(sym_mol_r1)), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(mol_r)), scf.rhf.RHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(mol_u)), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(mol_r1)), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(sym_mol_r)), scf.rhf_symm.RHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(sym_mol_u)), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.density_fit(scf.HF(sym_mol_r1)), scf.hf_symm.ROHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(mol_r)), scf.rhf.RHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(mol_u)), scf.uhf.UHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(mol_r1)), scf.rohf.ROHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(sym_mol_r)), scf.rhf_symm.RHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(sym_mol_u)), scf.uhf_symm.UHF))
        self.assertTrue(isinstance(scf.newton(scf.HF(sym_mol_r1)), scf.hf_symm.ROHF))

    def test_fast_newton(self):
        nao = mol.nao_nr()
        dm0 = numpy.zeros((nao,nao))
        mf = scf.fast_newton(scf.RHF(mol), dm0=dm0, dual_basis=True,
                             kf_trust_region=3.)
        self.assertAlmostEqual(mf.e_tot, -75.983948497843272, 9)


if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
