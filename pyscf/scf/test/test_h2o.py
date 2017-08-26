#!/usr/bin/env python
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
from pyscf.scf import dhf

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


class KnowValues(unittest.TestCase):
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
        uhf = dhf.RHF(mol)
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
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.064429619915702, 8)

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
        self.assertAlmostEqual(numpy.linalg.norm(dm), 2.923422868807739, 8)

        pmol = gto.M(atom='ghost-O 0 0 0; H 0 0 0.5; H 0 0.5 0', basis='ccpvdz')
        dm = scf.hf.init_guess_by_atom(pmol)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 0.86450726178750226, 8)

    def test_init_guess_1e(self):
        dm = scf.hf.init_guess_by_1e(mol)
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
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
            dm1 = mf1.init_guess_by_chkfile()
            self.assertAlmostEqual(numpy.linalg.norm(dm1), ref, 9)

        save(scf.hf.RHF)
        check(scf.hf.RHF, 5.2644790347333048)
        check(scf.rohf.ROHF, 3.7225488248743273)
        check(scf.uhf.UHF, 3.7225488248743273)
        check(scf.dhf.UHF, 3.7225488248743273)

        save(scf.uhf.UHF)
        check(scf.hf.RHF, 5.2644790347333048)
        check(scf.rohf.ROHF, 3.7225488248743273)
        check(scf.uhf.UHF, 3.7225488248743273)
        check(scf.dhf.UHF, 3.7225488248743273)

        save(scf.dhf.UHF)
        check(scf.dhf.UHF, 7.3540281989311271)

    def test_scanner(self):
        from pyscf import dft
        mol1 = molsym.copy()
        mol1.set_geom_('''
        O   0.   0.       .1
        H   0.   -0.757   0.587
        H   0.   0.757    0.587''')
        mf_scanner = scf.UHF(molsym).density_fit('weigend').as_scanner()
        self.assertAlmostEqual(mf_scanner(molsym), -75.98321088694874, 9)
        self.assertAlmostEqual(mf_scanner(mol1), -75.97901175977492, 9)

        mf_scanner = scf.fast_newton(scf.ROHF(molsym)).as_scanner()
        self.assertAlmostEqual(mf_scanner(molsym), -75.983948498066198, 9)
        self.assertAlmostEqual(mf_scanner(mol1), -75.97974371226907, 9)

        mf_scanner = dft.RKS(molsym).set(xc='bp86').as_scanner()
        self.assertAlmostEqual(mf_scanner(molsym), -76.385043416002361, 9)
        self.assertAlmostEqual(mf_scanner(mol1), -76.372784697245777, 9)



if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
