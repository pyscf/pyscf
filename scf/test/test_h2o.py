#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
import scipy.linalg
import tempfile
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
        rhf = scf.density_fit(scf.RHF(mol))
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
        mf = scf.density_fit(scf.ROHF(mol))
        mf.conv_tol = 1e-11
        self.assertAlmostEqual(mf.scf(), -75.5775921401438, 9)

    def test_nr_df_uhf(self):
        uhf = scf.density_fit(scf.UHF(mol))
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
        self.assertAlmostEqual(uhf.scf(), -76.038520472820863, 9)

    def test_r_rhf(self):
        uhf = dhf.RHF(mol)
        self.assertAlmostEqual(uhf.scf(), -76.038520472820863, 9)

    def test_level_shift_uhf(self):
        uhf = scf.UHF(mol)
        uhf.level_shift_factor = 1.2
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
        ftmp = tempfile.NamedTemporaryFile()
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.0381294999637398, 9)

        mf = scf.hf.RHF(mol)
        dm0 = scf.hf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.get_init_guess(key='minao')
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.uhf.UHF(mol)
        dm0 = scf.uhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.get_init_guess(key='minao')
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.DHF(mol)
        dm0 = scf.dhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.get_init_guess(key='minao')
        self.assertTrue(numpy.allclose(dm0, dm1))

    def test_init_guess_atom(self):
        dm = scf.hf.init_guess_by_atom(mol)
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile()
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 3.0644293224517574, 9)

        mf = scf.hf.RHF(mol)
        dm0 = scf.rhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_atom(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.uhf.UHF(mol)
        dm0 = scf.uhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_atom(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.DHF(mol)
        dm0 = scf.dhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_atom(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

    def test_init_guess_1e(self):
        dm = scf.hf.init_guess_by_1e(mol)
        s = scf.hf.get_ovlp(mol)
        occ, mo = scipy.linalg.eigh(dm, s, type=2)
        ftmp = tempfile.NamedTemporaryFile()
        scf.chkfile.dump_scf(mol, ftmp.name, 0, occ, mo, occ)
        self.assertAlmostEqual(numpy.linalg.norm(dm), 5.3700827555643791, 9)

        mf = scf.hf.RHF(mol)
        dm0 = scf.rhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.uhf.UHF(mol)
        dm0 = scf.uhf.init_guess_by_chkfile(mol, ftmp.name, project=False)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertTrue(numpy.allclose(dm0, dm1))

        mf = scf.hf.ROHF(mol)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertAlmostEqual(numpy.linalg.norm(dm1),
                               5.3700827555643791/numpy.sqrt(2), 9)

        mf = scf.DHF(mol)
        dm1 = mf.init_guess_by_1e(mol)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 7.5925203207385055, 9)

    def test_init_guess_chkfile(self):
        ftmp = tempfile.NamedTemporaryFile()
        def save(HFclass):
            mf0 = HFclass(mol)
            mf0.chkfile = ftmp.name
            h = mf0.get_hcore(mol)
            s = mf0.get_ovlp(mol)
            f = mf0.get_fock(h, s, numpy.zeros_like(h), 0)
            e, mo = mf0.eig(f, s)
            occ = mf0.get_occ(e, mo)
            mf0.dump_chk(0, e, mo, occ)
        def check(HFclass, ref):
            mol1 = mol.copy()
            mol1.basis = 'cc-pvdz'
            mol1.build()
            mf1 = HFclass(mol1)
            mf1.chkfile = ftmp.name
            dm1 = mf1.init_guess_by_chkfile()
            self.assertAlmostEqual(numpy.linalg.norm(dm1), ref, 9)

        save(scf.hf.RHF)
        check(scf.hf.RHF, 12.960598264586917)
        check(scf.hf.ROHF, 9.1645269211240379)
        check(scf.uhf.UHF, 9.1645269211240379)
        check(scf.dhf.UHF, 9.1645269211240095)

        save(scf.uhf.UHF)
        check(scf.hf.RHF, 12.960598264586917)
        check(scf.hf.ROHF, 9.1645269211240379)
        check(scf.uhf.UHF, 9.1645269211240379)
        check(scf.dhf.UHF, 9.1645269211240095)

        save(scf.dhf.UHF)
        check(scf.dhf.UHF, 18.013448619780711)



if __name__ == "__main__":
    print("Full Tests for H2O")
    unittest.main()
