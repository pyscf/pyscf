#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.kernel()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 13.649710173723346, 9)

    def test_1e(self):
        mf = scf.hf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

    def test_1e_symm(self):
        molsym = gto.M(
            atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            basis = 'cc-pvdz',
            symmetry = 1,
        )
        mf = scf.hf_symm.HF1e(molsym)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

    def test_energy_tot(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, -59.332199154299914, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(abs(pop).sum(), 22.941032799355845, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(abs(pop).sum(), 22.056441149586863, 7)

    def test_analyze(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        mo = numpy.random.random((nao,nao))
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.0048449691540391, 6)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_nr_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.rohf.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)

    def test_damping(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        s = scf.hf.get_ovlp(mol)
        d = numpy.random.random((nao,nao))
        d = d + d.T
        f = scf.hf.damping(s, d, scf.hf.get_hcore(mol), .5)
        self.assertAlmostEqual(numpy.linalg.norm(f), 23361.854064083178, 9)

    def test_level_shift(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        s = scf.hf.get_ovlp(mol)
        d = numpy.random.random((nao,nao))
        d = d + d.T
        f = scf.hf.level_shift(s, d, scf.hf.get_hcore(mol), .5)
        self.assertAlmostEqual(numpy.linalg.norm(f), 94.230157719053565, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.hf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 199.66041114502335, 9)

    def test_hf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.0048449691540391, 6)

    def test_hf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        mf.irrep_nelec = {'B1':4}
        self.assertAlmostEqual(mf.scf(), -75.074736446470723, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.9778759898704612, 6)

    def test_hf_symm_rohf(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.6782452972117743, 6)

    def test_hf_symm_rohf_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        mf.irrep_nelec = {'B1':(2,1)}
        self.assertAlmostEqual(mf.scf(), -75.008317646307404, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.7873076011029529, 6)

    def test_n2_symm(self):
        pmol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''
                N     0    0    0
                N     0    0    1''',
            symmetry = 1,
            basis = 'cc-pvdz')
        mf = scf.hf_symm.RHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)

    def test_n2_symm_rohf(self):
        pmol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''
                N     0    0    0
                N     0    0    1''',
            symmetry = 1,
            charge = 1,
            spin = 1,
            basis = 'cc-pvdz')
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.33899076078299, 9)

    def test_dot_eri_dm(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        j0, k0 = scf.hf.dot_eri_dm(mf._eri, dm+dm.T, hermi=0)
        j1, k1 = scf.hf.dot_eri_dm(mf._eri, dm+dm.T, hermi=1)
        self.assertTrue(numpy.allclose(j0,j1))
        self.assertTrue(numpy.allclose(k0,k1))
        j1, k1 = scf.hf.dot_eri_dm(mf._eri, dm, hermi=0)
        self.assertAlmostEqual(numpy.linalg.norm(j1), 77.035779188661465, 9)
        self.assertAlmostEqual(numpy.linalg.norm(k1), 46.253491700647963, 9)

    def test_ghost_atm_meta_lowdin(self):
        mol = gto.Mole()
        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.spin = 1
        mol.symmetry = True
        mol.basis = {'O':'ccpvdz', 'H':'ccpvdz',
                     'GHOST': gto.basis.load('ccpvdz','H')}
        mol.build()
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -75.393287998638741, 9)


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()

