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

mf = scf.UHF(mol)
mf.conv_tol = 1e-14
mf.scf()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.uhf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 13.649710173723337, 9)

    def test_energy_tot(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, 57.122667754846844, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(numpy.linalg.norm(pop), 8.3342045408596057, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 12.32518616560702, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='minao')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 12.375046214734942, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='scf')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 12.177665514896324, 9)

    def test_analyze(self):
        nao = mol.nao_nr()
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 2.8318530439275791, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.uhf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 398.09239104094513, 9)

    def test_spin_square(self):
        self.assertAlmostEqual(mf.spin_square(mf.mo_coeff)[0], 0, 9)

    def test_map_rhf_to_uhf(self):
        scf.uhf.map_rhf_to_uhf(scf.RHF(mol))

    def test_uhf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.uhf_symm.UHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)

    def test_uhf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.uhf_symm.UHF(pmol)
        mf.irrep_nelec = {'B1':(2,1)}
        self.assertAlmostEqual(mf.scf(), -75.010623169610966, 9)

    def test_n2_symm(self):
        pmol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''
                N     0    0    0
                N     0    0    1''',
            symmetry = 1,
            basis = 'cc-pvdz')
        mf = scf.uhf_symm.UHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)


if __name__ == "__main__":
    print("Full Tests for uhf")
    unittest.main()

