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
mf.scf()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.uhf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 23.074873357239472, 9)

    def test_energy_tot(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, -9.9092383592937665, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = (numpy.random.random((nao,nao)),
              numpy.random.random((nao,nao)))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(numpy.linalg.norm(pop), 6.641191810110950, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 9.4803469531219982, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='minao')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 9.5893817617320511, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='scf')
        self.assertAlmostEqual(numpy.linalg.norm(pop), 9.4950935664854565, 9)

    def test_analyze(self):
        numpy.random.seed(5)
        nao = mol.nao_nr()
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 2.2649896039590094, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.hf_energy, -76.026765673119627, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.uhf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 260.17203605790604, 9)

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


if __name__ == "__main__":
    print("Full Tests for uhf")
    unittest.main()

