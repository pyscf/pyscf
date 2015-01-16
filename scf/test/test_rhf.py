#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = '/dev/null',
    atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587''',
    basis = 'cc-pvdz',
)

mf = scf.RHF(mol)
mf.scf()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 23.074873357239454, 9)

    def test_1e(self):
        mf = scf.hf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

    def test_energy_tot(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, -56.139015588339575, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(abs(pop).sum(), 18.076192698218936, 9)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm)
        self.assertAlmostEqual(abs(pop).sum(), 17.481299501546292, 9)

    def test_analyze(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        mo = numpy.random.random((nao,nao))
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.2031790165528986, 9)

    def test_scf(self):
        self.assertAlmostEqual(mf.hf_energy, -76.026765673119627, 9)

    def test_nr_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)

    def test_damping(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        s = scf.hf.get_ovlp(mol)
        d = numpy.random.random((nao,nao))
        d = d + d.T
        f = scf.hf.damping(s, d, scf.hf.get_hcore(mol), .5)
        self.assertAlmostEqual(numpy.linalg.norm(f), 10748.99052829406, 9)

    def test_level_shift(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        s = scf.hf.get_ovlp(mol)
        d = numpy.random.random((nao,nao))
        d = d + d.T
        f = scf.hf.level_shift(s, d, scf.hf.get_hcore(mol), .5)
        self.assertAlmostEqual(numpy.linalg.norm(f), 66.16324989296858, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.hf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 130.60201981225086, 9)

    def test_hf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)

    def test_hf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        mf.irrep_nocc = {'B1':4}
        self.assertAlmostEqual(mf.scf(), -75.074736446470723, 9)

    def test_hf_symm_rohf(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)

    def test_hf_symm_rohf_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        mf.irrep_nocc_alpha = {'B1':2}
        mf.irrep_nocc_beta = {'B1':1}
        self.assertAlmostEqual(mf.scf(), -75.008317646307404, 9)

if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()

