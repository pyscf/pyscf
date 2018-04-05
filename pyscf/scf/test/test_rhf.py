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

import numpy
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf

mol = gto.M(
    verbose = 7,
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

n2sym = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = '''
        N     0    0    0
        N     0    0    1''',
    symmetry = 1,
    basis = 'cc-pvdz')
n2mf = scf.RHF(n2sym).set(conv_tol=1e-10).run()


class KnowValues(unittest.TestCase):
    def test_init_guess_minao(self):
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm).sum(), 13.649710173723346, 9)

    def test_1e(self):
        mf = scf.rohf.HF1e(mol)
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
        self.assertAlmostEqual(abs(pop).sum(), 22.048073484937646, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='minao')
        self.assertAlmostEqual(abs(pop).sum(), 22.098274261783196, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='scf')
        self.assertAlmostEqual(abs(pop).sum(), 22.117869619510266, 7)

    def test_analyze(self):
        popandchg, dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.05844441822, 8)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_nr_rhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.RHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

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
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.0049439389172425, 6)

    def test_hf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        mf.irrep_nelec = {'B1':4}
        self.assertAlmostEqual(mf.scf(), -75.074736446470723, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.9779576643902912, 6)

    def test_hf_symm_rohf(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)
        pop, chg = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.6783793407635832, 6)

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
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.7873920690764575, 6)

    def test_n2_symm(self):
        mf = scf.hf_symm.RHF(n2sym)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)

    def test_n2_symm_rohf(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.33899076078299, 9)

    def test_n2_symm_fixnocc(self):
        mf = scf.hf_symm.RHF(n2sym)
        mf.irrep_nelec = {'A1g':8, 'A1u':2, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -106.52905502298771, 9)

    def test_n2_symm_rohf_fixnocc(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.hf_symm.ROHF(pmol)
        mf.irrep_nelec = {'A1g':6, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -108.21954550790898, 9)

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
        mol.verbose = 0
        mol.spin = 1
        mol.symmetry = True
        mol.basis = {'O':'ccpvdz', 'H':'ccpvdz',
                     'GHOST': gto.basis.load('ccpvdz','H')}
        mol.build()
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -75.393287998638741, 9)

    def test_rhf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=10)
        mf = scf.hf.RHF(mol)
        energy = numpy.array([-10, -1, 1, -2, 0, -3])
        self.assertTrue(numpy.allclose(mf.get_occ(energy), [2, 2, 0, 2, 2, 2]))

    def test_rhf_symm_get_occ(self):
        mf = scf.RHF(n2sym).set(verbose = 0)
        orbsym = numpy.array([0 , 5, 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5])
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        mo_coeff = lib.tag_array(numpy.eye(energy.size), orbsym=orbsym)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))

    def test_rohf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=8, spin=2)
        mf = scf.rohf.ROHF(mol)
        energy = numpy.array([-10, -1, 1, -2, 0, -3])
        self.assertTrue(numpy.allclose(mf.get_occ(energy), [2, 1, 0, 2, 1, 2]))
        pmol = n2sym.copy()
        pmol.spin = 2
        pmol.symmetry = False
        mf = scf.rohf.ROHF(pmol).set(verbose = 0)
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [0, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        # 0 virtual
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [2, 2, 1, 2, 2, 2, 2, 1]))
        # 0 core
        mf.nelec = (14, 0)
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_rohf_symm_get_occ(self):
        pmol = n2sym.copy()
        pmol.charge = 0
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0)
        orbsym = numpy.array([0 , 5, 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5])
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        mo_coeff = lib.tag_array(numpy.eye(energy.size), orbsym=orbsym)
        mf.irrep_nelec = {'A1g':7, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [2, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 2, 0, 2, 0, 1, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))

    def test_rohf_symm_dump_flags(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

        mf.irrep_nelec = {'A1g':6, 'A1u':10, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

    def test_rhf_dip_moment(self):
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_rohf_dip_moment(self):
        mf = scf.ROHF(mol).run()
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_get_wfnsym(self):
        self.assertEqual(n2mf.wfnsym, 0)

        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0).run()
        self.assertTrue(mf.wfnsym in (2, 3))

    def test_complex_orbitals(self):
        nao = mol.nao_nr()
        mf = scf.RHF(mol)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.RHF(mol).set(max_memory=0)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.rohf.ROHF(mol)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.rohf.ROHF(mol).set(max_memory=0)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()

