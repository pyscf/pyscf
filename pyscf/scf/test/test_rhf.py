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

import copy
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

def tearDownModule():
    global mol, mf, n2sym, n2mf
    mol.stdout.close()
    n2sym.stdout.close()
    del mol, mf, n2sym, n2mf


class KnownValues(unittest.TestCase):
    def test_init_guess_minao(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H1    0    -0.757   0.587
        H2    0    0.757    0.587''',
            basis = 'ccpvdz',
        )
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(lib.finger(dm), 2.5912875957299684, 9)

        mol1 = gto.M(atom='Mo', basis='lanl2dz', ecp='lanl2dz',
                     verbose=7, output='/dev/null')
        dm = scf.hf.get_init_guess(mol1, key='minao')
        self.assertAlmostEqual(lib.finger(dm), 1.5371195992125495, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, mol1.intor('int1e_ovlp')), 14, 9)

        mol1.basis = 'sto3g'
        mol1.build(0, 0)
        dm = scf.hf.get_init_guess(mol1, key='minao')
        self.assertAlmostEqual(lib.finger(dm), 1.8936729909734513, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, mol1.intor('int1e_ovlp')), 13.4787347477, 7)
        mol1.stdout.close()

    def test_init_guess_atom(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H1    0    -0.757   0.587
        H2    0    0.757    0.587''',
            basis = 'ccpvdz',
        )
        dm = scf.hf.get_init_guess(mol, key='atom')
        self.assertAlmostEqual(lib.finger(dm), 2.7458577873928842, 9)

        dm = scf.ROHF(mol).init_guess_by_atom()
        self.assertAlmostEqual(lib.finger(dm[0]), 2.7458577873928842/2, 9)

    def test_init_guess_chk(self):
        dm = scf.hf.SCF(mol).get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.finger(dm), 2.5912875957299684, 9)

        dm = mf.get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.finger(dm), 3.2111753674560535, 9)

    def test_1e(self):
        mf = scf.rohf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

        mf = scf.RHF(gto.M(atom='H', spin=1))
        self.assertAlmostEqual(mf.kernel(), -0.46658184955727555, 9)

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

    def test_get_grad(self):
        g = mf.get_grad(mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(abs(g).max(), 0, 6)

    def test_input_diis(self):
        adiis = scf.hf.diis.ADIIS(mol)
        mf1 = scf.RHF(mol)
        mf1.DIIS = scf.hf.diis.ADIIS
        mf1.max_cycle = 4
        eref = mf1.kernel()

        mf1.diis = adiis
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -75.987815719969291, 9)

        dm = mf1.make_rdm1()
        mf1.max_cycle = 3
        e2 = mf1.kernel(dm)
        self.assertAlmostEqual(e2, eref, 9)

    def test_energy_tot(self):
        e = n2mf.energy_tot(n2mf.make_rdm1())
        self.assertAlmostEqual(e, n2mf.e_tot, 9)

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
        pop, chg = mf.mulliken_pop(mol, [dm*.5]*2)
        self.assertAlmostEqual(abs(pop).sum(), 22.941032799355845, 7)

        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(abs(pop).sum(), 22.048073484937646, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='minao')
        self.assertAlmostEqual(abs(pop).sum(), 22.098274261783196, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='scf')
        self.assertAlmostEqual(abs(pop).sum(), 22.117869619510266, 7)

        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, [dm*.5]*2, pre_orth_method='ano')
        self.assertAlmostEqual(abs(pop).sum(), 22.048073484937646, 7)

    def test_analyze(self):
        popandchg, dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.0584447549532596, 8)
        popandchg, dip = mf.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.2031790129016922, 6)

        mf1 = mf.view(scf.rohf.ROHF)
        popandchg, dip = mf1.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 4.0049440587033116, 6)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.2031790129016922, 6)

        mf1 = copy.copy(n2mf)
        (pop, chg), dip = n2mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.5467414321488357, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 0, 9)
        mf1 = copy.copy(n2mf)
        mf1.mo_coeff = numpy.array(n2mf.mo_coeff)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.8893148995392353, 6)

        mf1 = n2mf.view(scf.hf_symm.ROHF)
        mf1.mo_coeff = numpy.array(n2mf.mo_coeff)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.8893148995392353, 6)

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

        pmol = gto.Mole()
        pmol.atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587'''
        pmol.basis = '6-31g'
        pmol.cart = True

        mf1 = scf.hf.SCF(pmol)
        mf1.direct_scf = True
        mf1.max_memory = 0
        nao = pmol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,3,nao,nao)) - .5 + 0j
        vhf2 = mf1.get_veff(pmol, dm[0,0], hermi=0)
        self.assertEqual(vhf2.ndim, 2)

        vhf3 = mf1.get_veff(pmol, dm[0], hermi=0)
        self.assertEqual(vhf3.ndim, 3)
        self.assertAlmostEqual(abs(vhf3[0]-vhf2).max(), 0, 12)

        vhf4 = mf1.get_veff(pmol, dm, hermi=0)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(lib.finger(vhf4), 4.9026999849223287, 12)
        self.assertAlmostEqual(abs(vhf4[0]-vhf3).max(), 0, 12)

    def test_hf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.0049439389172425, 6)

    def test_hf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        mf.irrep_nelec = {'B1':4}
        self.assertAlmostEqual(mf.scf(), -75.074736446470723, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.9779576643902912, 6)

    def test_hf_symm_rohf(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)
        (pop, chg), dip = mf.analyze()
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
        (pop, chg), dip = mf.analyze()
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
        mol.stdout.close()

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

        mo_coeff = numpy.eye(energy.size)
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
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
        mol.stdout.close()

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

        mf1 = scf.RHF(mol).set(verbose=0).view(scf.hf_symm.ROHF)
        self.assertTrue(numpy.allclose(mf1.get_occ(energy, mo_coeff),
                [0 ,2 ,0 ,0 ,0 ,0 ,2 ,0 ,0 ,0 ,0 ,0 ,2 ,0 ,2 ,0 ,0 ,0 ,0 ,2]))

    def test_get_occ_extreme_case(self):
        mol = gto.M(atom='He', verbose=7, output='/dev/null')
        mf = scf.RHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, -2.8077839575399737, 12)

        mol.charge = 2
        mf = scf.RHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, 0, 12)
        mol.stdout.close()

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

    def test_apply(self):
        from pyscf import mp
        self.assertTrue(isinstance(mf.apply(mp.MP2), mp.mp2.RMP2))
        mf1 = scf.RHF(mol)
        self.assertTrue(isinstance(mf1.apply('MP2'), mp.mp2.RMP2))

    def test_update_from_chk(self):
        mf1 = scf.RHF(mol).update(mf.chkfile)
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 12)

    def test_mute_chkfile(self):
        # To ensure "mf.chkfile = None" does not affect post-SCF calculations
        mol = gto.M(atom='he', basis='6-311g', verbose=0)
        mf1 = scf.RHF(mol)
        mf1.chkfile = None
        mf1.newton().kernel()
        mf1.apply('CISD').run()
        mf1.apply('CCSD').run()
        mf1.apply('TDHF').run()
        mf1.apply('CASSCF', 2, 2).run()
        mf1.nuc_grad_method().run()

    def test_as_scanner(self):
        mf_scanner = mf.as_scanner().as_scanner()
        mf_scanner.chkfile = None
        self.assertAlmostEqual(mf_scanner(mol), mf.e_tot, 9)

        mf_scanner = mf.x2c().density_fit().newton().as_scanner()
        mf_scanner.chkfile = None
        self.assertAlmostEqual(mf_scanner(mol.atom), -76.075408156235909, 9)

        mol1 = gto.M(atom='H 0 0 0; H 0 0 .9', basis='cc-pvdz')
        ref = scf.RHF(mol1).x2c().density_fit().run()
        e1 = mf_scanner('H 0 0 0; H 0 0 .9')
        self.assertAlmostEqual(e1, -1.116394048204042, 9)
        self.assertAlmostEqual(e1, ref.e_tot, 9)

        mfs = scf.RHF(mol1).as_scanner()
        mfs.__dict__.update(scf.chkfile.load(ref.chkfile, 'scf'))
        e = mfs(mol1)
        self.assertAlmostEqual(e, -1.1163913004438035, 9)

    def test_natm_eq_0(self):
        mol = gto.M()
        mol.nelectron = 2
        mf = scf.hf.RHF(mol)
        mf.get_hcore = lambda *args: numpy.diag(numpy.arange(3))
        mf.get_ovlp = lambda *args: numpy.eye(3)
        mf._eri = numpy.zeros((3,3,3,3))
        for i in range(3):
            mf._eri[i,i,i,i] = .2
        dm = mf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(lib.finger(dm), 2., 9)
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, 0.2, 9)

    def test_uniq_var(self):
        mo_occ = mf.mo_occ.copy()
        nmo = mo_occ.size
        nocc = numpy.count_nonzero(mo_occ > 0)
        nvir = nmo - nocc
        numpy.random.seed(1)
        f = numpy.random.random((nmo,nmo))
        f_uniq = scf.hf.pack_uniq_var(f, mo_occ)
        self.assertEqual(f_uniq.size, nocc*nvir)
        f1 = scf.hf.unpack_uniq_var(f_uniq, mo_occ)
        self.assertAlmostEqual(abs(f1 + f1.T).max(), 0, 12)

        mo_occ[4:7] = 1
        ndocc = 4
        nocc = 7
        f_uniq = scf.hf.pack_uniq_var(f, mo_occ)
        self.assertEqual(f_uniq.size, nocc*(nmo-ndocc)-(nocc-ndocc)**2)

        f1 = scf.hf.unpack_uniq_var(f_uniq, mo_occ)
        self.assertAlmostEqual(abs(f1 + f1.T).max(), 0, 12)

    def test_check_convergence(self):
        mf1 = copy.copy(n2mf)
        mf1.diis = False
        count = [0]
        def check_convergence(envs):
            count[0] += 1
            return envs['norm_gorb'] < 0.1
        mf1.check_convergence = check_convergence
        mf1.kernel()
        self.assertAlmostEqual(mf1.e_tot, -108.9297980718255, 9)
        self.assertEqual(count[0], 3)

    def test_canonicalize(self):
        n2_rohf = n2mf.view(scf.hf_symm.ROHF)
        e, c = n2_rohf.canonicalize(n2mf.mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

        mo_coeff = numpy.array(n2mf.mo_coeff)
        e, c = n2mf.canonicalize(mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

        n2_rohf = n2mf.view(scf.rohf.ROHF)
        e, c = n2_rohf.canonicalize(n2mf.mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

    def test_get_irrep_nelec(self):
        fock = n2mf.get_fock()
        s1e = n2mf.get_ovlp()
        e, c = n2mf.eig(fock, s1e)
        mo_occ = n2mf.get_occ(e, c)
        irrep_nelec = n2mf.get_irrep_nelec(n2sym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], 4)
        self.assertEqual(irrep_nelec['A1g'], 6)
        self.assertEqual(irrep_nelec['E1ux'], 2)
        self.assertEqual(irrep_nelec['E1uy'], 2)
        n2_rhf = copy.copy(n2mf)
        n2_rhf.irrep_nelec = irrep_nelec
        n2_rhf.irrep_nelec['A2g'] = 0
        n2_rhf.irrep_nelec['E2gx'] = 2
        self.assertRaises(ValueError, n2_rhf.build)
        n2_rhf.irrep_nelec['A1g'] = 32
        self.assertRaises(ValueError, n2_rhf.build)

        n2_rohf = n2mf.view(scf.hf_symm.ROHF)
        irrep_nelec = n2_rohf.get_irrep_nelec(n2sym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], (2,2))
        self.assertEqual(irrep_nelec['A1g'], (3,3))
        self.assertEqual(irrep_nelec['E1ux'], (1,1))
        self.assertEqual(irrep_nelec['E1uy'], (1,1))

        n2_rohf.irrep_nelec = irrep_nelec
        n2_rohf.irrep_nelec['A2g'] = 0
        n2_rohf.nelec = (8,6)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (2,2)
        n2_rohf.irrep_nelec['E2gx'] = 0
        n2_rohf.irrep_nelec['E2gy'] = 0
        n2_rohf.irrep_nelec['E2ux'] = 0
        n2_rohf.irrep_nelec['E2uy'] = 0
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (4,2)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (0,2)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (3,2)
        n2_rohf.irrep_nelec['A1u'] = (2,3)
        self.assertRaises(ValueError, n2_rohf.build)

    def test_rohf_spin_square(self):
        mf1 = mf.view(scf.rohf.ROHF)
        ss, s = mf1.spin_square()
        self.assertAlmostEqual(ss, 0, 12)
        self.assertAlmostEqual(s, 1, 12)

        mf1.nelec = (6, 4)
        ss, s = mf1.spin_square()
        self.assertAlmostEqual(ss, 2, 12)
        self.assertAlmostEqual(s, 3, 12)


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()

