#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
import scipy.linalg
import unittest
from pyscf import lib
from pyscf import gto
from pyscf import scf

try:
    from pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None


def make_disp_mol():
    return gto.M(
        atom='''
C  -0.65830719,  0.61123287, -0.00800148
C   0.73685281,  0.61123287, -0.00800148
C   1.43439081,  1.81898387, -0.00800148
C   0.73673681,  3.02749287, -0.00920048
''',
        basis='ccpvtz',
        charge=1,
        spin=1,
        output='/dev/null')

def setUpModule():
    global mol, mf, n2sym, n2mf, mol2, mf2, bak
    mol = gto.M(
        verbose = 7,
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

    mol2 = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
    O     0    0        0
    H     0    -0.757   0.587
    H     0    0.757    0.587''',
        basis = 'cc-pvdz',
        spin = 2,
    )
    mf2 = scf.UHF(mol2).run(conv_tol=1e-10)

    n2sym = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
            N     0    0    0
            N     0    0    1''',
        symmetry = 1,
        basis = 'cc-pvdz')
    n2mf = scf.UHF(n2sym).set(conv_tol=1e-10).run()

def tearDownModule():
    global mol, mf, n2sym, n2mf, mol2, mf2, bak
    mol.stdout.close()
    mol2.stdout.close()
    n2sym.stdout.close()
    del mol, mf, n2sym, n2mf, mol2, mf2

class KnownValues(unittest.TestCase):
    def test_init_guess_minao(self):
        mf = scf.UHF(mol)
        dm1 = mf.init_guess_by_minao(mol, breaksym=False)
        self.assertAlmostEqual(abs(dm1).sum(), 13.649710173723337, 9)
        dm2 = scf.uhf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(abs(dm2).sum(), 12.913908927027279, 9)
        mf.init_guess_breaksym = 2
        dm1 = mf.init_guess_by_minao(mol)
        self.assertAlmostEqual(abs(dm1).sum(), 13.649710173723337, 9)

    def test_init_guess_1e(self):
        dm1 = scf.uhf.init_guess_by_1e(mol, breaksym=False)
        self.assertAlmostEqual(lib.fp(dm1), -0.17065579929349839, 9)
        dm2 = scf.uhf.get_init_guess(mol, key='hcore')
        self.assertAlmostEqual(lib.fp(dm2), 0.69685247431623965, 9)
        self.assertAlmostEqual(abs(dm1[0]-dm2[0]).max(), 0, 9)

    def test_init_guess_atom(self):
        dm1 = mf.init_guess_by_atom(mol, breaksym=False)
        self.assertAlmostEqual(lib.fp(dm1), 0.05094548752961081, 6)
        dm2 = scf.uhf.get_init_guess(mol, key='atom')
        self.assertAlmostEqual(lib.fp(dm2), 0.054774967429943755, 6)
        self.assertAlmostEqual(abs(dm1[1]-dm2[1]).max(), 0, 9)

    def test_init_guess_huckel(self):
        dm1 = mf.init_guess_by_huckel(mol, breaksym=False)
        self.assertAlmostEqual(lib.fp(dm1), 0.6442338252028256, 7)
        dm2 = scf.uhf.UHF(mol).get_init_guess(mol, key='huckel')
        self.assertAlmostEqual(lib.fp(dm2), 0.6174062069308063, 7)

    def test_init_guess_mod_huckel(self):
        dm1 = mf.init_guess_by_mod_huckel(mol, breaksym=False)
        self.assertAlmostEqual(lib.fp(dm1), 0.575004422279537, 7)
        dm2 = scf.uhf.UHF(mol).get_init_guess(mol, key='mod_huckel')
        self.assertAlmostEqual(lib.fp(dm2), 0.601086728278398, 7)

    def test_1e(self):
        mf = scf.uhf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

        mf = scf.UHF(gto.M(atom='H', spin=-1))
        self.assertAlmostEqual(mf.kernel(), -0.46658184955727555, 9)
        mf = scf.UHF(gto.M(atom='H', spin=1, symmetry=1))
        self.assertAlmostEqual(mf.kernel(), -0.46658184955727555, 9)

    def test_init_guess_sap(self):
        dm1 = mf.init_guess_by_sap(mol, breaksym=False)
        self.assertAlmostEqual(lib.fp(dm1), 0.9867930552338582, 7)
        dm2 = scf.uhf.UHF(mol).get_init_guess(mol, key='sap')
        self.assertAlmostEqual(lib.fp(dm2), 0.6440359527450615, 7)

    def test_break_spin_symm_mix(self):
        # H2 at equilibrium: verify DM properties of the breaksym='mix' initial guess
        mol_h2 = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', spin=0, verbose=0)
        s = mol_h2.intor_symmetric('int1e_ovlp')

        mf_h2 = scf.UHF(mol_h2)
        dm = mf_h2.init_guess_by_minao(mol_h2, breaksym='mix')
        dma, dmb = dm

        # spin symmetry must be broken
        self.assertFalse(numpy.allclose(dma, dmb))

        # electron count preserved: Tr(S * DM) = N_elec per spin
        self.assertAlmostEqual(numpy.einsum('ij,ji->', s, dma), 1.0, 5)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', s, dmb), 1.0, 5)

        # DMs must be positive semidefinite (physically valid density matrices)
        self.assertTrue(numpy.all(numpy.linalg.eigvalsh(dma) > -1e-10))
        self.assertTrue(numpy.all(numpy.linalg.eigvalsh(dmb) > -1e-10))

        # Delocalization: the rotated HOMO retains cross-atom (off-diagonal)
        # character, unlike breaksym=1 which explicitly zeroes those blocks.
        slices = mol_h2.aoslice_by_atom()
        p0, p1 = slices[0][2], slices[0][3]
        p2, p3 = slices[1][2], slices[1][3]
        self.assertGreater(abs(dma[p0:p1, p2:p3]).max(), 0.1)

        # breaksym=1 zeros the cross-atom block in dmb; confirm the contrast
        _, dmb1 = mf_h2.init_guess_by_minao(mol_h2, breaksym=1)
        self.assertAlmostEqual(abs(dmb1[p0:p1, p2:p3]).max(), 0.0, 10)
    def test_break_spin_symm_mix_h2_dissociation(self):
        # At stretched H2 (well past the Coulson-Fischer point) UHF with
        # breaksym='mix' should find a lower-energy broken-symmetry solution
        # than RHF, with the unpaired electrons localised on separate atoms.
        mol_h2 = gto.M(atom='H 0 0 0; H 0 0 4.0', basis='sto-3g', spin=0, verbose=0)

        e_rhf = scf.RHF(mol_h2).kernel()

        mf_uhf = scf.UHF(mol_h2)
        mf_uhf.init_guess_breaksym = 'mix'
        e_uhf = mf_uhf.kernel()

        self.assertTrue(mf_uhf.converged)
        # broken-symmetry UHF must be lower than RHF at stretched geometry
        self.assertLess(e_uhf, e_rhf)
        # significant spin contamination expected (<S^2> -> 1 as R -> inf)
        self.assertGreater(mf_uhf.spin_square()[0], 0.5)

    def test_get_grad(self):
        g = mf2.get_grad(mf2.mo_coeff, mf2.mo_occ)
        self.assertAlmostEqual(abs(g).max(), 0, 6)

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
        self.assertAlmostEqual(numpy.linalg.norm(pop), 12.322626374896178, 9)

    def test_mulliken_spin_pop(self):
        Ms_true = [0.990027, 0.504987, 0.504987]
        _, Ms = mf2.mulliken_spin_pop()
        self.assertAlmostEqual(Ms[0], Ms_true[0],5)
        self.assertAlmostEqual(Ms[1], Ms_true[1],5)
        self.assertAlmostEqual(Ms[2], Ms_true[2],5)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_uhf_d3bj(self):
        mol = make_disp_mol()
        mf = scf.UHF(mol)
        mf.disp = 'd3bj'
        e_disp = mf.get_dispersion()
        print(e_disp)
        self.assertAlmostEqual(e_disp, -0.030566786972, 9)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_uhf_d4(self):
        mol = make_disp_mol()
        mf = scf.UHF(mol)
        mf.disp = 'd4'
        e_disp = mf.get_dispersion()
        print(e_disp)
        self.assertAlmostEqual(e_disp, -0.0096708308236, 9)

    def test_scf_negative_spin(self):
        mol = gto.M(atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            basis = '6-31g',
            spin = -2,
        )
        mf = scf.UHF(mol).run(conv_tol=1e-10)
        self.assertAlmostEqual(mf.mo_occ[1].sum(), 6, 14)
        self.assertAlmostEqual(mf.e_tot, -75.726396909036637, 9)

        mol = gto.M(atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            symmetry = 1,
            basis = '6-31g',
            spin = -2,
        )
        mf = scf.UHF(mol).set(conv_tol=1e-10)
        mf.irrep_nelec = {'B2': (1, 2), 'B1': (1, 0)}
        mf.run()
        self.assertAlmostEqual(mf.mo_occ[1].sum(), 6, 14)
        self.assertAlmostEqual(mf.e_tot, -75.224503772055755, 9)

    def test_nr_uhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.UHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

    def test_nr_uhf_symm_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        pmol.symmetry = 1
        pmol.build()
        mf = scf.UHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

    def test_spin_square(self):
        self.assertAlmostEqual(mf.spin_square(mf.mo_coeff)[0], 0, 9)

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
        mf.irrep_nelec = {'B2':(2,1)}
        self.assertAlmostEqual(mf.scf(), -75.010623169610966, 9)

    def test_n2_symm(self):
        mf = scf.uhf_symm.UHF(n2sym)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)

        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.uhf_symm.UHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.34691774091894, 9)

    def test_n2_symm_uhf_fixnocc(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.uhf_symm.UHF(pmol)
        mf.irrep_nelec = {'A1g':6, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -108.22558478425401, 9)
        mf.irrep_nelec = {'A1g':(3,3), 'A1u':(2,1), 'E1ux':(1,1), 'E1uy':(1,1)}
        self.assertAlmostEqual(mf.scf(), -108.22558478425401, 9)

    def test_uhf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=8, spin=2)
        mf = scf.uhf.UHF(mol)
        energy = numpy.array(([-10, -1, 1, -2, 0, -3], [8, 2, 4, 3, 0, 5]))
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                                       ([1, 1, 0, 1, 1, 1], [0, 1, 0, 1, 1, 0])))
        pmol = n2sym.copy()
        pmol.spin = 2
        pmol.symmetry = False
        mf = scf.UHF(pmol).set(verbose = 0)
        energy = numpy.array([[34, 2 , 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7],
                              [29, 26, 13, 54, 18, 78, 85, 49, 84, 62, 42, 74, 20, 61, 51, 34, 2 , 33, 52, 3]])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [[0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))
        mol.stdout.close()

    def test_uhf_symm_get_occ(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0)
        orbsym = numpy.array([[0 , 5 , 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5],
                              [5 , 0 , 6 , 7 , 5 , 10, 11, 0 , 5 , 0 , 5 , 5 , 6 , 7 , 0 , 2 , 3 , 0 , 2 , 3]])
        energy = numpy.array([[34, 2 , 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7],
                              [29, 26, 13, 54, 18, 78, 85, 49, 84, 62, 42, 74, 20, 61, 51, 34, 2 , 33, 52, 3]])
        mf.irrep_nelec = {'A1g':7, 'A1u':3, 'E1ux':2, 'E1uy':2}
        mo_coeff = lib.tag_array([numpy.eye(energy.size)]*2, orbsym=orbsym)
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
                 [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]))
        mf.irrep_nelec = {'A1g':(5,2), 'A1u':(1,2)}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [[0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]))

    def test_uhf_symm_dump_flags(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

    def test_det_ovlp(self):
        s, x = mf.det_ovlp(mf.mo_coeff, mf.mo_coeff, mf.mo_occ, mf.mo_occ)
        self.assertAlmostEqual(s, 1.000000000, 9)
        self.assertAlmostEqual(numpy.trace(x[0]), mol.nelec[0]*1.000000000, 9)
        self.assertAlmostEqual(numpy.trace(x[0]), mol.nelec[1]*1.000000000, 9)

    def test_dip_moment(self):
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_get_wfnsym(self):
        self.assertEqual(n2mf.wfnsym, 0)

        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.UHF(pmol).set(verbose = 0).run()
        self.assertTrue(mf.wfnsym in (2, 3))

    def test_complex_orbitals(self):
        nao = mol.nao_nr()
        mf = scf.UHF(mol)
        mf.kernel(numpy.zeros((2,nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.UHF(mol).set(max_memory=0)
        mf.kernel(numpy.zeros((2,nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_make_asm_dm(self):
        mo_ba = (mf2.mo_coeff[1], mf2.mo_coeff[0])
        mo_occ = mf.mo_occ
        det, x = mf2.det_ovlp(mf2.mo_coeff, mo_ba, mo_occ, mo_occ)
        s = mf2.get_ovlp()
        self.assertAlmostEqual(det, 0.95208556738844452, 6)
        dm = mf2.make_asym_dm(mf2.mo_coeff, mo_ba, mo_occ, mo_occ, x)
        self.assertAlmostEqual(numpy.einsum('ij,ji', s, dm[0]), 5, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji', s, dm[1]), 5, 9)

    def test_analyze(self):
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.05844441822, 5)
        (pop, chg), dip = mf.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 3.2031790129016922, 6)

        mf1 = n2mf.copy()
        (pop, chg), dip = n2mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 4.5467414321488357, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 0, 9)
        mf1.mo_coeff = numpy.array(n2mf.mo_coeff)
        (pop, chg), dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(pop[0]+pop[1]), 3.8893148995392353, 6)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.uhf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 398.09239104094513, 9)

        pmol = gto.Mole()
        pmol.atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587'''
        pmol.basis = '6-31g'
        pmol.cart = True

        mf1 = scf.uhf.UHF(pmol)
        mf1.direct_scf = True
        mf1.max_memory = 0
        nao = pmol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,3,nao,nao)) - .5 + 0j
        vhf3a = mf1.get_veff(pmol, dm[:,0], hermi=0)
        vhf3b = mf1.get_veff(pmol, dm[:,1], hermi=0)
        vhf3c = mf1.get_veff(pmol, dm[:,2], hermi=0)
        vhf3 = numpy.array((vhf3a, vhf3b, vhf3c)).transpose(1,0,2,3)

        vhf4 = mf1.get_veff(pmol, dm, hermi=0)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(abs(vhf4-vhf3).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vhf4), -9.9614575705134953, 12)

    def test_natm_eq_0(self):
        mol = gto.M()
        mol.spin = 2
        mol.nelectron = 2
        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: numpy.diag(numpy.arange(3))
        mf.get_ovlp = lambda *args: numpy.eye(3)
        mf._eri = numpy.zeros((3,3,3,3))
        for i in range(3):
            mf._eri[i,i,i,i] = .2
        dm = mf.get_init_guess(mol, key='hcore')
        self.assertTrue(numpy.allclose(dm[0].diagonal(), [1,1,0]))
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, 1.0, 9)

    def test_canonicalize(self):
        mo_coeff = numpy.array(n2mf.mo_coeff)
        e, c = n2mf.canonicalize(mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(abs(e - n2mf.mo_energy).max(), 0, 6)

        n2_uhf = n2mf.view(scf.uhf.UHF)
        e, c = n2_uhf.canonicalize(n2mf.mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(abs(e - n2mf.mo_energy).max(), 0, 6)

    def test_energy_tot1(self):
        e = n2mf.energy_tot(n2mf.make_rdm1())
        self.assertAlmostEqual(e, n2mf.e_tot, 9)

    def test_get_occ_extreme_case(self):
        mol = gto.M(atom='He', verbose=7, output='/dev/null')
        mf = scf.UHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, -2.8077839575399737, 12)

        mol.charge = 2
        mf = scf.UHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, 0, 12)
        mol.stdout.close()

    def test_damping(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        f = numpy.asarray([scf.hf.get_hcore(mol)]*2)
        df  = numpy.random.rand(2,nao,nao)
        f_prev = f + df
        damp = 0.3
        f_damp = scf.uhf.get_fock(mf, h1e=0, s1e=0, vhf=f, dm=0, cycle=0,
                                 diis_start_cycle=2, damp_factor=damp, fock_last=f_prev)
        self.assertAlmostEqual(abs(f_damp[0] - (f[0]*(1-damp) + f_prev[0]*damp)).max(), 0, 9)
        self.assertAlmostEqual(abs(f_damp[1] - (f[1]*(1-damp) + f_prev[1]*damp)).max(), 0, 9)

    def test_get_irrep_nelec(self):
        fock = n2mf.get_fock()
        s1e = n2mf.get_ovlp()
        e, c = n2mf.eig(fock, s1e)
        mo_occ = n2mf.get_occ(e, c)
        n2_uhf = n2mf.view(scf.uhf_symm.UHF)
        irrep_nelec = n2_uhf.get_irrep_nelec(n2sym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], (2,2))
        self.assertEqual(irrep_nelec['A1g'], (3,3))
        self.assertEqual(irrep_nelec['E1ux'], (1,1))
        self.assertEqual(irrep_nelec['E1uy'], (1,1))
        mo_coeff = numpy.array(c)
        irrep_nelec = n2_uhf.get_irrep_nelec(n2sym, mo_coeff, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], (2,2))
        self.assertEqual(irrep_nelec['A1g'], (3,3))
        self.assertEqual(irrep_nelec['E1ux'], (1,1))
        self.assertEqual(irrep_nelec['E1uy'], (1,1))

        n2_uhf.irrep_nelec = irrep_nelec
        n2_uhf.irrep_nelec['A2g'] = 0
        n2_uhf.nelec = (8,6)
        self.assertRaises(ValueError, n2_uhf.build)
        n2_uhf.irrep_nelec['A1g'] = (2,2)
        n2_uhf.irrep_nelec['E2gx'] = 0
        n2_uhf.irrep_nelec['E2gy'] = 0
        n2_uhf.irrep_nelec['E2ux'] = 0
        n2_uhf.irrep_nelec['E2uy'] = 0
        self.assertRaises(ValueError, n2_uhf.build)
        n2_uhf.irrep_nelec['A1g'] = (2,0)
        self.assertRaises(ValueError, n2_uhf.build)
        n2_uhf.irrep_nelec['A1g'] = (0,1)
        self.assertRaises(ValueError, n2_uhf.build)

        n2_uhf.irrep_nelec.pop('E2ux')
        n2_uhf.irrep_nelec['A1g'] = (0,0)
        self.assertRaises(ValueError, n2_uhf.build)

    def test_max_cycle0(self):
        mf = scf.UHF(mol)
        mf.max_cycle = 0
        dm = mf.get_init_guess()
        mf.kernel(dm)
        self.assertAlmostEqual(mf.e_tot, -75.799022820714526, 9)

        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, -75.983602246415373, 9)

    def test_custom_h1e(self):
        h1 = scf.hf.get_hcore(n2sym)
        s1 = scf.hf.get_ovlp(n2sym)
        mf = scf.UHF(n2sym)
        mf.get_hcore = lambda *args: (h1, h1)
        e = mf.kernel()
        self.assertAlmostEqual(e, -108.9298383856092, 9)

        mf = scf.uhf.HF1e(n2sym)
        mf.get_hcore = lambda *args: (h1, h1)
        eref = scipy.linalg.eigh(h1, s1)[0][0] + n2sym.energy_nuc()
        e = mf.kernel()
        self.assertAlmostEqual(e, eref, 9)


if __name__ == "__main__":
    print("Full Tests for uhf")
    unittest.main()
