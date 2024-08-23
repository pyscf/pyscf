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

import unittest
from functools import reduce
import numpy
import scipy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import fci

def setUpModule():
    global mol, mfr, mcr, mfu, mcu, mcr_prg, mcr_prb, mfr_prg, mfr_prb, mcu_prg, mfu_prg
    b = 1.4
    mol = gto.Mole()
    mol.build(
    verbose = 7,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    symmetry = 1
    )
    mfr = scf.RHF(mol)
    mfr.scf()
    mcr = mcscf.CASSCF(mfr, 4, 4)
    mcr.conv_tol_grad = 1e-6
    mcr.mc1step()[0]

    mfu = scf.UHF(mol)
    mfu.scf()
    mcu = mcscf.UCASSCF(mfu, 4, 4)
    mcu.conv_tol_grad = 1e-6
    mcu.mc1step()[0]

    mol_prg = gto.M(
    verbose = 0,
    atom = [
        ['N',(  0.000000,  0.000000, -(b+0.1)/2)],
        ['N',(  0.000000,  0.000000,  (b+0.1)/2)], ],
    basis = 'ccpvdz',
    symmetry=1)
    mfr_prg = scf.RHF(mol_prg).set (max_cycle=1).run()
    mcr_prg = mcscf.CASSCF(mfr_prg, 4, 4).set (max_cycle_macro=1).run()
    mfu_prg = scf.UHF(mol_prg).set (max_cycle=1).run()
    mcu_prg = mcscf.UCASSCF(mfu_prg, 4, 4).set (max_cycle_macro=1).run()

    mol_prb = mol.copy ()
    mol_prb.basis = {'N': 'aug-cc-pvdz' }
    mol_prb.build ()
    mfr_prb = scf.RHF(mol_prb).set (max_cycle=1).run()
    mcr_prb = mcscf.CASSCF(mfr_prb, 4, 4).set (max_cycle_macro=1).run()


def tearDownModule():
    global mol, mfr, mcr, mfu, mcu, mcr_prg, mcr_prb, mfr_prg, mfr_prb, mcu_prg, mfu_prg
    mol.stdout.close()
    del mol, mfr, mcr, mfu, mcu, mcr_prg, mcr_prb, mfr_prg, mfr_prb, mcu_prg, mfu_prg


class KnownValues(unittest.TestCase):
    def test_spin_square(self):
        ss = mcscf.addons.spin_square(mcr)[0]
        self.assertAlmostEqual(ss, 0, 7)

    def test_ucasscf_spin_square(self):
        ss = mcscf.addons.spin_square(mcu)[0]
        self.assertAlmostEqual(ss, 0, 7)

    def test_rcas_natorb(self):
        mo1, ci1, mocc1 = mcscf.addons.cas_natorb(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(mo1)  , 9.9260608594977491, 5)
        self.assertAlmostEqual(numpy.linalg.norm(mocc1), 5.1687145190800079, 5)

#TODO:    def test_ucas_natorb(self):
#TODO:        mo2, ci2, mocc2 = mcscf.addons.cas_natorb(mcu)
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(mo2)  , 11.4470460817871*numpy.sqrt(2), 7)
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(mocc2), 2.59144951056707/numpy.sqrt(2), 7)

    def test_get_fock(self):
        f1 = mcscf.addons.get_fock(mcr)
        self.assertTrue(numpy.allclose(f1, f1.T))
        self.assertAlmostEqual(numpy.linalg.norm(f1), 25.482177562349467, 5)
#TODO:        f1 = mcscf.addons.get_fock(mcu)
#TODO:        self.assertTrue(numpy.allclose(f1[0], f1[0].T))
#TODO:        self.assertTrue(numpy.allclose(f1[1], f1[1].T))
#TODO:        self.assertAlmostEqual(numpy.linalg.norm(f1), 23.597476504476919*numpy.sqrt(2), 6)

    def test_canonicalize1(self):
        numpy.random.seed(1)
        f1 = numpy.random.random(mcr.mo_coeff.shape)
        u1 = numpy.linalg.svd(f1)[0]
        mo1 = numpy.dot(mcr.mo_coeff, u1)
        mo1 = lib.tag_array(mo1, orbsym=mcr.mo_coeff.orbsym)
        mo, ci, mo_e = mcr.canonicalize(mo1)
        e1 = numpy.einsum('ji,jk,ki', mo, f1, mo)
        self.assertAlmostEqual(e1, 44.2658681077, 7)
        self.assertAlmostEqual(lib.fp(mo_e), 5.1364166175063097, 5)

        mo, ci, mo_e = mcr.canonicalize(mo1, eris=mcr.ao2mo(mcr.mo_coeff))
        e1 = numpy.einsum('ji,jk,ki', mo, f1, mo)
        self.assertAlmostEqual(e1, 44.2658681077, 7)
        self.assertAlmostEqual(lib.fp(mo_e), 4.1206025804989173, 4)

        mcr1 = mcr.copy()
        mcr1.frozen = 2
        mo, ci, mo_e = mcr1.canonicalize(mo1)
        self.assertAlmostEqual(lib.fp(mo_e), 6.6030999409178577, 5)

        mcr1.frozen = [0,1]
        mo, ci, mo_e = mcr1.canonicalize(mo1)
        self.assertAlmostEqual(lib.fp(mo_e), 6.6030999409178577, 5)

        mcr1.frozen = [1,12]
        mo, ci, mo_e = mcr1.canonicalize(mo1)
        self.assertAlmostEqual(lib.fp(mo_e), 5.2182584355788162, 5)

    def test_canonicalize(self):
        mo, ci, mo_e = mcr.canonicalize()
        self.assertAlmostEqual(numpy.linalg.norm(mo), 9.9260608594977242, 7)
        mo, ci, mo_e = mcr.canonicalize(eris=mcr.ao2mo(mcr.mo_coeff))
        self.assertAlmostEqual(numpy.linalg.norm(mo), 9.9260608594977242, 7)

    def test_make_rdm12(self):
        dmr = mcscf.addons.make_rdm1(mcr)
        dm1, dm2 = mcscf.addons.make_rdm12(mcr)
        self.assertTrue(numpy.allclose(dmr, dm1))
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 3.8205551262007567, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dm2), 14.987267883423314, 5)

    def test_make_rdm1s(self):
        dm1 = mcscf.addons.make_rdm1s(mcr)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7015404376335805, 5)
        dm1 = mcscf.addons.make_rdm1s(mcu)
        self.assertAlmostEqual(numpy.linalg.norm(dm1), 2.7015404376335805, 5)

    def test_sort_mo(self):
        mo1 = numpy.arange(mfr.mo_energy.size).reshape(1,-1)
        ref = [[0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        mo2 = mcscf.addons.sort_mo(mcr, mo1, [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, ref))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [5,6,7,9])
        self.assertTrue(numpy.allclose(mo2, (ref,ref)))
        mo2 = mcscf.addons.sort_mo(mcu, (mo1,mo1), [[5,6,7,9],[5,6,8,9]])
        ref1 = [[0, 1, 2, 3, 6, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
        self.assertTrue(numpy.allclose(mo2, (ref,ref1)))

    def test_sort_mo_by_irrep(self):
        mc1 = mcscf.CASSCF(mfr, 8, 4)
        mo0 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {'E1ux':2, 'E1uy':2, 'E1gx':2, 'E1gy':2})
        mo1 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {2:2, 3:2, 6:2, 7:2}, {2:0, 3:0, 6:0, 7:0})
        mo2 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, (0,0,2,2,0,0,2,2))
        mo3 = mcscf.sort_mo_by_irrep(mc1, mfr.mo_coeff, {'E1ux':2, 'E1uy':2, 2:2, 3:2})
        self.assertTrue(numpy.allclose(mo0, mo1))
        self.assertTrue(numpy.allclose(mo0, mo2))
        self.assertTrue(numpy.allclose(mo0, mo3))

    def test_sort_mo_by_irrep1(self):
        mol = gto.M(atom='N 0 0 -.45; N 0 0 .45', basis='ccpvdz',
                    symmetry=True, verbose=0)
        mf = scf.RHF(mol).run()
        mc1 = mcscf.CASSCF(mf, 6, 6)
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'A1g': 1, 'A1u': 1, 'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1g': 2, 'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1g': 2, 'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'E1uy': 1, 'E1ux': 1, 'E1gy': 1, 'E1gx': 1},
                {'A1u': 2})
        self.assertEqual(list(caslst), [4,5,7,8,9,10])
        caslst = mcscf.addons.caslst_by_irrep(mc1, mf.mo_coeff,
                {'A1g': 1, 'A1u': 1}, {'E1uy': 1, 'E1ux': 1})
        self.assertEqual(list(caslst), [3,6,8,9,12,13])

        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'A1g': 1, 'A1u': 1}, {'E1uy': 3, 'E1ux': 3})
        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'A1g': 3, 'A1u': 4}, {'E1uy': 1, 'E1ux': 1})
        self.assertRaises(ValueError, mcscf.addons.caslst_by_irrep, mc1, mf.mo_coeff,
                          {'E2ux': 2, 'E2uy': 2}, {'E1uy': 1, 'E1ux': 1})

    def test_make_natural_orbitals_from_restricted(self):
        from pyscf import mp, ci, cc
        npt = numpy.testing

        mol = gto.M(atom = 'C 0 0 0; O 0 0 1.2', basis = '3-21g', spin = 0)
        myhf = scf.RHF(mol).run()
        ncas, nelecas = (8, 10)
        mymc = mcscf.CASCI(myhf, ncas, nelecas)

        # Test MP2
        # Trusted results from ORCA v4.2.1
        rmp2_noons = [1.99992732,1.99989230,1.99204357,1.98051334,1.96825487,1.94377615,1.94376239,0.05792320,0.05791037,0.02833335,0.00847013,0.00531989,0.00420320,0.00420280,0.00257965,0.00101638,0.00101606,0.00085503]
        mymp = mp.MP2(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
        npt.assert_array_almost_equal(rmp2_noons, noons, decimal=5)
        mymc.kernel(natorbs)

        # The tests below are only to ensure that `make_natural_orbitals` can 
        # run at all since we've confirmed above that the NOONs are correct.
        # Test CISD
        mycisd = ci.CISD(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(mycisd)
        mymc.kernel(natorbs)

        # Test CCSD
        myccsd = cc.CCSD(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(myccsd)
        mymc.kernel(natorbs)

    def test_make_natural_orbitals_from_unrestricted(self):
        from pyscf import mp, ci, cc
        npt = numpy.testing

        mol = gto.M(atom = 'O 0 0 0; O 0 0 1.2', basis = '3-21g', spin = 2)
        myhf = scf.UHF(mol).run()
        ncas, nelecas = (8, 12)
        mymc = mcscf.CASCI(myhf, ncas, nelecas)

        # Test UHF
        # The tests below are only to ensure that `make_natural_orbitals` can 
        # run at all since we've confirmed above that the NOONs are correct for MP2
        mcscf.addons.make_natural_orbitals(myhf)

        # Test MP2
        # Trusted results from ORCA v4.2.1
        rmp2_noons = [1.99992786,1.99992701,1.99414062,1.98906552,1.96095173,1.96095165,1.95280755,1.02078458,1.02078457,0.04719006,0.01274288,0.01274278,0.00728679,0.00582683,0.00543964,0.00543962,0.00290772,0.00108258]
        mymp = mp.UMP2(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
        npt.assert_array_almost_equal(rmp2_noons, noons)
        mymc.kernel(natorbs)

        # The tests below are only to ensure that `make_natural_orbitals` can 
        # run at all since we've confirmed above that the NOONs are correct.
        # Test CISD
        mycisd = ci.CISD(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(mycisd)
        mymc.kernel(natorbs)

        # Test CCSD
        myccsd = cc.CCSD(myhf).run()
        noons, natorbs = mcscf.addons.make_natural_orbitals(myccsd)
        mymc.kernel(natorbs)

    def test_state_average(self):
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.fcisolver = fci.solver(mol, singlet=False)
        mc.state_average_((.64,.36))
        e = mc.kernel()
        e = mc.e_states
        self.assertAlmostEqual(mc.e_tot, -108.83342083775061, 7)
        self.assertAlmostEqual(mc.e_average, -108.83342083775061, 7)
        self.assertAlmostEqual(e[0]*.64+e[1]*.36, -108.83342083775061, 7)
        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.52396929381500434, 4)

        mc = mc.state_average((.64,.36)).run()
        self.assertAlmostEqual(mc.e_tot, -108.83342083775061, 7)
        self.assertAlmostEqual(mc.e_average, -108.83342083775061, 7)

    def test_state_average_fci_dmrg(self):
        fcisolver1 = fci.direct_spin1_symm.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_spin1_symm.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            def approx_kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            @property
            def orbsym(self):
                return fcisolver1.orbsym
            @orbsym.setter
            def orbsym(self, x):
                fcisolver1.orbsym = x
            spin_square = None
            large_ci = None
            transform_ci_for_orbital_rotation = None

        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.fcisolver = FCI_as_DMRG(mol)
        mc.fcisolver.nroots =  fcisolver1.nroots = 2
        mc.state_average_((.64,.36))
        mc.kernel()
        e = mc.e_states
        self.assertAlmostEqual(e[0]*.64+e[1]*.36, -108.83342083775061, 7)
        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.52396929381500434*2, 4)

    def test_state_average_mix(self):
        solver1 = fci.FCI(mol)
        solver1.spin = 0
        solver1.nroots = 2
        solver2 = fci.FCI(mol, singlet=False)
        solver2.wfnsym = 'A1u'
        solver2.spin = 2
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc = mcscf.addons.state_average_mix_(mc, [solver1, solver2],
                                             (0.25,0.25,0.5))
        mc.kernel()
        e = mc.e_states
        self.assertAlmostEqual(mc.e_tot, -108.80340952016508, 7)
        self.assertAlmostEqual(mc.e_average, -108.80340952016508, 7)
        self.assertAlmostEqual(numpy.dot(e,[.25,.25,.5]), -108.80340952016508, 7)
        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.52172669549357464, 4)
        self.assertAlmostEqual(lib.fp(dm1[1]), 0.53366776017869022, 4)
        self.assertAlmostEqual(lib.fp(dm1[0]+dm1[1]), 1.0553944556722636, 4)

        mc.cas_natorb()

    def test_state_average_mix_fci_dmrg(self):
        fcisolver1 = fci.direct_spin0_symm.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_spin0_symm.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            def approx_kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            @property
            def orbsym(self):
                return fcisolver1.orbsym
            @orbsym.setter
            def orbsym(self, x):
                fcisolver1.orbsym = x
            spin_square = None
            large_ci = None
            transform_ci_for_orbital_rotation = None

        solver1 = FCI_as_DMRG(mol)
        solver1.spin =    fcisolver1.spin = 0
        solver1.nroots =  fcisolver1.nroots = 2
        solver2 = fci.FCI(mol, singlet=False)
        solver2.wfnsym = 'A1u'
        solver2.spin = 2
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc = mcscf.addons.state_average_mix_(mc, [solver1, solver2],
                                             (0.25,0.25,0.5))
        mc.kernel()
        e = mc.e_states
        self.assertAlmostEqual(numpy.dot(e, [.25,.25,.5]), -108.80340952016508, 7)
        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 1.0553944556722636, 4)
        self.assertEqual(dm1[1], None)

        mc.cas_natorb()

    def test_state_specific(self):
        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.fcisolver = fci.solver(mol, singlet=False)
        mc.state_specific_(state=1)
        e = mc.kernel()[0]
        self.assertAlmostEqual(e, -108.70065770892457, 7)
        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 0.54605283139098515, 4)

        mc = mcscf.CASSCF(mfr, 4, 4)
        mc.state_specific_(state=0)
        e = mc.kernel()[0]
        self.assertAlmostEqual(mc.e_tot, mcr.e_tot, 7)
        dm1 = mc.analyze()
        dmref = mcr.analyze()
        self.assertAlmostEqual(float(abs(dm1[0]-dmref[0]).max()), 0, 4)

    def test_project_init_guess_geom (self):
        mfr_mo_norm = numpy.einsum ('ip,ip->p', mfr.mo_coeff.conj (),
            mfr_prg.get_ovlp ().dot (mfr.mo_coeff))
        mfr_mo_norm = mfr.mo_coeff / numpy.sqrt (mfr_mo_norm)[None,:]
        mo1 = mcscf.addons.project_init_guess (mcr_prg, mfr.mo_coeff)
        s1 = reduce(numpy.dot, (mo1.T, mfr_prg.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)

    def test_project_init_guess_basis (self):
        mo1 = mcscf.addons.project_init_guess (mcr_prb, mfr.mo_coeff, prev_mol=mfr.mol)
        s1 = reduce(numpy.dot, (mo1.T, mfr_prb.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 6.782329983125268, 9)

    def test_project_init_guess_uhf (self):
        mo1_u = mcscf.addons.project_init_guess (mcu_prg, mfu.mo_coeff)
        for mo1 in mo1_u:
            s1 = reduce(numpy.dot, (mo1.T, mfu_prg.get_ovlp(), mo1))
            self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                             s1.shape[0])
            self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)

    def test_project_init_guess_activefirst (self):
        with lib.temporary_env (mcr_prg, ncas=6, ncore=3):
            mo1 = mcscf.addons.project_init_guess (mcr_prg, mfr.mo_coeff, priority='active')
        s1 = reduce(numpy.dot, (mo1.T, mfr_prg.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)
        mfr_mo_norm = numpy.einsum ('ip,ip->p', mfr.mo_coeff.conj (),
            mfr_prg.get_ovlp ().dot (mfr.mo_coeff))
        mfr_mo_norm = mfr.mo_coeff / numpy.sqrt (mfr_mo_norm)[None,:]
        s2 = [reduce (numpy.dot, (mfr_prg.get_ovlp (), mo1[:,i], mfr_mo_norm[:,i]))
            for i in (1,3)] # core, active (same irrep)
        self.assertAlmostEqual (s2[1], 1.0, 9)
        self.assertFalse (s2[0] > s2[1])

    def test_project_init_guess_corefirst (self):
        with lib.temporary_env (mcr_prg, ncas=6, ncore=3):
            mo1 = mcscf.addons.project_init_guess (mcr_prg, mfr.mo_coeff, priority='core')
        s1 = reduce(numpy.dot, (mo1.T, mfr_prg.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)
        mfr_mo_norm = numpy.einsum ('ip,ip->p', mfr.mo_coeff.conj (),
            mfr_prg.get_ovlp ().dot (mfr.mo_coeff))
        mfr_mo_norm = mfr.mo_coeff / numpy.sqrt (mfr_mo_norm)[None,:]
        s1 = [reduce (numpy.dot, (mfr_prg.get_ovlp (), mo1[:,i], mfr_mo_norm[:,i]))
            for i in (1,3)] # core, active (same irrep)
        self.assertAlmostEqual (s1[0], 1.0, 9)
        self.assertTrue (s1[0] > s1[1])

    def test_project_init_guess_gramschmidt (self):
        gram_schmidt_idx = numpy.arange (27, dtype=int)[:,None].tolist ()
        mo1 = mcscf.addons.project_init_guess (mcr_prg, mfr.mo_coeff, priority=gram_schmidt_idx)
        s1 = reduce(numpy.dot, (mo1.T, mfr_prg.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)
        mf2moi = reduce (numpy.dot, (mfr_prg.mo_coeff.conj ().T, mfr_prg.get_ovlp (), mfr.mo_coeff))
        Q, R = scipy.linalg.qr (mf2moi) # Arbitrary sign, so abs below
        mo2 = numpy.dot (mfr_prg.mo_coeff, Q)
        s2 = numpy.abs (reduce (numpy.dot, (mo1.conj ().T, mfr_prg.get_ovlp (), mo2)))
        self.assertAlmostEqual(numpy.linalg.norm(s2), 5.2915026221291841, 9)

    def test_project_init_guess_prioritylists (self):
        pr = [[[27],[5,3],[6,12]],[[5],[17],[13,10,8,6]]]
        mo1_u = mcscf.addons.project_init_guess (mcu_prg, mfu.mo_coeff, priority=pr)
        s0 = mfu_prg.get_ovlp ()
        for ix, mo1 in enumerate (mo1_u):
            s1 = reduce(numpy.dot, (mo1.T, s0, mo1))
            self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
            self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)
            mfu_mo = mfu.mo_coeff[ix]
            mfu_mo_norm = numpy.einsum ('ip,ip->p', mfu_mo.conj (), s0.dot (mfu_mo))
            mfu_mo_norm = mfu.mo_coeff[ix] / numpy.sqrt (mfu_mo_norm)[None,:]
            p = pr[ix][0][0]
            s2 = reduce (numpy.dot, (mfu_prg.get_ovlp (), mo1[:,p], mfu_mo_norm[:,p]))
            self.assertAlmostEqual (s2, 1.0, 9)

    def test_project_init_guess_usehfcore (self):
        mo1 = mcscf.addons.project_init_guess (mcr_prg, mfr.mo_coeff, use_hf_core=True)
        s1 = reduce(numpy.dot, (mo1.T, mfr_prg.get_ovlp(), mo1))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s1)[0]>1e-10),
                         s1.shape[0])
        self.assertAlmostEqual(numpy.linalg.norm(s1), 5.2915026221291841, 9)
        s2 = reduce (numpy.dot, (mo1[:,:5].T, mfr_prg.get_ovlp (), mfr_prg.mo_coeff[:,:5]))
        self.assertEqual(numpy.count_nonzero(numpy.linalg.eigh(s2)[0]>1e-10),
                         s2.shape[0])
        self.assertAlmostEqual (numpy.linalg.norm (s2), 2.23606797749979, 9)


    def test_state_average_bad_init_guess(self):
        mc = mcscf.CASCI(mfr, 4, 4)
        mc.run()
        mc.state_average_([.8, .2])
        mscan = mc.as_scanner()
        e = mscan(mol)
        self.assertAlmostEqual(e, -108.84390277715984, 8)


if __name__ == "__main__":
    print("Full Tests for mcscf.addons")
    unittest.main()
