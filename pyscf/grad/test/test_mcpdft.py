#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Test API:
#   0. Initialize from mol, mf, and mc (done)
#   1. kernel (done)
#   2. optimize_mcscf_ (done)
#   3. compute_pdft_ (done)
#   4. energy_tot (done) 
#   5. get_energy_decomposition (done)
#   6. checkpoint stuff
#   7. get_pdft_veff (maybe this elsewhere?)
# In the context of:
#   1. CASSCF, CASCI
#   2. Symmetry, with and without
#   3. State average, state average mix w/ different spin states

# Some assertAlmostTrue thresholds are loose because we are only
# trying to test the API here; we need tight convergence and grids
# to reproduce well when OMP is on.
import numpy as np
from pyscf import gto, scf, mcscf, lib, fci, dft
from pyscf import mcpdft
import unittest


mol_nosym = mol_sym = mf_nosym = mf_sym = mc_nosym = mc_sym = mcp = None
def auto_setup (xyz='Li 0 0 0\nH 1.5 0 0'):
    mol_nosym = gto.M (atom = xyz, basis = 'sto3g',
                       output = '/dev/null', verbose = 0)
    mol_sym = gto.M (atom = xyz, basis = 'sto3g', symmetry=True,
                     output = '/dev/null', verbose = 0)
    mf_nosym = scf.RHF (mol_nosym).run ()
    mc_nosym = mcscf.CASSCF (mf_nosym, 5, 2).run ()
    mf_sym = scf.RHF (mol_sym).run ()
    mc_sym = mcscf.CASSCF (mf_sym, 5, 2).run ()
    mcp_ss_nosym = mcpdft.CASSCF (mc_nosym, 'ftLDA,VWN3', 5, 2,
                                  grids_level=1).run ()
    mcp_ss_sym = mcpdft.CASSCF (mc_sym, 'ftLDA,VWN3', 5, 2,
                                grids_level=1).run ()
    mcp_sa_0 = mcp_ss_nosym.state_average ([1.0/5,]*5).run ()
    solver_S = fci.solver (mol_nosym, singlet=True).set (spin=0, nroots=2)
    solver_T = fci.solver (mol_nosym, singlet=False).set (spin=2, nroots=3)
    mcp_sa_1 = mcp_ss_nosym.state_average_mix (
        [solver_S,solver_T], [1.0/5,]*5).set(ci=None).run ()
    solver_A1 = fci.solver (mol_sym).set (wfnsym='A1', nroots=3)
    solver_E1x = fci.solver (mol_sym).set (wfnsym='E1x', nroots=1, spin=2)
    solver_E1y = fci.solver (mol_sym).set (wfnsym='E1y', nroots=1, spin=2)
    mcp_sa_2 = mcp_ss_sym.state_average_mix (
        [solver_A1,solver_E1x,solver_E1y], [1.0/5,]*5).set(ci=None).run ()
    mcp = [[mcp_ss_nosym, mcp_ss_sym], [mcp_sa_0, mcp_sa_1, mcp_sa_2]]
    nosym = [mol_nosym, mf_nosym, mc_nosym]
    sym = [mol_sym, mf_sym, mc_sym]
    return nosym, sym, mcp

def setUpModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, original_grids
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
    nosym, sym, mcp = auto_setup ()
    mol_nosym, mf_nosym, mc_nosym = nosym
    mol_sym, mf_sym, mc_sym = sym

def tearDownModule():
    global mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, original_grids
    mol_nosym.stdout.close ()
    mol_sym.stdout.close ()
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    del mol_nosym, mf_nosym, mc_nosym, mol_sym, mf_sym, mc_sym, mcp, original_grids

class KnownValues(unittest.TestCase):

    def test_scanner (self):
        mcp1 = auto_setup (xyz='Li 0 0 0\nH 1.55 0 0')[-1]
        for mol0, mc0, mc1 in zip ([mol_nosym, mol_sym], mcp[0], mcp1[0]):
            mc0_grad = mc0.nuc_grad_method ()
            mc1_gradscanner = mc1.nuc_grad_method ().as_scanner ()
            de0 = lib.fp (mc0_grad.kernel ())
            e1, de1 = mc1_gradscanner (mol0)
            de1 = lib.fp (de1) 
            with self.subTest (case='SS', symm=mol0.symmetry):
                self.assertTrue(mc0_grad.converged)
                self.assertTrue(mc1_gradscanner.converged)
                self.assertAlmostEqual (de0, de1, delta=1e-6)
        for ix, (mc0, mc1) in enumerate (zip (mcp[1], mcp1[1])):
            tms = (0,1,'mixed')[ix]
            sym = bool (ix//2)
            mol0 = [mol_nosym, mol_sym][int(sym)]
            mc0_grad = mc0.nuc_grad_method ()
            mc1_gradscanner = mc1.nuc_grad_method ().as_scanner ()
            for state in range (5):
                with self.subTest (case='SA', state=state, symm=mol0.symmetry, triplet_ms=tms):
                    de0 = lib.fp (mc0_grad.kernel (state=state))
                    e1, de1 = mc1_gradscanner (mol0, state=state)   
                    de1 = lib.fp (de1)
                    self.assertTrue(mc0_grad.converged)
                    self.assertTrue(mc1_gradscanner.converged)
                    self.assertAlmostEqual (de0, de1, delta=1e-5)

    def test_gradients (self):
        ref_ss = 5.29903936e-03
        ref_sa = [5.66392595e-03,3.67724051e-02,3.62698260e-02,2.53851408e-02,2.53848341e-02]
        # Source: numerical @ this program
        for mc, symm in zip (mcp[0], (False, True)):
            with self.subTest (case='SS', symmetry=symm):
                de = mc.nuc_grad_method ().kernel ()[0,0]
                self.assertAlmostEqual (de, ref_ss, 6)
        for ix, mc in enumerate (mcp[1]):
            tms = (0,1,'mixed')[ix]
            sym = bool (ix//2)
            mc_grad = mc.nuc_grad_method ()
            for state in range (5):
                with self.subTest (case='SA', state=state, symmetry=sym, triplet_ms=tms):
                    i = np.argsort (mc.e_states)[state]
                    de = mc_grad.kernel (state=i)[0,0]
                    self.assertAlmostEqual (de, ref_sa[state], 5)

    def test_triplet_mol (self):
        '''Check that energies & gradients do not depend on if the parent MF is RHF or ROHF'''
        mc = mcpdft.CASSCF (mf_nosym, 'ftLDA,VWN3', 5, (2,0),
                            grids_level=1).run (mo_coeff=mcp[1][0].mo_coeff.copy ())
        mc_grad = mc.nuc_grad_method ()
        e_ref = mc.e_tot
        de_ref = mc_grad.kernel ()
        self.assertTrue (mc.converged)
        self.assertTrue (mc_grad.converged)
        mo_coeff = mc.mo_coeff.copy ()
        mol = mol_nosym.copy ()
        mol.spin = 2
        mol.build ()
        mf = scf.RHF (mol).run ()
        mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 5, (2,0),
                            grids_level=1).run (mo_coeff=mo_coeff)
        self.assertTrue (mc.converged)
        self.assertAlmostEqual (mc.e_tot, e_ref, 6)
        mc_grad = mc.nuc_grad_method ()
        self.assertAlmostEqual (lib.fp (mc_grad.kernel ()), lib.fp (de_ref), 6)
        self.assertTrue (mc_grad.converged)

    def test_triplet_mol_df (self):
        '''Check that energies & gradients do not depend on if the parent MF is RHF or ROHF
        Density fitting can cause a weird interaction.'''
        mc = mcpdft.CASSCF (mf_nosym.density_fit ().run (), 'ftLDA,VWN3', 5, (2,0),
                            grids_level=1).run (mo_coeff=mcp[1][0].mo_coeff.copy ())
        mc_grad = mc.nuc_grad_method ()
        e_ref = mc.e_tot
        de_ref = mc_grad.kernel ()
        self.assertTrue (mc.converged)
        self.assertTrue (mc_grad.converged)
        mo_coeff = mc.mo_coeff.copy ()
        mol = mol_nosym.copy ()
        mol.spin = 2
        mol.build ()
        mf = scf.RHF (mol).density_fit ().run ()
        mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 5, (2,0),
                            grids_level=1).run (mo_coeff=mo_coeff)
        self.assertTrue (mc.converged)
        self.assertAlmostEqual (mc.e_tot, e_ref, 6)
        mc_grad = mc.nuc_grad_method ()
        self.assertAlmostEqual (lib.fp (mc_grad.kernel ()), lib.fp (de_ref), 6)
        self.assertTrue (mc_grad.converged)

if __name__ == "__main__":
    print("Full Tests for MC-PDFT gradients API")
    unittest.main()


