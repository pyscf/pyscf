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
import numpy as np
from scipy import linalg
from pyscf import gto, scf, mcscf 
from pyscf import mcpdft
import unittest, math


def get_lih (r):
    mol = gto.M (atom='Li 0 0 0\nH {} 0 0'.format (r), basis='sto3g',
                 output='/dev/null', verbose=0)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
    mc.fix_spin_(ss=0)
    mc = mc.multi_state ([0.5,0.5], 'cms').run (conv_tol=1e-8)
    return mol, mf, mc

def setUpModule():
    global mol, mf, mc
    mol, mf, mc = get_lih (1.5)

def tearDownModule():
    global mol, mf, mc
    mol.stdout.close ()
    del mol, mf, mc

class KnownValues(unittest.TestCase):

    def test_reference_adiabats (self):
        # Recover SA-CASSCF properly
        mc_ref = mcscf.CASCI (mf, 2, 2)
        mc_ref.fix_spin_(ss=0)
        mc_ref.fcisolver.nroots = 2
        mc_ref.kernel (mo_coeff=mc.mo_coeff)
        ci_test = mc.get_ci_adiabats (uci='MCSCF')
        for state in 0,1:
            with self.subTest (state=state):
                e1 = mc.e_mcscf[state]
                e2 = mc_ref.e_tot[state]
                self.assertAlmostEqual (e1, e2, 8)
                ovlp = np.dot (mc_ref.ci[state].ravel (),
                               ci_test[state].ravel ())
                self.assertAlmostEqual (abs(ovlp), 1.0, 8)

    def test_intermediate_diabats (self):
        # Diabat optimization and energy expression
        mc_ref = mcpdft.CASCI (mf, 'ftLDA,VWN3', 2, 2, grids_level=1)
        mc_ref.fix_spin_(ss=0)
        mc_ref.fcisolver.nroots = 2
        mc_ref.kernel (mo_coeff=mc.mo_coeff)
        mc_ref.compute_pdft_energy_(ci=mc.ci)
        for state in 0,1:
            with self.subTest (state=state):
                e1 = mc.hdiag_pdft[state]
                e2 = mc_ref.e_tot[state]
                self.assertAlmostEqual (e1, e2, 8)
        Qaa_max = mc.diabatizer ()[0]
        rand_gen = np.zeros ((2,2))
        rand_gen[1,0] = math.pi * ((2*np.random.rand())-1) 
        rand_gen -= rand_gen.T
        rand_u = linalg.expm (rand_gen)
        rand_ci = mc.get_ci_basis (uci=rand_u)
        Qaa_rand = mc.diabatizer (ci=rand_ci)[0]
        self.assertLessEqual (Qaa_rand, Qaa_max)

    def test_final_adiabats (self):
        # Indirect energy calculation
        dh = mc.get_heff_pdft () - mc.heff_mcscf
        dh_offdiag = dh - np.diag (dh.diagonal ())
        self.assertAlmostEqual (np.amax (np.abs (dh_offdiag)), 0, 9)
        dh = np.dot (dh, mc.si_pdft)
        dh = np.dot (mc.si_pdft.conj ().T, dh)
        mc_ref = mcscf.CASCI (mf, 2, 2)
        mc_ref.mo_coeff = mc.mo_coeff
        cisolver = mc_ref.fcisolver
        h1, h0 = mc_ref.get_h1eff ()
        h2 = mc_ref.get_h2eff ()
        h2eff = cisolver.absorb_h1e (h1, h2, 2, (1,1), 0.5)
        ci = mc.get_ci_adiabats (uci='MSPDFT')
        hci = [cisolver.contract_2e (h2eff, c, 2, (1,1)) for c in ci]
        chc = np.tensordot (ci, hci, axes=((1,2),(1,2))) + dh
        e_ref = chc.diagonal () 
        chc_offdiag = chc - np.diag (e_ref)
        self.assertAlmostEqual (np.amax (np.abs (chc_offdiag)), 0, 9)
        for state in 0,1:
            with self.subTest (state=state):
                e1 = mc.e_states[state]
                e2 = e_ref[state] + h0
                self.assertAlmostEqual (e1, e2, 8)

    def test_diabatize (self):
        f_ref = mc.diabatizer ()[0]
        theta_rand = 360 * np.random.rand () - 180
        ct = math.cos (theta_rand)
        st = math.sin (theta_rand)
        u_rand = np.array ([[ct,st],[-st,ct]])
        ci_rand = mc.get_ci_basis (uci=u_rand)
        f_test = mc.diabatizer (ci=ci_rand)[0]
        self.assertLessEqual (f_test, f_ref)
        conv, ci_test = mc.diabatize (ci=ci_rand)
        f_test = mc.diabatizer (ci=ci_test)[0]
        self.assertTrue (conv)
        self.assertAlmostEqual (f_test, f_ref, 9)

if __name__ == "__main__":
    print("Full Tests for MS-PDFT energy API")
    unittest.main()






