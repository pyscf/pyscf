#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Arpan Choudhury (University of Chicago) <arpanchoudhury29@gmail.com>
#

'''Nonequilibrium PCM solvation for SA-CASSCF, SA-MC-PDFT and L-PDFT.

The slow (orientational) charges are taken from state 0 and the fast
(electronic) charges from state .rfroot, the latter evaluated with the optical
dielectric constant.

The reference values below are self-generated. They cannot be taken from
another program: the solvent energy depends on the cavity, and the cavity
construction (radii, tessellation, switching function) differs between codes,
so the same scheme gives a different number elsewhere.
'''

import unittest
import numpy
from pyscf import gto, scf, mcscf, mcpdft, solvent
from pyscf.solvent import pcm, _state_solvation

WEIGHTS = [0.5, 0.5]


def setUpModule():
    global mol, mf
    mol = gto.M(
        atom='''
H              1.110158    0.935182   -0.000022
C              0.537003    0.000000    0.000011
H              1.110154   -0.935184   -0.000022
O             -0.680291    0.000000   -0.000003''',
        basis='sto-3g',
        output='/dev/null',
        verbose=0)
    mf = solvent.PCM(scf.RHF(mol), 'water')
    mf.kernel()


def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


def _neq_solvent(equilibrium_solvation=False, eps_optical=None):
    '''A solvent object of its own, so the SCF's is not modified.'''
    cm = pcm.PCM(mol, 'water')
    cm.verbose = 0
    if eps_optical is not None:
        cm.eps_optical = eps_optical
    cm.rfroot = 1
    cm.equilibrium_solvation = equilibrium_solvation
    return cm


class KnownValues(unittest.TestCase):
    def test_sa_casscf_neq(self):
        mc = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        mc.fix_spin_(ss=0)
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_states[0], -112.416375735352, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.265284640312, 6)

    def test_sa_mcpdft_neq(self):
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4).state_average_(WEIGHTS)
        mc.fix_spin_(ss=0)
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_states[0], -112.811383851113, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.656837117804, 6)

    def test_ss_mcpdft_neq(self):
        # state-specific: the slow charges come from a converged ground-state
        # density, and the fast charges from the excited state.
        eq = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
        eq.conv_tol = 1e-10
        eq = solvent.PCM(eq, pcm.PCM(mol, 'water'))
        eq.kernel()
        self.assertTrue(eq.converged)

        cm = pcm.PCM(mol, 'water')
        cm.verbose = 0
        cm.equilibrium_solvation = False
        cm.refdm = eq.make_rdm1()
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4).state_specific_(2)
        mc.fix_spin_(ss=0)
        mc.conv_tol = 1e-10
        mc = solvent.PCM(mc, cm)
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_tot, -112.678529351332, 6)

    def test_lpdft_neq(self):
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.multi_state(WEIGHTS, 'lin')
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_states[0], -112.813009540099, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.656196490203, 6)

    def test_lpdft_neq_hybrid(self):
        # tPBE0 has cas_hyb = 0.25. Both the (1 - lambda) v in hcore and the
        # lambda * E_mcscf diagonal contribute the reaction field, and the two
        # must sum to exactly one V. tPBE (lambda = 0) blind to either
        # does not catch this.
        mc = mcpdft.CASSCF(mf, 'tPBE0', 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.multi_state(WEIGHTS, 'lin')
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_states[0], -112.713851125972, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.558468572360, 6)


class TestIdentities(unittest.TestCase):
    def test_eps_optical_equal_eps_reduces_to_equilibrium(self):
        # With eps_optical == eps the fast charges are the total charges, so
        # the slow charges vanish and every state falls back to its own
        # equilibrium energy. The reaction field matches as well, so both runs
        # converge to the same wavefunction.
        eq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        eq.fix_spin_(ss=0)
        eq = solvent.PCM(eq, _neq_solvent(equilibrium_solvation=True))
        eq.kernel()
        self.assertTrue(eq.converged)

        cm = _neq_solvent()
        cm.eps_optical = cm.eps
        neq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        neq.fix_spin_(ss=0)
        neq = solvent.PCM(neq, cm)
        neq.kernel()
        self.assertTrue(neq.converged)
        for i, (e_eq, e_neq) in enumerate(zip(eq.e_states, neq.e_states)):
            self.assertAlmostEqual(e_eq, e_neq, 7, msg='state %d' % i)

    def test_lambda_one_reduces_to_sa_casscf(self):
        # With cas_hyb = 1 the L-PDFT+solvent must return the SA-CASSCF+solvent energies.
        # This pins which E_mcscf the lambda*E_mcscf diagonal needs: the
        # expectation value with v in hcore, not the solvated free energy.
        otxc = 't0.0*PBE + 1.0*HF, 0.0*PBE + 1.0*HF'

        ref = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        ref.fix_spin_(ss=0)
        ref.conv_tol = 1e-10
        ref = solvent.PCM(ref, _neq_solvent())
        ref.kernel()
        self.assertTrue(ref.converged)
        mc = mcpdft.CASSCF(mf, otxc, 4, 4)
        mc.fix_spin_(ss=0)
        mc.conv_tol = 1e-10
        mc = mc.multi_state(WEIGHTS, 'lin')
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertTrue(mc.converged)

        for i, (a, b) in enumerate(zip(ref.e_states, mc.e_states)):
            self.assertAlmostEqual(a, b, 7, msg='state %d' % i)

    def test_neq_differs_from_eq(self):
        eq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        eq.fix_spin_(ss=0)
        eq = solvent.PCM(eq, _neq_solvent(equilibrium_solvation=True))
        eq.kernel()
        self.assertTrue(eq.converged)

        neq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        neq.fix_spin_(ss=0)
        neq = solvent.PCM(neq, _neq_solvent())
        neq.kernel()
        self.assertTrue(neq.converged)

        self.assertTrue(abs(eq.e_states[1] - neq.e_states[1]) > 1e-6)


class DefaultPathUnchanged(unittest.TestCase):
    '''With neither rfroot nor refdm set, the solvent wrappers must run the
    standard equilibrium path, including for a state-averaged solver. These
    energies are from PySCF 2.15.0, before this feature was added.
    '''

    def _sa_casscf(self):
        mc = mcscf.CASSCF(scf.RHF(mol).run(), 4, 4).state_average_(WEIGHTS)
        mc.fix_spin_(ss=0)
        return mc

    def test_sa_casscf_pcm(self):
        mc = solvent.PCM(self._sa_casscf())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertEqual(mc.with_solvent.solvation_type(), 'default')
        self.assertAlmostEqual(mc.e_states[0], -112.422444755989, 7)
        self.assertAlmostEqual(mc.e_states[1], -112.271909087233, 7)

    def test_sa_casscf_ddcosmo(self):
        mc = solvent.ddCOSMO(self._sa_casscf())
        mc.kernel()
        self.assertTrue(mc.converged)
        self.assertEqual(mc.with_solvent.solvation_type(), 'default')
        self.assertAlmostEqual(mc.e_states[0], -112.430592598076, 7)
        self.assertAlmostEqual(mc.e_states[1], -112.279992222338, 7)

    def test_state_id_does_not_switch_mode(self):
        # state_id selects which root polarizes the cavity; it must not
        # select a solvation scheme.
        cm = pcm.PCM(mol, 'water')
        for state_id in (0, 1, 2):
            cm.state_id = state_id
            self.assertEqual(cm.solvation_type(), 'default')


class Dispatch(unittest.TestCase):
    '''refdm and rfroot are what select a nonequilibrium scheme, and the
    combinations that would silently ignore one of them raise.'''

    def _cm(self, rfroot=None, refdm=None, equilibrium_solvation=False):
        cm = pcm.PCM(mol)
        cm.rfroot = rfroot
        cm.refdm = refdm
        cm.equilibrium_solvation = equilibrium_solvation
        return cm

    def test_neither_set(self):
        self.assertEqual(self._cm().solvation_type(), 'default')
        self.assertEqual(self._cm(equilibrium_solvation=True).solvation_type(),
                         'default')

    def test_rfroot_only(self):
        self.assertEqual(self._cm(rfroot=1).solvation_type(), 'sa_neq')
        self.assertEqual(
            self._cm(rfroot=1, equilibrium_solvation=True).solvation_type(),
            'sa_eq')

    def test_refdm_only(self):
        dm = mf.make_rdm1()
        self.assertEqual(self._cm(refdm=dm).solvation_type(), 'ss_neq')
        self.assertRaises(
            ValueError,
            self._cm(refdm=dm, equilibrium_solvation=True).solvation_type)

    def test_both_set(self):
        dm = mf.make_rdm1()
        for eq in (False, True):
            self.assertRaises(
                NotImplementedError,
                self._cm(rfroot=1, refdm=dm,
                         equilibrium_solvation=eq).solvation_type)

class Guards(unittest.TestCase):
    '''Invalid input must raise rather than return a wrong number.'''

    def _sa_solvent(self, rfroot):
        cm = pcm.PCM(mol)
        cm.rfroot = rfroot
        return cm

    def test_rfroot_out_of_range(self):
        dms = numpy.asarray([mf.make_rdm1()] * 2)
        self.assertRaises(ValueError, _state_solvation.kernel_sa_neq,
                          self._sa_solvent(2), dms)

    def test_negative_rfroot(self):
        dms = numpy.asarray([mf.make_rdm1()] * 2)
        self.assertRaises(ValueError, _state_solvation.kernel_sa_neq,
                          self._sa_solvent(-1), dms)

    def test_single_dm_into_sa_kernel(self):
        # happens when the solvent object is shared with an SCF object
        self.assertRaises(ValueError, _state_solvation.kernel_sa_neq,
                          self._sa_solvent(0), mf.make_rdm1())

    def test_frozen_solvent_with_state_average(self):
        mc = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        self.assertRaises(NotImplementedError, solvent.PCM, mc,
                          pcm.PCM(mol), mf.make_rdm1())

    def test_cms_pdft_refused(self):
        # The MS-PDFT effective Hamiltonian is built from intermediate states
        # that the reaction-field terms do not account for.
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.multi_state(WEIGHTS, 'cms')
        self.assertRaises(NotImplementedError, solvent.PCM, mc)

    def test_neq_gradient_refused(self):
        mc = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        mc.fix_spin_(ss=0)
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertRaises(NotImplementedError, mc.nuc_grad_method)

    def test_mcpdft_gradient_refused(self):
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
        mc = solvent.PCM(mc, pcm.PCM(mol, 'water'))
        self.assertRaises(NotImplementedError, mc.nuc_grad_method)

    def test_wrapper_does_not_mutate_scf_solvent(self):
        mc = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        mc = solvent.PCM(mc)
        mc.with_solvent.rfroot = 1
        self.assertIsNone(mf.with_solvent.rfroot)

if __name__ == "__main__":
    print("Tests for nonequilibrium PCM solvation")
    unittest.main()
