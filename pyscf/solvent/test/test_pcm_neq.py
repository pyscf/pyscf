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
        self.assertAlmostEqual(mc.e_states[0], -112.416375735352, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.265284640312, 6)

    def test_sa_mcpdft_neq(self):
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4).state_average_(WEIGHTS)
        mc.fix_spin_(ss=0)
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertAlmostEqual(mc.e_states[0], -112.811383851113, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.656837117804, 6)

    def test_lpdft_neq(self):
        mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
        mc.fix_spin_(ss=0)
        mc = mc.multi_state(WEIGHTS, 'lin')
        mc = solvent.PCM(mc, _neq_solvent())
        mc.kernel()
        self.assertAlmostEqual(mc.e_states[0], -112.813009540099, 6)
        self.assertAlmostEqual(mc.e_states[1], -112.656196490203, 6)

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

        cm = _neq_solvent()
        cm.eps_optical = cm.eps
        neq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        neq.fix_spin_(ss=0)
        neq = solvent.PCM(neq, cm)
        neq.kernel()

        for i, (e_eq, e_neq) in enumerate(zip(eq.e_states, neq.e_states)):
            self.assertAlmostEqual(e_eq, e_neq, 7, msg='state %d' % i)

    def test_neq_differs_from_eq(self):
        eq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        eq.fix_spin_(ss=0)
        eq = solvent.PCM(eq, _neq_solvent(equilibrium_solvation=True))
        eq.kernel()

        neq = mcscf.CASSCF(mf, 4, 4).state_average_(WEIGHTS)
        neq.fix_spin_(ss=0)
        neq = solvent.PCM(neq, _neq_solvent())
        neq.kernel()

        self.assertTrue(abs(eq.e_states[1] - neq.e_states[1]) > 1e-6)


class Dispatch(unittest.TestCase):
    '''solvation_type() must keep every existing configuration on the
    unmodified equilibrium path.'''

    def test_default(self):
        cm = pcm.PCM(mol)
        self.assertEqual(cm.solvation_type(), 'default')

    def test_state_id_does_not_switch_mode(self):
        # state_id selects which root polarizes the cavity; it must not
        # select a solvation scheme.
        cm = pcm.PCM(mol)
        for state_id in (0, 1, 2):
            cm.state_id = state_id
            self.assertEqual(cm.solvation_type(), 'default')

    def test_refdm_enables_ss_neq(self):
        cm = pcm.PCM(mol)
        cm.refdm = mf.make_rdm1()
        self.assertEqual(cm.solvation_type(), 'ss_neq')
        cm.equilibrium_solvation = True
        self.assertEqual(cm.solvation_type(), 'default')

    def test_rfroot_enables_sa(self):
        cm = pcm.PCM(mol)
        cm.rfroot = 1
        self.assertEqual(cm.solvation_type(), 'sa_neq')
        cm.equilibrium_solvation = True
        self.assertEqual(cm.solvation_type(), 'sa_eq')


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


if __name__ == "__main__":
    print("Tests for nonequilibrium PCM solvation")
    unittest.main()
