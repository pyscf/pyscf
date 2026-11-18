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

'''
Attach solvent models to post-MCSCF methods.

References:
[1] https://doi.org/10.1021/jp000997s
    Nonequilibrium CASSCF in PCM.

[2] https://doi.org/10.1039/c1sc00313e
    Slow/fast charge partitioning.

[3] https://doi.org/10.1021/ct5010388
[4] https://doi.org/10.1021/acs.jctc.4c01473
    Self-consistency protocols for the reaction-field operator in a
    state-averaged reference.

[5] A. Choudhury, M. R. Hermes, D. G. Truhlar, L. Gagliardi. Vertical
    excitation energies in solution via a two-time-scale solvation model
    coupled to pair-density functional theory. J. Chem. Theory Comput. 2026.
    https://doi.org/10.1021/acs.jctc.6c01690
    The scheme implemented here, including the protocol used for the
    reference SA-CASSCF (Section 2.2 and Table 3).
'''

import numpy
from scipy import linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.solvent._attach_solvent import (_Solvation, _for_casscf,
                                           _states_dm1_ao, CASSCFWithSolvent)

'''
Refer to the following paper for the definition of MC-PDFT+PCM and L-PDFT+PCM algorithms:
A. Choudhury, M. R. Hermes, D. G. Truhlar, L. Gagliardi; Vertical
Excitation Energies in Solution via a Two-Time-Scale Solvation Model Coupled to
Pair-Density Functional Theory. J. Chem. Theory Comput. 2026;
https://doi.org/10.1021/acs.jctc.6c01690
'''
def _for_mcpdft(mc, solvent_obj, dm=None):
    '''Add solvent model to MC-PDFT method.'''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    from pyscf import mcscf
    if not isinstance(mc, mcscf.mc1step.CASSCF):
        raise NotImplementedError(
            'Solvent for MC-PDFT with a CASCI reference is not implemented.')

    from pyscf.mcscf.addons import StateAverageFCISolver
    if dm is not None and isinstance(mc.fcisolver, StateAverageFCISolver):
        raise NotImplementedError(
            'A frozen solvent potential is not supported for '
            'state-averaged MC-PDFT')

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = MCPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (MCPDFTWithSolvent, mc.__class__), name)


class MCPDFTWithSolvent(CASSCFWithSolvent):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self._e_tot_without_solvent = 0
        self._e_states_corrected = 0

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, MCPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        del obj._e_tot_without_solvent
        del obj._e_states_corrected
        return obj

    # Additional methods for MCPDFT with solvent can be added here
    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT
        calculation, with the solvent responding to the density.'''
        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            (self.e_mcscf, self.e_cas, self.ci,
             self.mo_coeff, self.mo_energy) = self._mc_class.kernel(
                 self, mo_coeff, ci0, **kwargs)
        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, dump_chk=True,
                             verbose=None, **kwargs):
        '''Compute the MC-PDFT energy(ies) with the MC-SCF wave function
        and add the solvent contribution to each state.
        '''
        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        log = logger.new_logger(self, verbose)

        # dump_chk is deferred until the solvent terms are in
        e_tot, e_ot, e_states = super().compute_pdft_energy_(
            mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
            grids_level=grids_level, grids_attr=grids_attr,
            dump_chk=False, verbose=verbose, **kwargs)

        e_solv = self.with_solvent.e
        if isinstance(self, StateAverageMCSCFSolver):
            e_states = [ei + e_solv[i] for i, ei in enumerate(e_states)]
            try:
                self.e_states = e_states
            except AttributeError as e:
                self.fcisolver.e_states = e_states
                assert (self.e_states is e_states), str(e)
            e_states = self.e_states
            self.e_tot = numpy.dot(e_states, self.weights)
            for i, ei in enumerate(e_states):
                log.note('  State %d  E(MC-PDFT+solvent) = %.15g', i, ei)
        elif isinstance(self.e_tot, list):
            raise NotImplementedError(
                'Solvent for MC-PDFT with nroots > 1 and no state averaging')
        else:
            self.e_tot = e_tot + e_solv
            e_states = [self.e_tot]
            log.note('E(MC-PDFT+solvent) = %.15g', self.e_tot)

        if dump_chk:
            e_tot = self.e_tot
            e_ot = self.e_ot
            self.dump_chk(locals())

        return self.e_tot, self.e_ot, e_states

    def nuc_grad_method(self):
        raise NotImplementedError('Gradients for MC-PDFT with solvent not available yet.')

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_mcpdft(obj, self.with_solvent)
        return lib.to_gpu(self, obj)



def _for_lpdft(mc, solvent_obj, dm=None):
    '''Add solvent model to L-PDFT method.'''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        raise NotImplementedError(
            'A frozen solvent potential is not supported for L-PDFT')

    sol_cas = LPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (LPDFTWithSolvent, mc.__class__), name)


class LPDFTWithSolvent(CASSCFWithSolvent):
    _keys = {'with_solvent', 'lpdft_max_iter', 'lpdft_conv_tol'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self._e_tot_without_solvent = 0
        self._e_states_uncorrected = 0
        self._e_states_corrected = 0
        self.lpdft_max_iter = getattr(mc, 'lpdft_max_iter', 50)
        self.lpdft_conv_tol = getattr(mc, 'lpdft_conv_tol', 1e-7)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, LPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        del obj._e_tot_without_solvent
        del obj._e_states_uncorrected
        del obj._e_states_corrected
        return obj

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the SA-CASSCF reference with the solvent responding to
        the state densities.'''
        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            (self.e_mcscf, self.e_cas, self.ci,
             self.mo_coeff, self.mo_energy) = self._mc_class.kernel(
                 self, mo_coeff, ci0, **kwargs)
        # The L-PDFT replaces <v,dm> with E_solv per state, so the Hamiltonian
        # must carry V exactly once: H_solv = lambda (H^el + V) +
        # (1 - lambda)(H^L-PDFT + V). hcore supplies (1 - lambda) V, so the
        # MCSCF energy should come as the expectation value with V in hcore
        # rather than the corrected free energy. Only matters for hybrid on-top functionals.
        self.e_mcscf = numpy.asarray(self._e_states_uncorrected)
        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def get_solv_dms(self):
        '''AO-basis 1-RDM of each L-PDFT state.'''
        return _states_dm1_ao(self)


    def kernel(self, mo_coeff=None, ci0=None, ot=None, verbose=None, dump_chk=True):

        if ot is None:
            ot = self.otfnal
        ot.reset(mol=self.mol)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        log = logger.new_logger(self, verbose)
        if ci0 is None and isinstance(getattr(self, "ci", None), list):
            ci0 = [c.copy() for c in self.ci]

        # SA-CASSCF+PCM optimization
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
        ci_mcscf = [c.copy() for c in self.ci]

        with_solvent = self.with_solvent
        dms = self.get_solv_dms()
        _, vmat = with_solvent.kernel(dms)
        with_solvent.v = vmat

        lpdft_conv = False
        _iter = 0
        e_tot_prev = 0.0

        # recompute solvent potential from L-PDFT states densities until convergence
        log.info("\nStarting L-PDFT+solvent iterations...")
        while not lpdft_conv and _iter < self.lpdft_max_iter:

            self.lpdft_ham = self.make_lpdft_ham_(
                ot=ot, ci=ci_mcscf)

            if hasattr(self, "_irrep_slices"):
                e_states, si_pdft = zip(*map(self._eig_si, self.lpdft_ham))
                e_states = numpy.concatenate(e_states)
                si_pdft  = linalg.block_diag(*si_pdft)
            else:
                e_states, si_pdft = self._eig_si(self.lpdft_ham)

            self.si_pdft  = si_pdft
            self.ci = list(numpy.tensordot(si_pdft.T, numpy.asarray(ci_mcscf), axes=1))

            # energy corrections
            dm = self.get_solv_dms()
            edups = []
            for i in range(len(e_states)):
                edup = numpy.einsum('ij,ji->', vmat, dm[i])
                edups.append(edup)

            e_solv, v_solv = with_solvent.kernel(dm)
            for i in range(len(e_states)):
                e_states[i] = e_states[i] - edups[i] + e_solv[i]

            self.e_states = e_states
            self.e_tot    = numpy.dot(e_states, self.weights)

            lpdft_dE =abs(self.e_tot - e_tot_prev)
            _iter += 1
            log.info(f"** Solvent self-consistent cycle {_iter}:\n dE = {lpdft_dE}")
            self._finalize_lin()

            if (_iter >= self.lpdft_max_iter) and (lpdft_dE > self.lpdft_conv_tol):
                log.info("self-consistent L-PDFT+solvent not converged within the maximum number of iterations \n"
                "Increase mc.lpdft_max_iter.")

            if (lpdft_dE <= self.lpdft_conv_tol):
                lpdft_conv = True
                log.info("self-consistent L-PDFT+solvent converged")

            e_tot_prev = self.e_tot
            vmat = v_solv
            with_solvent.v = v_solv

        return (self.e_tot, self.e_mcscf, self.e_cas,
                self.ci, self.mo_coeff, self.mo_energy)


    def _finalize_lin(self):
        log = logger.Logger(self.stdout, self.verbose)
        nroots = len(self.e_states)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, "spin_square", None):
            ss = self.fcisolver.states_spin_square(self.ci, self.ncas, self.nelecas)[0]


            for i in range(nroots):
                log.note(
                    "  State %d weight %g  E(LPDFT+solvent) = %.15g  S^2 = %.7f",
                    i,
                    self.weights[i],
                    self.e_states[i],
                    ss[i],
                )

        else:
            for i in range(nroots):
                log.note(
                    "  State %d weight %g  E(LPDFT+solvent) = %.15g",
                    i,
                    self.weights[i],
                    self.e_states[i],
                )

    def nuc_grad_method(self):
        raise NotImplementedError('Gradients for L-PDFT with solvent not available yet.')

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_lpdft(obj, self.with_solvent)
        return lib.to_gpu(self, obj)
