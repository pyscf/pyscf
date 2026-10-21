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
'''

import numpy
from scipy import linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.solvent._attach_solvent import _Solvation, _for_casscf


def _for_mcpdft(mc, solvent_obj, dm=None):
    '''Add solvent model to MC-PDFT method.'''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = MCPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (MCPDFTWithSolvent, mc.__class__), name)


class MCPDFTWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, MCPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    # Additional methods for MCPDFT with solvent can be added here
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)


    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''

        # redefining the mc_obj here
        from pyscf.mcscf.addons import StateAverageFCISolver

        if isinstance(self.fcisolver, StateAverageFCISolver):
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas).state_average_(self.weights)
        else:
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas)

        mc_obj.__dict__.update(self.__dict__)
        mc_obj = _for_casscf(mc_obj, self.with_solvent, dm=None)
        mc_obj.__class__ = type(mc_obj.__class__.__name__, (mc_obj.__class__,),
                         {'dump_flags': self.dump_flags(verbose=self.verbose)})

        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = mc_obj.kernel()

        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, dump_chk=True, verbose=None, **kwargs):
        '''Compute the MC-PDFT energy(ies) (and update stored data)
        with the MC-SCF wave function fixed. '''

        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if ci is not None: self.ci = ci
        if ot is not None: self.otfnal = ot
        if otxc is not None: self.otxc = otxc
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len(grids_attr): self.grids.__dict__.update(**grids_attr)
        if verbose is None: verbose = self.verbose
        self.verbose = self.otfnal.verbose = verbose
        nroots = getattr(self.fcisolver, 'nroots', 1)


        epdft = [self.energy_tot(mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                                 logger_tag=f'MC-PDFT state {ix}')
                 for ix in range(nroots)]
        self.e_ot = [e_ot for e_tot, e_ot in epdft]

        from pyscf.mcscf.addons import StateAverageMCSCFSolver
        if isinstance(self, StateAverageMCSCFSolver):
            e_states = [e_tot for e_tot, e_ot in epdft]
            try:
                self.e_states = e_states
            except AttributeError as e:
                self.fcisolver.e_states = e_states
                assert (self.e_states is e_states), str(e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be
            # because that makes no sense
            self.e_tot = numpy.dot(e_states, self.weights)
            e_states = self.e_states

            g_states = []

            for i, ei in enumerate(e_states):
                g_states.append(ei+self.with_solvent.e[i])
                logger.note(self,'  State %d   E(MC-PDFT+solvent) = %.15g',
                            i, ei+self.with_solvent.e[i])

            # quantity update
            try:
                self.e_states = g_states
            except AttributeError as e:
                self.fcisolver.e_states = g_states
                assert (self.e_states is g_states), str(e)

            self.e_tot = numpy.dot(self.e_states, self.weights)
            e_states = self.e_states

        elif nroots > 1:  # nroots>1 CASCI
            self.e_tot = [e_tot for e_tot, e_ot in epdft]
            e_states = self.e_tot

        else:  # nroots==1 not StateAverage class
            self.e_tot, self.e_ot = epdft[0]
            e_states = [self.e_tot]

            logger.note(self,'  State %d   E(MC-PDFT+solvent) = %.15g',
                        len(e_states)-1, self.e_tot+self.with_solvent.e)

            # quantity update
            self.e_tot = self.e_tot + self.with_solvent.e
            e_states = [self.e_tot]

        if dump_chk:
            e_tot = self.e_tot
            e_ot = self.e_ot
            self.dump_chk(locals())

        return self.e_tot, self.e_ot, e_states

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
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = LPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (LPDFTWithSolvent, mc.__class__), name)


class LPDFTWithSolvent(_Solvation):
    _keys = {'with_solvent'}
    _extra_keys = {'lpdft_max_iter', 'lpdft_conv_tol'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self.lpdft_max_iter = getattr(mc, 'lpdft_max_iter', 50)
        self.lpdft_conv_tol = getattr(mc, 'lpdft_conv_tol', 1e-7)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, LPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    # Additional methods for LPDFT with solvent are added here -->
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''


        # redefining the mc_obj here
        from pyscf.mcscf.addons import StateAverageFCISolver

        if isinstance(self.fcisolver, StateAverageFCISolver):
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas).state_average_(self.weights)
        else:
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas)

        mc_obj.__dict__.update(self.__dict__)
        mc_obj = _for_casscf(mc_obj, self.with_solvent, dm=None)
        mc_obj._keys = mc_obj._keys | self._extra_keys
        mc_obj.__class__ = type(mc_obj.__class__.__name__, (mc_obj.__class__,),
                         {'dump_flags': self.dump_flags(verbose=self.verbose)})

        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = mc_obj.kernel()

        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


    def make_lpdft_ham_(self, mo_coeff=None, ci=None, ot=None,
                        vmat=None):
        """Compute the L-PDFT Hamiltonian

        Args:
            mo_coeff : ndarray of shape (nao, nmo)
                A full set of molecular orbital coefficients. Taken from self if
                not provided.

            ci : list of ndarrays of length nroots
                CI vectors should be from a converged CASSCF/CASCI calculation

            ot : an instance of on-top functional class - see otfnal.py

            vmat : ndarray of shape (nao, nao)
                Solvent potential matrix


        Returns:
            lpdft_ham : ndarray of shape (nroots, nroots) or (nirreps, nroots, nroots)
                Linear approximation to the MC-PDFT energy expressed as a
                hamiltonian in the basis provided by the CI vectors. If
                StateAverageMix, then returns the block diagonal of the lpdft
                hamiltonian for each irrep.
        """
        from pyscf.fci import direct_spin1
        from pyscf.mcpdft.lpdft import _LPDFTMix
        from pyscf.mcpdft import _dms


        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        if ot is None:
            ot = self.otfnal


        ot.reset(mol=self.mol)

        spin = abs(self.nelecas[0] - self.nelecas[1])
        omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
        if abs(omega) > 1e-11:
            raise NotImplementedError("range-separated on-top functionals")
        if abs(hyb[0] - hyb[1]) > 1e-11:
            raise NotImplementedError(
                "hybrid functionals with different exchange, correlations components"
            )

        cas_hyb = hyb[0]


        ncas = self.ncas

        casdm1s_0, casdm2_0 = self.get_casdm12_0(ci=ci)


        self.veff1, self.veff2, E_ot = self.get_pdft_veff(
            mo=mo_coeff,
            ci=ci,
            casdm1s=casdm1s_0,
            casdm2=casdm2_0,
            drop_mcwfn=True,
            incl_energy=True,
            ot=ot
        )

        # This is all standard procedure for generating the hamiltonian in PySCF
        h1, h0 = self.get_h1lpdft(E_ot, casdm1s_0, casdm2_0, hyb=1.0 - cas_hyb, mo_coeff=mo_coeff,
                                  vmat=vmat)
        h2 = self.get_h2lpdft()
        h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, self.nelecas, 0.5)

        def construct_ham_slice(solver, slice, nelecas):
            ci_irrep = ci[slice]
            if hasattr(solver, "orbsym"):
                solver.orbsym = self.fcisolver.orbsym

            hc_all_irrep = [solver.contract_2e(h2eff, c, ncas, nelecas) for c in ci_irrep]
            lpdft_irrep = numpy.tensordot(ci_irrep, hc_all_irrep, axes=((1, 2), (1, 2)))
            diag_idx = numpy.diag_indices_from(lpdft_irrep)
            lpdft_irrep[diag_idx] += h0 + cas_hyb * self.e_mcscf[slice]


            return lpdft_irrep

        if not isinstance(self, _LPDFTMix):
            return construct_ham_slice(direct_spin1, slice(0, len(ci)), self.nelecas)

        # We have a StateAverageMix Solver
        self._irrep_slices = []
        start = 0
        for solver in self.fcisolver.fcisolvers:
            end = start + solver.nroots
            self._irrep_slices.append(slice(start, end))
            start = end

        return [
            construct_ham_slice(s, irrep, self.fcisolver._get_nelec(s, self.nelecas))
            for s, irrep in zip(self.fcisolver.fcisolvers, self._irrep_slices)
        ]

    def get_lpdft_hconst(
        self,
        E_ot,
        casdm1s_0,
        casdm2_0,
        hyb=1.0,
        ncas=None,
        ncore=None,
        veff1=None,
        veff2=None,
        mo_coeff=None,
    ):
        """Compute h_const for the L-PDFT Hamiltonian

        Args:
        self : instance of class _PDFT

        E_ot : float
            On-top energy

        casdm1s_0 : ndarray of shape (2, ncas, ncas)
            Spin-separated 1-RDM in the active space generated from expansion
            density.

        casdm2_0 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed 2-RDM in the active space generated from expansion
            density.

        Kwargs:
        hyb : float
            Hybridization constant (lambda term)

        ncas : float
            Number of active space MOs

        ncore: float
            Number of core MOs

        veff1 : ndarray of shape (nao, nao)
            1-body effective potential in the AO basis computed using the
            zeroth-order densities.

        veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.

        Returns:
            Constant term h_const for the expansion term.
        """

        from pyscf.mcpdft import _dms

        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore
        if veff1 is None:
            veff1 = self.veff1
        if veff2 is None:
            veff2 = self.veff2
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        nocc = ncore + ncas

        # Get the 1-RDM matrices
        casdm1_0 = casdm1s_0[0] + casdm1s_0[1]
        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0, mo_coeff=mo_coeff)
        dm1 = dm1s[0] + dm1s[1]

        # Coulomb interaction
        vj = self._scf.get_j(dm=dm1)
        e_veff1_j = numpy.tensordot(veff1 + hyb * 0.5 * vj, dm1)

        # Deal with 2-electron on-top potential energy
        e_veff2 = veff2.energy_core
        e_veff2 += numpy.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1_0)
        e_veff2 += 0.5 * numpy.tensordot(
            veff2.papa[ncore:nocc, :, ncore:nocc, :], casdm2_0, axes=4
        )
        # h_nuc + E_ot - 1/2 g_pqrs D_pq D_rs - V_pq D_pq - 1/2 v_pqrs d_pqrs
        energy_core = hyb * self.energy_nuc() + E_ot - e_veff1_j - e_veff2


        return energy_core

    def get_lpdft_hcore_only(self, casdm1s_0, hyb=1.0, mo_coeff=None,
                          ncore=None, ncas=None, vmat=None):
        """
        Returns the lpdft hcore AO integrals weighted by the
        hybridization factor. Excludes the MC-SCF (wfn) component.

        Kwargs:
        vmat : ndarray of shape (nao, nao)
                Solvent potential matrix

        """
        if vmat is None:
            vmat = self.with_solvent.v

        from pyscf.mcpdft import _dms

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0, mo_coeff=mo_coeff,
                                    ncore=ncore, ncas=ncas)
        dm1 = dm1s[0] + dm1s[1]
        v_j = self._scf.get_j(dm=dm1)
        h_eff = hyb * self._scf.get_hcore() + self.veff1 + hyb * v_j

        h_eff += vmat  # add the solvent potential
        return h_eff

    def get_lpdft_hcore(self, casdm1s_0=None, mo_coeff=None, ncore=None, ncas=None, vmat=None):
        """
        Returns the full lpdft hcore AO integrals. Includes the MC-SCF
        (wfn) component for hybrid functionals.

        Kwargs:
        vmat : ndarray of shape (nao, nao)
                Solvent potential matrix
        """

        if vmat is None:
            vmat = self.with_solvent.v


        if casdm1s_0 is None:
            casdm1s_0 = self.get_casdm12_0()[0]

        spin = abs(self.nelecas[0] - self.nelecas[1])
        cas_hyb = self.otfnal._numint.rsh_and_hybrid_coeff(self.otfnal.otxc, spin=spin)[
            2
        ]
        hyb = 1.0 - cas_hyb[0]

        return cas_hyb[0] * self._scf.get_hcore() + self.get_lpdft_hcore_only(
            casdm1s_0, hyb=hyb, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas, vmat=vmat
        )

    def transformed_h1e_for_cas(
        self, E_ot, casdm1s_0, casdm2_0, hyb=1.0, mo_coeff=None, ncas=None, ncore=None,
        vmat=None,
    ):
        """Compute the CAS one-particle L-PDFT Hamiltonian

        Args:
            mc : instance of a _PDFT object

            E_ot : float
                On-top energy

            casdm1s_0 : ndarray of shape (2,ncas,ncas)
                Spin-separated 1-RDM in the active space generated from expansion
                density

            casdm2_0 : ndarray of shape (ncas,ncas,ncas,ncas)
                Spin-summed 2-RDM in the active space generated from expansion
                density

            hyb : float
                Hybridization constant (lambda term)

            mo_coeff : ndarray of shape (nao,nmo)
                A full set of molecular orbital coefficients. Taken from self if
                not provided.

            ncas : int
                Number of active space molecular orbitals

            ncore : int
                Number of core molecular orbitals

            vmat : ndarray of shape (nao, nao)
                The solvent potential matrix.



        Returns:
            A tuple, the first is the effective one-electron linear PDFT
            Hamiltonian defined in CAS space, the second is the modified core
            energy.
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore

        nocc = ncore + ncas
        mo_core = mo_coeff[:, :ncore]
        mo_cas = mo_coeff[:, ncore:nocc]

        # h_pq + V_pq + J_pq all in AO integrals
        hcore_eff = self.get_lpdft_hcore_only(casdm1s_0, hyb=hyb, mo_coeff=mo_coeff,
                                        ncore=ncore, ncas=ncas,
                                        vmat=vmat)
        energy_core = self.get_lpdft_hconst(E_ot, casdm1s_0, casdm2_0, hyb,
                                      mo_coeff=mo_coeff, ncore=ncore,
                                      ncas=ncas)

        if mo_core.size != 0:
            core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
            # This is precomputed in MRH's ERIS object
            energy_core += self.veff2.energy_core
            energy_core += numpy.tensordot(core_dm, hcore_eff).real

        h1eff = mo_cas.conj().T @ hcore_eff @ mo_cas
        # Add in the 2-electron portion that acts as a 1-electron operator
        h1eff += self.veff2.vhf_c[ncore:nocc, ncore:nocc]

        return h1eff, energy_core

    def get_h1lpdft(self, E_ot, casdm1s_0, casdm2_0, hyb=1.0, mo_coeff=None,
                    vmat=None):
        return self.transformed_h1e_for_cas(
            E_ot, casdm1s_0, casdm2_0, hyb=hyb, mo_coeff=mo_coeff,
            vmat=vmat)

    def get_solv_dms(self):
        from pyscf.mcscf.addons import StateAverageFCISolver
        _sa_casdms = StateAverageFCISolver.states_make_rdm1(self.fcisolver, self.ci, self.ncas, self.nelecas)

        dm = []
        for _sa_casdm in _sa_casdms:
            mocore = self.mo_coeff[:,:self.ncore]
            mocas = self.mo_coeff[:,self.ncore:self.ncore+self.ncas]
            _sa_1rdm = numpy.dot(mocore, mocore.conj().T) * 2
            _sa_1rdm = _sa_1rdm + reduce(numpy.dot, (mocas, _sa_casdm, mocas.conj().T))
            dm.append(_sa_1rdm)

        return dm

    def kernel(self, mo_coeff=None, ci0=None, ot=None, verbose=None, dump_chk=True):

        from pyscf.mcpdft import _dms

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
                ot=ot, ci=ci_mcscf, vmat=vmat)

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

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_lpdft(obj, self.with_solvent)
        return lib.to_gpu(self, obj)
