# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory for
unrestricted references
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.scf import _vhf
from pyscf.agf2 import ragf2, _agf2, mpi_helper
from pyscf.agf2 import aux_space as aux
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.mp.ump2 import get_frozen_mask as _get_frozen_mask

BLKMIN = getattr(__config__, 'agf2_blkmin', 1)


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped,
        for a single spin.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : tuple of GreensFunction
            Occupied Green's function for each spin
        gf_vir : tuple of GreensFunction
            Virtual Green's function for each spin

    Kwargs:
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Returns:
        :class:`SelfEnergy`
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_occ[1]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction
    assert type(gf_vir[1]) is aux.GreensFunction

    nmo = eri.nmo
    noa, nob = gf_occ[0].naux, gf_occ[1].naux
    nva, nvb = gf_vir[0].naux, gf_vir[1].naux
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)

    ci_a, ei_a = gf_occ[0].coupling, gf_occ[0].energy
    ci_b, ei_b = gf_occ[1].coupling, gf_occ[1].energy
    ca_a, ea_a = gf_vir[0].coupling, gf_vir[0].energy
    ca_b, ea_b = gf_vir[1].coupling, gf_vir[1].energy

    mem_incore = (nmo[0]*noa*(noa*nva+nob*nvb)) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mem_incore+mem_now < agf2.max_memory) or agf2.incore_complete:
        qeri = _make_qmo_eris_incore(agf2, eri, (ci_a, ci_a, ca_a), (ci_b, ci_b, ca_b), spin=0)
    else:
        qeri = _make_qmo_eris_outcore(agf2, eri, (ci_a, ci_a, ca_a), (ci_b, ci_b, ca_b), spin=0)

    if isinstance(qeri[0], np.ndarray):
        vv, vev = _agf2.build_mats_uagf2_incore(qeri, (ei_a, ei_b), (ea_a, ea_b), **facs)
    else:
        vv, vev = _agf2.build_mats_uagf2_outcore(qeri, (ei_a, ei_b), (ea_a, ea_b), **facs)

    e, c = _agf2.cholesky_build(vv, vev)
    se_a = aux.SelfEnergy(e, c, chempot=gf_occ[0].chempot)
    se_a.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = get_frozen_mask(agf2)
        coupling = np.zeros((nmo[0], se_a.naux))
        coupling[mask[0]] = se_a.coupling
        se_a = aux.SelfEnergy(se_a.energy, coupling, chempot=se_a.chempot)

    cput0 = log.timer('se part (alpha)', *cput0)

    mem_incore = (nmo[1]*nob*(nob*nvb+noa*nva)) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mem_incore+mem_now < agf2.max_memory) or agf2.incore_complete:
        qeri = _make_qmo_eris_incore(agf2, eri, (ci_a, ci_a, ca_a), (ci_b, ci_b, ca_b), spin=1)
    else:
        qeri = _make_qmo_eris_outcore(agf2, eri, (ci_a, ci_a, ca_a), (ci_b, ci_b, ca_b), spin=1)

    if isinstance(qeri[0], np.ndarray):
        vv, vev = _agf2.build_mats_uagf2_incore(qeri, (ei_b, ei_a), (ea_b, ea_a), **facs)
    else:
        vv, vev = _agf2.build_mats_uagf2_outcore(qeri, (ei_b, ei_a), (ea_b, ea_a), **facs)

    e, c = _agf2.cholesky_build(vv, vev)
    se_b = aux.SelfEnergy(e, c, chempot=gf_occ[1].chempot)
    se_b.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = get_frozen_mask(agf2)
        coupling = np.zeros((nmo[1], se_b.naux))
        coupling[mask[1]] = se_b.coupling
        se_b = aux.SelfEnergy(se_b.energy, coupling, chempot=se_b.chempot)

    cput0 = log.timer('se part (beta)', *cput0)

    return (se_a, se_b)


def get_fock(agf2, eri, gf=None, rdm1=None):
    ''' Computes the physical space Fock matrix in MO basis. If :attr:`rdm1`
        is not supplied, it is built from :attr:`gf`, which defaults to
        the mean-field Green's function.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals

    Kwargs:
        gf : GreensFunction
            Auxiliaries of the Green's function
        rdm1 : 2D array
            Reduced density matrix

    Returns:
        ndarray of physical space Fock matrix
    '''

    if rdm1 is None:
        rdm1 = agf2.make_rdm1(gf)

    vj_aa, vk_aa = agf2.get_jk(eri.eri_aa, rdm1=rdm1[0])
    vj_bb, vk_bb = agf2.get_jk(eri.eri_bb, rdm1=rdm1[1])
    vj_ab = agf2.get_jk(eri.eri_ab, rdm1=rdm1[1], with_k=False)[0]
    vj_ba = agf2.get_jk(eri.eri_ba, rdm1=rdm1[0], with_k=False)[0]

    fock_a = eri.h1e[0] + vj_aa + vj_ab - vk_aa
    fock_b = eri.h1e[1] + vj_bb + vj_ba - vk_bb

    fock = (fock_a, fock_b)

    return fock


def fock_loop(agf2, eri, gf, se):
    ''' Self-consistent loop for the density matrix via the HF self-
        consistent field.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin

    Returns:
        :class:`SelfEnergy`, :class:`GreensFunction` and a boolean
        indicating whether convergence was successful.
    '''

    assert type(gf[0]) is aux.GreensFunction
    assert type(gf[1]) is aux.GreensFunction
    assert type(se[0]) is aux.SelfEnergy
    assert type(se[1]) is aux.SelfEnergy

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    diis = lib.diis.DIIS(agf2)
    diis.space = agf2.fock_diis_space
    diis.min_space = agf2.fock_diis_min_space
    focka, fockb = agf2.get_fock(eri, gf)
    sea, seb = se
    gfa, gfb = gf

    nalph, nbeta = agf2.nocc
    nmoa, nmob = eri.nmo
    nauxa, nauxb = sea.naux, seb.naux
    nqmoa, nqmob = nauxa+nmoa, nauxb+nmob
    bufa, bufb = np.zeros((nqmoa, nqmoa)), np.zeros((nqmob, nqmob))
    rdm1a_prev = 0
    rdm1b_prev = 0
    converged = False
    opts = dict(tol=agf2.conv_tol_nelec, maxiter=agf2.max_cycle_inner)

    for niter1 in range(1, agf2.max_cycle_outer+1):
        sea, opt = minimize_chempot(sea, focka, nalph, x0=sea.chempot,
                                    occupancy=1, **opts)
        seb, opt = minimize_chempot(seb, fockb, nbeta, x0=seb.chempot,
                                    occupancy=1, **opts)

        for niter2 in range(1, agf2.max_cycle_inner+1):
            wa, va = sea.eig(focka, chempot=0.0, out=bufa)
            wb, vb = seb.eig(fockb, chempot=0.0, out=bufb)
            sea.chempot, nerra = \
                    binsearch_chempot((wa, va), nmoa, nalph, occupancy=1)
            seb.chempot, nerrb = \
                    binsearch_chempot((wb, vb), nmob, nbeta, occupancy=1)
            nerr = max(nerra, nerrb)

            wa, va = sea.eig(focka, out=bufa)
            wb, vb = seb.eig(fockb, out=bufb)
            gfa = aux.GreensFunction(wa, va[:nmoa], chempot=sea.chempot)
            gfb = aux.GreensFunction(wb, vb[:nmob], chempot=seb.chempot)

            gf = (gfa, gfb)
            focka, fockb = agf2.get_fock(eri, gf)
            rdm1a, rdm1b = agf2.make_rdm1(gf)
            focka, fockb = diis.update(np.array((focka, fockb)), xerr=None)

            if niter2 > 1:
                derra = np.max(np.absolute(rdm1a - rdm1a_prev))
                derrb = np.max(np.absolute(rdm1b - rdm1b_prev))
                derr = max(derra, derrb)

                if derr < agf2.conv_tol_rdm1:
                    break

            rdm1a_prev = rdm1a.copy()
            rdm1b_prev = rdm1b.copy()

        log.debug1('fock loop %d  cycles = %d  dN = %.3g  |ddm| = %.3g',
                   niter1, niter2, nerr, derr)
        cput1 = log.timer_debug1('fock loop %d'%niter1, *cput1)

        if derr < agf2.conv_tol_rdm1 and abs(nerr) < agf2.conv_tol_nelec:
            converged = True
            break

    se = (sea, seb)

    log.info('fock converged = %s' % converged)
    log.info(' alpha: chempot = %.9g  dN = %.3g  |ddm| = %.3g',
             sea.chempot, nerra, derra)
    log.info(' beta:  chempot = %.9g  dN = %.3g  |ddm| = %.3g',
             seb.chempot, nerrb, derrb)
    log.timer('fock loop', *cput0)

    return gf, se, converged


def energy_1body(agf2, eri, gf):
    ''' Calculates the one-body energy according to the UHF form.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin

    Returns:
        One-body energy
    '''

    assert type(gf[0]) is aux.GreensFunction
    assert type(gf[1]) is aux.GreensFunction

    rdm1 = agf2.make_rdm1(gf)
    fock = agf2.get_fock(eri, gf)

    e1b_a = 0.5 * np.sum(rdm1[0] * (eri.h1e[0] + fock[0]))
    e1b_b = 0.5 * np.sum(rdm1[1] * (eri.h1e[1] + fock[1]))

    e1b = e1b_a + e1b_b
    e1b += agf2.energy_nuc()

    return e1b


def energy_2body(agf2, gf, se):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula. The formula is symmetric and only
        one side needs to be calculated.

    Args:
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin

    Returns:
        Two-body energy
    '''

    e2b_a = ragf2.energy_2body(agf2, gf[0], se[0])
    e2b_b = ragf2.energy_2body(agf2, gf[1], se[1])

    e2b = (e2b_a + e2b_b) * 0.5

    return e2b


def energy_mp2(agf2, gf, se):
    ''' Calculates the two-bdoy energy using analytically integrated
        Galitskii-Migdal formula for an MP2 self-energy. Per the
        definition of one- and two-body partitioning in the Dyson
        equation, this reuslt is half of :func:`energy_2body`.

    Args:
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin

    Returns:
        MP2 energy
    '''

    emp2_a = ragf2.energy_mp2(agf2, gf[0], se[0])
    emp2_b = ragf2.energy_mp2(agf2, gf[1], se[1])

    emp2 = (emp2_a + emp2_b) * 0.5

    return emp2


class UAGF2(ragf2.RAGF2):
    ''' Unrestricted AGF2 with canonical HF reference

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-8.
        conv_tol_nelec : float
            Convergence threshold for the number of electrons. Default
            value is 1e-6.
        max_cycle : int
            Maximum number of AGF2 iterations. Default value is 50.
        max_cycle_outer : int
            Maximum number of outer Fock loop iterations. Default
            value is 20.
        max_cycle_inner : int
            Maximum number of inner Fock loop iterations. Default
            value is 50.
        weight_tol : float
            Threshold in spectral weight of auxiliaries to be considered
            zero. Default 1e-11.
        diis : bool or lib.diis.DIIS
            Whether to use DIIS, can also be a lib.diis.DIIS object. Default
            value is True.
        diis_space : int
            DIIS space size. Default value is 8.
        diis_min_space : int
            Minimum space of DIIS. Default value is 1.
        fock_diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        fock_diis_min_space :
            Minimum space of DIIS. Default value is 1.
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        damping : float
            Damping factor for the self-energy. Default value is 0.0

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_init : float
            Initial correlation energy (truncated MP2)
        converged : bool
            Whether convergence was successful
        se : tuple of SelfEnergy
            Auxiliaries of the self-energy for each spin
        gf : tuple of GreensFunction
            Auxiliaries of the Green's function for each spin
    '''

    energy_1body = energy_1body
    energy_2body = energy_2body
    fock_loop = fock_loop
    build_se_part = build_se_part

    def ao2mo(self, mo_coeff=None):
        ''' Get the electronic repulsion integrals in MO basis.
        '''

        nmo = max(self.nmo)
        mem_incore = ((nmo*(nmo+1)//2)**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if (self._scf._eri is not None and
                (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            eri = _make_mo_eris_incore(self, mo_coeff)
        else:
            logger.warn(self, 'MO eris are outcore - this may be very '
                              'slow for agf2. increasing max_memory or '
                              'using density fitting is recommended.')
            eri = _make_mo_eris_outcore(self, mo_coeff)

        return eri

    def make_rdm1(self, gf=None):
        ''' Compute the one-body reduced density matrix in MO basis.

        Kwargs:
            gf : tuple of GreensFunction
                Auxiliaries of the Green's functions for each spin

        Returns:
            tuple of ndarray of density matrices
        '''

        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        rdm1_a = gf[0].make_rdm1(occupancy=1)
        rdm1_b = gf[1].make_rdm1(occupancy=1)

        return (rdm1_a, rdm1_b)

    def get_fock(self, eri=None, gf=None, rdm1=None):
        ''' Computes the physical space Fock matrix in MO basis.
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf

        return get_fock(self, eri, gf=gf, rdm1=rdm1)

    def energy_mp2(self, mo_energy=None, se=None):
        if mo_energy is None: mo_energy = self.mo_energy
        if se is None: se = self.build_se(gf=self.gf)

        self.e_init = energy_mp2(self, mo_energy, se)

        return self.e_init

    def init_gf(self, frozen=False):
        ''' Builds the Hartree-Fock Green's function.

        Returns:
            tuple of :class:`GreensFunction`, tuple of :class:`SelfEnergy`
        '''

        nmoa, nmob = self.nmo
        nocca, noccb = self.nocc

        energy = self.mo_energy
        coupling = (np.eye(nmoa), np.eye(nmob))

        focka = np.diag(energy[0])
        fockb = np.diag(energy[1])

        cpt_a = binsearch_chempot(focka, nmoa, nocca, occupancy=1)[0]
        cpt_b = binsearch_chempot(fockb, nmob, noccb, occupancy=1)[1]

        if frozen:
            mask = get_frozen_mask(self)
            energy = (energy[0][mask[0]], energy[1][mask[1]])
            coupling = (coupling[0][:,mask[0]], coupling[1][:,mask[1]])

        gf_a = aux.GreensFunction(energy[0], coupling[0], chempot=cpt_a)
        gf_b = aux.GreensFunction(energy[1], coupling[1], chempot=cpt_b)

        gf = (gf_a, gf_b)

        return gf

    def build_gf(self, eri=None, gf=None, se=None):
        ''' Builds the auxiliaries of the Green's functions by solving
            the Dyson equation for each spin.

        Kwargs:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : tuple of GreensFunction
                Auxiliaries of the Green's function for each spin
            se : tuple of SelfEnergy
                Auxiliaries of the self-energy for each spin

        Returns:
            tuple of :class:`GreensFunction`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()
        if se is None: se = self.build_se(eri, gf)

        focka, fockb = self.get_fock(eri, gf)

        gf_a = se[0].get_greens_function(focka)
        gf_b = se[1].get_greens_function(fockb)

        return (gf_a, gf_b)

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        ''' Builds the auxiliaries of the self-energy.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : tuple of GreensFunction
                Auxiliaries of the Green's function

        Kwargs:
            os_factor : float
                Opposite-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            ss_factor : float
                Same-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            se_prev : SelfEnergy
                Previous self-energy for damping. Default value is None

        Returns
            tuple of :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = (gf[0].get_occupied(), gf[1].get_occupied())
        gf_vir = (gf[0].get_virtual(), gf[1].get_virtual())

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)

        se_a = aux.combine(se_occ[0], se_vir[0])
        se_b = aux.combine(se_occ[1], se_vir[1])

        if se_prev is not None and self.damping != 0.0:
            se_a_prev, se_b_prev = se_prev
            se_a.coupling *= np.sqrt(1.0-self.damping)
            se_b.coupling *= np.sqrt(1.0-self.damping)
            se_a_prev.coupling *= np.sqrt(self.damping)
            se_b_prev.coupling *= np.sqrt(self.damping)
            se_a = aux.combine(se_a, se_a_prev)
            se_b = aux.combine(se_b, se_b_prev)
            se_a = se_a.compress(n=(None,0))
            se_b = se_b.compress(n=(None,0))

        return (se_a, se_b)

    def run_diis(self, se, diis=None):
        ''' Runs the direct inversion of the iterative subspace for the
            self-energy.

        Args:
            se : SelfEnergy
                Auxiliaries of the self-energy
            diis : lib.diis.DIIS
                DIIS object

        Returns:
            tuple of :class:`SelfEnergy`
        '''

        if diis is None:
            return se

        se_occ_a, se_occ_b = (se[0].get_occupied(), se[1].get_occupied())
        se_vir_a, se_vir_b = (se[0].get_virtual(), se[1].get_virtual())

        vv_occ_a = np.dot(se_occ_a.coupling, se_occ_a.coupling.T)
        vv_occ_b = np.dot(se_occ_b.coupling, se_occ_b.coupling.T)
        vv_vir_a = np.dot(se_vir_a.coupling, se_vir_a.coupling.T)
        vv_vir_b = np.dot(se_vir_b.coupling, se_vir_b.coupling.T)

        vev_occ_a = np.dot(se_occ_a.coupling * se_occ_a.energy[None], se_occ_a.coupling.T)
        vev_occ_b = np.dot(se_occ_b.coupling * se_occ_b.energy[None], se_occ_b.coupling.T)
        vev_vir_a = np.dot(se_vir_a.coupling * se_vir_a.energy[None], se_vir_a.coupling.T)
        vev_vir_b = np.dot(se_vir_b.coupling * se_vir_b.energy[None], se_vir_b.coupling.T)

        dat = np.array([vv_occ_a, vv_vir_a, vev_occ_a, vev_vir_a,
                        vv_occ_b, vv_vir_b, vev_occ_b, vev_vir_b])
        dat = diis.update(dat)
        vv_occ_a, vv_vir_a, vev_occ_a, vev_vir_a, \
                vv_occ_b, vv_vir_b, vev_occ_b, vev_vir_b = dat

        se_occ_a = aux.SelfEnergy(*_agf2.cholesky_build(vv_occ_a, vev_occ_a), chempot=se[0].chempot)
        se_vir_a = aux.SelfEnergy(*_agf2.cholesky_build(vv_vir_a, vev_vir_a), chempot=se[0].chempot)
        se_occ_b = aux.SelfEnergy(*_agf2.cholesky_build(vv_occ_b, vev_occ_b), chempot=se[1].chempot)
        se_vir_b = aux.SelfEnergy(*_agf2.cholesky_build(vv_vir_b, vev_vir_b), chempot=se[1].chempot)
        se = (aux.combine(se_occ_a, se_vir_a), aux.combine(se_occ_b, se_vir_b))

        return se

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.agf2 import dfuagf2
        myagf2 = dfuagf2.DFUAGF2(self._scf)
        myagf2.__dict__.update(self.__dict__)

        if with_df is not None:
            myagf2.with_df = with_df

        if auxbasis is not None and myagf2.with_df.auxbasis != auxbasis:
            import copy
            myagf2.with_df = copy.copy(myagf2.with_df)
            myagf2.with_df.auxbasis = auxbasis

        return myagf2

    def get_ip(self, gf, nroots=5):
        gf_occ = (gf[0].get_occupied(), gf[1].get_occupied())
        spin = np.array([0,]*gf_occ[0].naux + [1,]*gf_occ[1].naux)
        e_ip = np.concatenate([gf_occ[0].energy, gf_occ[1].energy], axis=0)
        v_ip = np.concatenate([gf_occ[0].coupling, gf_occ[1].coupling], axis=1)

        mask = np.argsort(e_ip)
        spin = list(spin[mask][-nroots:])[::-1]
        e_ip = list(-e_ip[mask][-nroots:])[::-1]
        v_ip = list(v_ip[:,mask][:,-nroots:].T)[::-1]

        return e_ip, v_ip, spin

    def ipagf2(self, nroots=5):
        e_ip, v_ip, spin = self.get_ip(self.gf, nroots=nroots)

        for n, en, vn, sn in zip(range(nroots), e_ip, v_ip, spin):
            qpwt = np.linalg.norm(vn)**2
            tag = ['alpha', 'beta'][sn]
            logger.note(self, 'IP energy level %d E = %.16g  QP weight = %0.6g  (%s)', n, en, qpwt, tag)

        if nroots == 1:
            return e_ip[0], v_ip[0]
        else:
            return e_ip, v_ip

    def get_ea(self, gf, nroots=5):
        gf_vir = (gf[0].get_virtual(), gf[1].get_virtual())
        spin = np.array([0,]*gf_vir[0].naux + [1,]*gf_vir[1].naux)
        e_ea = np.concatenate([gf_vir[0].energy, gf_vir[1].energy], axis=0)
        v_ea = np.concatenate([gf_vir[0].coupling, gf_vir[1].coupling], axis=1)

        mask = np.argsort(e_ea)
        spin = list(spin[mask][:nroots])
        e_ea = list(e_ea[mask][:nroots])
        v_ea = list(v_ea[:,mask][:,:nroots].T)

        return e_ea, v_ea, spin

    def eaagf2(self, nroots=5):
        e_ea, v_ea, spin = self.get_ea(self.gf, nroots=nroots)

        for n, en, vn, sn in zip(range(nroots), e_ea, v_ea, spin):
            qpwt = np.linalg.norm(vn)**2
            tag = ['alpha', 'beta'][sn]
            logger.note(self, 'EA energy level %d E = %.16g  QP weight = %0.6g  (%s)', n, en, qpwt, tag)

        if nroots == 1:
            return e_ea[0], v_ea[0]
        else:
            return e_ea, v_ea


    @property
    def nocc(self):
        if self._nocc is None:
            self._nocc = (np.sum(self.mo_occ[0] > 0), np.sum(self.mo_occ[1] > 0))
        return self._nocc
    @nocc.setter
    def nocc(self, val):
        self._nocc = val

    @property
    def nmo(self):
        if self._nmo is None:
            self._nmo = (self.mo_occ[0].size, self.mo_occ[1].size)
        return self._nmo
    @nmo.setter
    def nmo(self, val):
        self._nmo = val

    @property
    def qmo_energy(self):
        return (self.gf[0].energy, self.gf[1].energy)

    @property
    def qmo_coeff(self):
        ''' Gives the couplings in AO basis '''
        return (np.dot(self.mo_coeff[0], self.gf[0].coupling),
                np.dot(self.mo_coeff[1], self.gf[1].coupling))

    @property
    def qmo_occ(self):
        coeff_a = self.gf[0].get_occupied().coupling
        coeff_b = self.gf[1].get_occupied().coupling
        occ_a = np.linalg.norm(coeff_a, axis=0) ** 2
        occ_b = np.linalg.norm(coeff_b, axis=0) ** 2
        vir_a = np.zeros_like(self.gf[0].get_virtual().energy)
        vir_b = np.zeros_like(self.gf[1].get_virtual().energy)
        qmo_occ_a = np.concatenate([occ_a, vir_a])
        qmo_occ_b = np.concatenate([occ_b, vir_b])
        return qmo_occ_a, qmo_occ_b


def get_frozen_mask(agf2):
    with lib.temporary_env(agf2, _nocc=None, _nmo=None):
        return _get_frozen_mask(agf2)


class _ChemistsERIs:
    ''' (pq|rs)

    MO integrals stored in s4 symmetry, we only need QMO integrals
    in low-symmetry tensors and s4 is highest supported by _vhf
    '''

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.nmo = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        self.mo_coeff = mo_coeff

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        h1e_ao = agf2._scf.get_hcore()
        vhf = agf2._scf.get_veff(agf2.mol, dm)
        fock_ao = agf2._scf.get_fock(vhf=vhf, dm=dm)

        self.h1e = (np.dot(np.dot(mo_coeff[0].conj().T, h1e_ao), mo_coeff[0]),
                    np.dot(np.dot(mo_coeff[1].conj().T, h1e_ao), mo_coeff[1]))
        self.fock = (np.dot(np.dot(mo_coeff[0].conj().T, fock_ao[0]), mo_coeff[0]),
                     np.dot(np.dot(mo_coeff[1].conj().T, fock_ao[1]), mo_coeff[1]))

        self.h1e = (mpi_helper.bcast(self.h1e[0]), mpi_helper.bcast(self.h1e[1]))
        self.fock = (mpi_helper.bcast(self.fock[0]), mpi_helper.bcast(self.fock[1]))

        self.e_hf = mpi_helper.bcast(agf2._scf.e_tot)

        self.nmo = agf2.nmo
        nocca, noccb = self.nocc = agf2.nocc
        self.mol = agf2.mol

        mo_e = (self.fock[0].diagonal(), self.fock[1].diagonal())
        gap_a = abs(mo_e[0][:nocca,None] - mo_e[0][None,nocca:]).min()
        gap_b = abs(mo_e[1][:noccb,None] - mo_e[1][None,noccb:]).min()
        gap = min(gap_a, gap_b)
        if gap < 1e-5:
            logger.warn(agf2, 'HOMO-LUMO gap %s may be too small for AGF2', gap)

        return self

def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    moa, mob = eris.mo_coeff
    nmoa, nmob = eris.nmo

    eri_aa = ao2mo.incore.full(agf2._scf._eri, moa, verbose=log)
    eri_bb = ao2mo.incore.full(agf2._scf._eri, mob, verbose=log)

    eri_aa = ao2mo.addons.restore('s4', eri_aa, nmoa)
    eri_bb = ao2mo.addons.restore('s4', eri_bb, nmob)

    eri_ab = ao2mo.incore.general(agf2._scf._eri, (moa,moa,mob,mob), verbose=log)
    assert eri_ab.shape == (nmoa*(nmob+1)//2, nmob*(nmob+1)//2)
    eri_ba = np.transpose(eri_ab)

    eris.eri_aa = eri_aa
    eris.eri_ab = eri_ab
    eris.eri_ba = eri_ba
    eris.eri_bb = eri_bb
    eris.eri = ((eri_aa, eri_ab), (eri_ba, eri_bb))

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_mo_eris_outcore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    mol = agf2.mol
    moa = np.asarray(eris.mo_coeff[0], order='F')
    mob = np.asarray(eris.mo_coeff[1], order='F')
    nmoa, nmob = eris.nmo

    eris.feri = lib.H5TmpFile()

    ao2mo.outcore.full(mol, moa, eris.feri, dataname='mo/aa')
    ao2mo.outcore.full(mol, mob, eris.feri, dataname='mo/bb')
    ao2mo.outcore.general(mol, (moa,moa,mob,mob), eris.feri, dataname='mo/ab', verbose=log)
    ao2mo.outcore.general(mol, (mob,mob,moa,moa), eris.feri, dataname='mo/ba', verbose=log)

    eris.eri_aa = eris.feri['mo/aa']
    eris.eri_ab = eris.feri['mo/ab']
    eris.eri_ba = eris.feri['mo/ba']
    eris.eri_bb = eris.feri['mo/bb']

    eris.eri = ((eris.eri_aa, eris.eri_ab), (eris.eri_ba, eris.eri_bb))

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs_a, coeffs_b, spin=None):
    ''' Returns nested tuple of ndarray

    spin = None: ((aaaa, aabb), (bbaa, bbbb))
    spin = 0: (aaaa, aabb)
    spin = 1: (bbbb, bbaa)
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmo = eri.nmo
    nmoa, nmob = nmo

    cxa = np.eye(nmoa)
    cxb = np.eye(nmob)
    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = get_frozen_mask(agf2)
        cxa = cxa[:,mask[0]]
        cxb = cxb[:,mask[1]]

    # npaira, npairb = nmoa*(nmoa+1)//2, nmob*(nmob+1)//2
    cia, cja, caa = coeffs_a
    cib, cjb, cab = coeffs_b
    nia, nja, naa = [x.shape[1] for x in coeffs_a]
    nib, njb, nab = [x.shape[1] for x in coeffs_b]

    if spin is None or spin == 0:
        c_aa = (cxa, cia, cja, caa)
        c_ab = (cxa, cia, cjb, cab)

        qeri_aa = ao2mo.incore.general(eri.eri_aa, c_aa, compact=False, verbose=log)
        qeri_ab = ao2mo.incore.general(eri.eri_ab, c_ab, compact=False, verbose=log)

        qeri_aa = qeri_aa.reshape(cxa.shape[1], nia, nja, naa)
        qeri_ab = qeri_ab.reshape(cxa.shape[1], nia, njb, nab)

    if spin is None or spin == 1:
        c_bb = (cxb, cib, cjb, cab)
        c_ba = (cxb, cib, cja, caa)

        qeri_bb = ao2mo.incore.general(eri.eri_bb, c_bb, compact=False, verbose=log)
        qeri_ba = ao2mo.incore.general(eri.eri_ba, c_ba, compact=False, verbose=log)

        qeri_bb = qeri_bb.reshape(cxb.shape[1], nib, njb, nab)
        qeri_ba = qeri_ba.reshape(cxb.shape[1], nib, nja, naa)

    if spin is None:
        qeri = ((qeri_aa, qeri_ab), (qeri_ba, qeri_bb))
    elif spin == 0:
        qeri = (qeri_aa, qeri_ab)
    elif spin == 1:
        qeri = (qeri_bb, qeri_ba)

    log.timer('QMO integral transformation', *cput0)

    return qeri

def _make_qmo_eris_outcore(agf2, eri, coeffs_a, coeffs_b, spin=None):
    ''' Returns nested tuple of H5 dataset

    spin = None: ((aaaa, aabb), (bbaa, bbbb))
    spin = 0: (aaaa, aabb)
    spin = 1: (bbbb, bbaa)
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmo = eri.nmo
    nmoa, nmob = nmo

    mask = get_frozen_mask(agf2)
    frozena = np.sum(~mask[0])
    frozenb = np.sum(~mask[1])

    # npaira, npairb = nmoa*(nmoa+1)//2, nmob*(nmob+1)//2
    cia, cja, caa = coeffs_a
    cib, cjb, cab = coeffs_b
    nia, nja, naa = [x.shape[1] for x in coeffs_a]
    nib, njb, nab = [x.shape[1] for x in coeffs_b]

    # possible to have incore MO, outcore QMO
    if getattr(eri, 'feri', None) is None:
        eri.feri = lib.H5TmpFile()
    else:
        for key in ['aa', 'ab', 'ba', 'bb']:
            if 'qmo/%s'%key in eri.feri:
                del eri.feri['qmo/%s'%key]

    if spin is None or spin == 0:
        eri.feri.create_dataset('qmo/aa', (nmoa-frozena, nia, nja, naa), 'f8')
        eri.feri.create_dataset('qmo/ab', (nmoa-frozena, nia, njb, nab), 'f8')

        blksize = _agf2.get_blksize(agf2.max_memory, (nmoa**3, nmoa*nja*naa),
                                                (nmoa*nmob**2, nmoa*njb*nab))
        blksize = min(nmoa, max(BLKMIN, blksize))
        log.debug1('blksize (uagf2._make_qmo_eris_outcore) = %d', blksize)

        tril2sq = lib.square_mat_in_trilu_indices(nmoa)
        q1 = 0
        for p0, p1 in lib.prange(0, nmoa, blksize):
            if not np.any(mask[0][p0:p1]):
                # block is fully frozen
                continue

            inds = np.arange(p0, p1)[mask[0][p0:p1]]
            q0, q1 = q1, q1 + len(inds)
            idx = list(np.concatenate(tril2sq[inds]))

            # aa
            buf = eri.eri_aa[idx] # (blk, nmoa, npaira)
            buf = buf.reshape((q1-q0)*nmoa, -1) # (blk*nmoa, npaira)

            jasym_aa, nja_aa, cja_aa, sja_aa = ao2mo.incore._conc_mos(cja, caa)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_aa, sja_aa, 's2kl', 's1')
            buf = buf.reshape(q1-q0, nmoa, nja, naa)

            buf = lib.einsum('xpja,pi->xija', buf, cia)
            eri.feri['qmo/aa'][q0:q1] = np.asarray(buf, order='C')

            # ab
            buf = eri.eri_ab[idx] # (blk, nmoa, npairb)
            buf = buf.reshape((q1-q0)*nmob, -1) # (blk*nmoa, npairb)

            jasym_ab, nja_ab, cja_ab, sja_ab = ao2mo.incore._conc_mos(cjb, cab)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_ab, sja_ab, 's2kl', 's1')
            buf = buf.reshape(q1-q0, nmoa, njb, nab)

            buf = lib.einsum('xpja,pi->xija', buf, cia)
            eri.feri['qmo/ab'][q0:q1] = np.asarray(buf, order='C')

    if spin is None or spin == 1:
        eri.feri.create_dataset('qmo/ba', (nmob-frozenb, nib, nja, naa), 'f8')
        eri.feri.create_dataset('qmo/bb', (nmob-frozenb, nib, njb, nab), 'f8')

        max_memory = agf2.max_memory - lib.current_memory()[0]
        blksize = int((max_memory/8e-6) / max(nmob**3+nmob*njb*nab,
                                              nmob*nmoa**2*nja*naa))
        blksize = min(nmob, max(BLKMIN, blksize))
        log.debug1('blksize (uagf2._make_qmo_eris_outcore) = %d', blksize)

        tril2sq = lib.square_mat_in_trilu_indices(nmob)
        q1 = 0
        for p0, p1 in lib.prange(0, nmob, blksize):
            if not np.any(mask[1][p0:p1]):
                # block is fully frozen
                continue

            inds = np.arange(p0, p1)[mask[1][p0:p1]]
            q0, q1 = q1, q1 + len(inds)
            idx = list(np.concatenate(tril2sq[inds]))

            # ba
            buf = eri.eri_ba[idx] # (blk, nmob, npaira)
            buf = buf.reshape((q1-q0)*nmob, -1) # (blk*nmob, npaira)

            jasym_ba, nja_ba, cja_ba, sja_ba = ao2mo.incore._conc_mos(cja, caa)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_ba, sja_ba, 's2kl', 's1')
            buf = buf.reshape(q1-q0, nmob, nja, naa)

            buf = lib.einsum('xpja,pi->xija', buf, cib)
            eri.feri['qmo/ba'][q0:q1] = np.asarray(buf, order='C')

            # bb
            buf = eri.eri_bb[idx] # (blk, nmob, npairb)
            buf = buf.reshape((q1-q0)*nmob, -1) # (blk*nmob, npairb)

            jasym_bb, nja_bb, cja_bb, sja_bb = ao2mo.incore._conc_mos(cjb, cab)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_bb, sja_bb, 's2kl', 's1')
            buf = buf.reshape(q1-q0, nmob, njb, nab)

            buf = lib.einsum('xpja,pi->xija', buf, cib)
            eri.feri['qmo/bb'][q0:q1] = np.asarray(buf, order='C')

    if spin is None:
        qeri = ((eri.feri['qmo/aa'], eri.feri['qmo/ab']),
                (eri.feri['qmo/ba'], eri.feri['qmo/bb']))
    elif spin == 0:
        qeri = (eri.feri['qmo/aa'], eri.feri['qmo/ab'])
    elif spin == 1:
        qeri = (eri.feri['qmo/bb'], eri.feri['qmo/ba'])

    log.timer('QMO integral transformation', *cput0)

    return qeri



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', charge=-1, spin=1, verbose=3)
    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-11
    uhf.run()

    uagf2 = UAGF2(uhf, frozen=0)

    uagf2.run()
    uagf2.ipagf2(nroots=5)
    uagf2.eaagf2(nroots=5)


    uagf2 = uagf2.density_fit()
    uagf2.run()
