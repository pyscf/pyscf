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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
# Author: Christopher Hillenbrand <chillenbrand15@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Periodic spin-unrestricted eigenvalue self-consistent GW method based on the analytic continuation scheme.

References:
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import os
import scipy
import numpy as np

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.pbc import df, dft, scf
from pyscf.pbc.lib import chkfile

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.pbc.gw.kugw_ac import KUGWAC, get_sigma, get_ef, _mo_energy_frozen


def kernel(gw):
    mf = gw._scf
    nkpts = gw.nkpts
    nmo = gw.nmo[0]

    # set frozen orbitals
    gw.set_frozen_orbs()
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz
    kptlist = gw.kptlist
    if kptlist is None:
        gw.kptlist = kptlist = range(gw.nkpts)
    mo_energy = gw.mo_energy
    mo_energy_frz = _mo_energy_frozen(gw, gw.mo_energy)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # set up Fermi level
    ef = gw.ef = get_ef(kmf=mf, mo_energy=gw.mo_energy)

    # set up RHF object
    if isinstance(mf.with_df, df.GDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).density_fit()
    elif isinstance(mf.with_df, df.RSDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).rs_density_fit()
    if hasattr(mf, "sigma"):
        uhf = scf.addons.smearing_(uhf, sigma=mf.sigma, method=mf.smearing_method)
    uhf.with_df = gw.with_df
    uhf.mo_energy = np.array(gw.mo_energy, copy=True)
    uhf.mo_coeff = np.array(gw.mo_coeff, copy=True)
    uhf.mo_occ = np.array(gw.mo_occ, copy=True)
    uhf.verbose = uhf.mol.verbose = 0

    # initialize DIIS
    gw_diis = lib.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space
    # get hcore and veff matrix
    with temporary_env(mf.mol, verbose=0):
        hcore_ao = mf.get_hcore()
        dm = mf.make_rdm1()
    with temporary_env(uhf, verbose=0), temporary_env(uhf.with_df, verbose=0):
        veff = uhf.get_veff(dm_kpts=dm)
    hcore = np.zeros_like(veff)
    for s in range(2):
        for k in range(nkpts):
            hcore[s, k] = mf.mo_coeff[s][k].T.conj() @ hcore_ao[k] @ mf.mo_coeff[s][k]
            veff[s, k] = mf.mo_coeff[s][k].T.conj() @ veff[s, k] @ mf.mo_coeff[s][k]

    # finite size correction for exchange self-energy
    if gw.fc:
        vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
        for s in range(2):
            nocc_full = int(np.sum(gw._scf.mo_occ[s][0])) // 2
            for k in range(nkpts):
                for i in range(nocc_full):
                    veff[s, k][i, i] = veff[s, k][i, i] + vk_corr
    ham_hf = hcore + veff

    cycle = 0
    while gw.converged is False and cycle < max(1, gw.max_cycle):
        # calculate self-energy on imaginary axis
        gw.sigmaI, gw.omega = sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, kptlist=kptlist, iw_cutoff=gw.ac_iw_cutoff,
            fullsigma=False)

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(list(range(nmo)), gw.nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        elif gw.ac == 'pes':
            raise NotImplementedError
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))

        acobj.ac_fit(sigmaI, omega, axis=-1)

        # follow the section 5.1.1 Method evGW in 10.1021/acs.jctc.5b01238
        # for each pole the QP-equation is solved self-consistently
        # only iterative quasiparticle equation is implemented here
        mo_energy_old = np.array(mo_energy, copy=True)
        for s in range(2):
            for ik, k in enumerate(kptlist):
                for ip, p in enumerate(orbs):

                    def quasiparticle(omega):
                        sigmaR = acobj[s, ik, ip].ac_eval(omega)
                        return omega - (ham_hf[s, k, p, p] + sigmaR).real

                    try:
                        mo_energy[s, k, p] = scipy.optimize.newton(
                            quasiparticle, mo_energy_old[s, k, p], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                        )
                    except RuntimeError:
                        logger.warn(gw, 'QPE for spin=%d k=%d orbital=%d not converged!', s, k, p)

        # update quasiparticle energy through DIIS
        if cycle >= gw.diis_start_cycle:
            mo_energy = gw_diis.update(mo_energy)

        # update attributes in the GW object
        gw.mo_energy = np.array(mo_energy, copy=True)
        ef = gw.ef = get_ef(mf, mo_energy=mo_energy)
        gw.acobj = acobj

        # update NON-FROZEN quantities
        mo_energy_frz = _mo_energy_frozen(gw, mo_energy)

        # check density matrix convergence
        diff = 0
        for s in range(2):
            for k in range(nkpts):
                diff += abs(np.sum(1.0 / mo_energy[s, k] - 1.0 / mo_energy_old[s, k]))
        diff /= 2.0 * nkpts * nmo * nmo
        if diff < gw.conv_tol:
            gw.converged = True

        logger.info(gw, 'UEVGW cycle= %d  |delta G|= %-.4e', cycle + 1, diff)
        cycle += 1

        if gw.verbose >= logger.DEBUG:
            with np.printoptions(threshold=len(mf.mo_energy[0][0])):
                for k in range(nkpts):
                    logger.debug(gw, 'cycle %d GW mo_energy spin-up @ k%d =\n%s', cycle + 1, k, mo_energy[0, k])
                for k in range(nkpts):
                    logger.debug(gw, 'cycle %d GW mo_energy spin-down @ k%d =\n%s', cycle + 1, k, mo_energy[1, k])

    if gw.verbose >= logger.DEBUG:
        logger.debug(gw, '')
        with np.printoptions(threshold=len(mf.mo_energy[0][0])):
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy spin-up @ k%d =\n%s', k, mo_energy[0, k])
            for k in range(nkpts):
                logger.debug(gw, '  GW mo_energy spin-down @ k%d =\n%s', k, mo_energy[1, k])
        logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    return


class KUEVGW(KUGWAC):
    def __init__(self, mf, frozen=None):
        KUGWAC.__init__(self, mf, frozen=frozen)

        # options
        self.max_cycle = 30  # max cycle
        self.conv_tol = 1e-5  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_start_cycle = 0  # DIIS start cycle
        self.diis_file = None  # DIIS file

        # results
        self.converged = False  # qsGW convergence

        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira = nmoa - nocca
        nvirb = nmob - noccb
        log.info('GW (nocca, noccb) = (%d, %d), (nvira, nvirb) = (%d, %d)', nocca, noccb, nvira, nvirb)
        log.info('nkpt = %d', self.nkpts)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', str(self.frozen))
        if self.kptlist is not None:
            log.info('k-point list = %s', str(self.kptlist))
        if self.orbs is not None:
            log.info('orbital list = %s', str(self.orbs))
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('finite size corrections = %s', self.fc)
        if self.fc_grid is not None:
            log.info('grids for finite size corrections = %s', self.fc_grid)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('number of grids = %d', self.nw)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        # evGW settings
        log.info('max cycle = %d', self.max_cycle)
        log.info('density matrix convergence tolerance = %.2e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS start cycle = %d', self.diis_start_cycle)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""
        if self.converged:
            logger.note(self, 'UEVGW converged.')
        else:
            logger.note(self, 'UEVGW not converged.')
        return

    def kernel(self, orbs=None, kptlist=None):
        """Run evGW calculation.

        Parameters
        ----------
        orbs : list, optional
            orbital list to calculate self-energy, by default None
        kptlist : list, optional
            k-point list to calculate self-energy, by default None
        """
        if self.mo_energy is None:
            self.mo_energy = np.array(self._scf.mo_energy, copy=True)
        if self.mo_coeff is None:
            self.mo_coeff = np.array(self._scf.mo_coeff, copy=True)
        if self.mo_occ is None:
            self.mo_occ = np.array(self._scf.mo_occ, copy=True)

        if isinstance(self.frozen, list) and (not isinstance(self.frozen[0], list)):
            # make sure self.frozen is a list of lists if not frozen core
            self.frozen = [self.frozen, self.frozen]
        else:
            assert self.frozen is None or isinstance(self.frozen, (int, np.int64))

        self.orbs = orbs
        self.kptlist = kptlist

        if hasattr(self._scf, 'sigma'):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        nmo = self.nmo[0]
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'KUEVGW', *cput0)
        self._finalize()
        return
