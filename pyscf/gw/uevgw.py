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
# Author: Christian Venturella <christian.venturella@gmail.com>
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Spin-unrestricted eigenvalue self-consistent GW method based on the analytic continuation scheme.

References:
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import numpy as np
import scipy
import time

from pyscf import dft, scf
from pyscf.lib import diis, einsum, logger, temporary_env

from pyscf.gw.ugw_ac import UGWAC, set_frozen_orbs, _mo_energy_without_core, _mo_without_core, get_sigma, \
    get_sigma_outcore
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mo_energy = gw.mo_energy

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mf_mo_energy_frz = _mo_energy_without_core(gw, gw._scf.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and gw.outcore is False:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = Lpq = gw.ao2mo(mo_coeff_frz)

    hcore = mf.get_hcore()
    hcore = np.stack([mo_coeff_frz[s].T @ hcore @ mo_coeff_frz[s] for s in range(2)], axis=0)
    if gw.vhf_df is False:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vjk = uhf.get_veff(dm=dm)
        vjk = np.stack([mo_coeff_frz[s].T @ vjk[s] @ mo_coeff_frz[s] for s in range(2)], axis=0)
        vj_ao = uhf.get_j(dm=dm)
        vj = np.array([mo_coeff_frz[s].T @ (vj_ao[0] + vj_ao[1]) @ mo_coeff_frz[s] for s in range(2)])
        vk = vjk - vj
    else:
        # TODO: smearing
        vj = 2.0 * einsum('sLii,sLpq->spq', Lpq[:, :nocc, :nocc], Lpq)
        vk = -einsum('sLpi,sLiq->spq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
        vjk = vj + vk
    gw.vk = vk
    ham_hf = hcore + vjk  # HF Hamiltonian in MO space

    gw_diis = diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        mo_energy_prev = mo_energy.copy()
        mo_energy_frz_prev = mo_energy_frz.copy()

        # compute self-energy on imaginary axis
        mo_energy_w = mf_mo_energy_frz if gw.W0 is True else mo_energy_frz
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_g=mo_energy_frz, mo_coeff=mo_coeff_frz,
                iw_cutoff=gw.ac_iw_cutoff, eval_freqs=eval_freqs_with_zero, mo_energy_w=mo_energy_w, fullsigma=False
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy=mo_energy_frz,
                iw_cutoff=gw.ac_iw_cutoff, eval_freqs=eval_freqs_with_zero,
                mo_energy_w=mo_energy_w, fullsigma=False
            )

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(orbs_frz, nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)
        gw.acobj = acobj

        # follow the section 5.1.1 Method evGW in 10.1021/acs.jctc.5b01238
        # for each pole the QP-equation is solved self-consistently
        # only iterative quasiparticle equation is implemented here
        mo_energy = mo_energy_prev.copy()
        for s in range(2):
            for ip, p in enumerate(orbs_frz):

                def quasiparticle(omega):
                    sigmaR = acobj[s, ip].ac_eval(omega).real
                    return omega - (ham_hf[s, p, p] + sigmaR)

                try:
                    mo_energy[s, orbs[ip]] = scipy.optimize.newton(
                        quasiparticle, mo_energy_prev[s, orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                    )
                except RuntimeError:
                    logger.warn(gw, 'QPE for spin=%d orbital=%d not converged!', s, orbs[ip])

        # update quasiparticle energy through DIIS
        gw.mo_energy = gw_diis.update(mo_energy)

        # update non-frozen attribute and Fermi level
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

        diff = abs(np.sum(1.0 / mo_energy_frz[0] - 1.0 / mo_energy_frz_prev[0])) / nmo / nmo / 2.0
        diff += abs(np.sum(1.0 / mo_energy_frz[1] - 1.0 / mo_energy_frz_prev[1])) / nmo / nmo / 2.0
        if diff < gw.conv_tol:
            conv = True

        logger.info(gw, 'UEVGW cycle= %d  |delta G|= %4.3g', cycle + 1, diff)
        cycle += 1

        with np.printoptions(threshold=len(mo_energy[0])):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])

    logger.debug(gw, 'UEVGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(mo_energy[0])):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])

    return


class UEVGW(UGWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        UGWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.W0 = False  # evGW0
        self.max_cycle = 30  # max evGW cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name
        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('frozen orbitals = %s', self.frozen)
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('outcore for self-energy= %s', self.outcore)
        if self.outcore is True:
            log.info('outcore segment size = %d', self.segsize)
        log.info('broadening parameter = %.3e', self.eta)
        if self.nw2 is None:
            log.info('number of grids = %d', self.nw)
        else:
            log.info('number of grids for integration= %d', self.nw)
            log.info('number of grids to be integrated = %d', self.nw2)
        log.info('analytic continuation method = %s', self.ac)
        log.info('imaginary frequency cutoff = %.1f', self.ac_iw_cutoff)
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        log.info('QPE max iter = %d', self.qpe_max_iter)
        log.info('QPE tolerance = %.1e', self.qpe_tol)
        # evGW parameters
        log.info('evGW0 = %s', self.W0)
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DISS file = %s', self.diis_file)
        log.info('')
        return

    def kernel(self):
        """Run a spin-unrestricted eigenvalue-self-consistent GW calculation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        if isinstance(self.frozen, list) and (not isinstance(self.frozen[0], list)):
            # make sure self.frozen is a list of lists if not frozen core
            self.frozen = [self.frozen, self.frozen]
        else:
            assert self.frozen is None or isinstance(self.frozen, (int, np.int64))

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return
