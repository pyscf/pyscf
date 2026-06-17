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
Spin-restricted eigenvalue self-consistent GW method based on the analytic continuation scheme.

References:
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import numpy as np
import scipy
import time

from pyscf import dft, scf
from pyscf.lib import diis, einsum, logger, temporary_env

from pyscf.gw.gw_ac import GWAC, set_frozen_orbs, _mo_energy_without_core, _mo_without_core, get_sigma, \
    get_sigma_outcore
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC

def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo
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
            Lpq = gw.Lpq = gw.ao2mo(mo_coeff_frz)

    hcore = mo_coeff_frz.T @ mf.get_hcore() @ mo_coeff_frz
    if gw.vhf_df is False:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
            rhf = mf
        else:
            rhf = scf.RHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vjk = mo_coeff_frz.T @ rhf.get_veff(dm=dm) @ mo_coeff_frz
        vk = vjk - mo_coeff_frz.T @ rhf.get_j(dm=dm) @ mo_coeff_frz
    else:
        # TODO: smearing
        vj = 2.0 * einsum('Lii,Lpq->pq', Lpq[:, :nocc, :nocc], Lpq)
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
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

        # compute self-energy on imaginary
        mo_energy_w = mf_mo_energy_frz if gw.W0 is True else mo_energy_frz
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
                mo_energy_w=mo_energy_w, eval_freqs=eval_freqs_with_zero, fullsigma=False
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, iw_cutoff=gw.ac_iw_cutoff,
                mo_energy_w=mo_energy_w,
                eval_freqs=eval_freqs_with_zero, fullsigma=False
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
        for ip, p in enumerate(orbs_frz):

            def quasiparticle(omega):
                sigmaR = acobj[ip].ac_eval(omega).real
                return omega - (ham_hf[p, p] + sigmaR)

            try:
                mo_energy[orbs[ip]] = scipy.optimize.newton(
                    quasiparticle, mo_energy_prev[orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter
                )
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', orbs[ip])

        # update quasiparticle energy through DIIS
        gw.mo_energy = gw_diis.update(mo_energy)

        # update non-frozen attribute and Fermi level
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

        diff = abs(np.sum(1.0 / mo_energy_frz - 1.0 / mo_energy_frz_prev)) / nmo / nmo
        if diff < gw.conv_tol:
            conv = True

        logger.info(gw, 'EVGW cycle= %d  |delta G|= %4.3g', cycle + 1, diff)
        cycle += 1

        with np.printoptions(threshold=len(gw._scf.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    logger.debug(gw, 'EVGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(gw._scf.mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    return


class EVGW(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.W0 = False  # evGW0
        self.max_cycle = 30  # max evGW cycle
        self.conv_tol = 1.0e-7  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name
        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
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
        if self.ac == "pade":
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
        """Run an eigenvalue-self-consistent GW calculation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return
