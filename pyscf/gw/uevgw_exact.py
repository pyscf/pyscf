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
# Author: Jiachen Li <lijiachen.duke@gmail.com>
#

"""
Spin-unrestricted eigenvalue self-consistent GW method based on the fully analytic scheme.

References:
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
    J. Chem. Theory Comput. 9, 1, 232-246 (2013)
"""

import numpy as np
import scipy
import time

from pyscf import dft, scf
from pyscf.lib import diis, einsum, logger, temporary_env

from pyscf.gw.ugw_exact_df import UGWExactDF, diagonalize_phrpa, get_transition_density, get_sigma


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mo_energy = gw.mo_energy

    if gw.Lpq is None:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = Lpq = gw.ao2mo(gw.mo_coeff)

    hcore = mf.get_hcore()
    hcore = np.stack([mf.mo_coeff[s].T @ hcore @ mf.mo_coeff[s] for s in range(2)], axis=0)
    if gw.vhf_df is False:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vjk = uhf.get_veff(dm=dm)
        vjk = np.stack([mf.mo_coeff[s].T @ vjk[s] @ mf.mo_coeff[s] for s in range(2)], axis=0)
        vj_ao = uhf.get_j(dm=dm)
        vj = np.array([mf.mo_coeff[s].T @ (vj_ao[0] + vj_ao[1]) @ mf.mo_coeff[s] for s in range(2)])
        vk = vjk - vj
    else:
        # TODO: smearing
        vj = 2.0 * einsum('sLii,sLpq->spq', Lpq[:, :nocc, :nocc], Lpq)
        vk = -einsum('sLpi,sLiq->spq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
        vjk = vj + vk
    ham_hf = hcore + vjk  # HF Hamiltonian in MO space

    gw_diis = diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        mo_energy_prev = mo_energy.copy()

        if (gw.W0 is True and cycle == 0) or gw.W0 is False:
            gw.exci, _ = exci, xpy = diagonalize_phrpa(nocc=nocc, mo_energy=mo_energy, Lpq=Lpq, RPAE=gw.RPAE)
            gw.rho = rho = get_transition_density(nocc=nocc, xpy=xpy, Lpq=Lpq)

        # follow the section 5.1.1 Method evGW in 10.1021/acs.jctc.5b01238
        # for each pole the QP-equation is solved self-consistently
        # only iterative quasiparticle equation is implemented here
        def quasiparticle(qp_energy):
            sigma = get_sigma(
                nocc=nocc, mo_energy=qp_energy, mo_energy_prev=mo_energy_prev, exci=exci, rho=rho, eta=gw.eta,
                fullsigma=False)
            return qp_energy - (ham_hf + sigma).diagonal(axis1=1, axis2=2)

        try:
            mo_energy = scipy.optimize.newton(
                quasiparticle, mo_energy_prev, tol=gw.qpe_tol * nmo * 2.0, maxiter=gw.qpe_max_iter
            )
        except RuntimeError:
            logger.warn(gw, 'quasiparticle equation fails to converge!')

        # update quasiparticle energy through DIIS
        gw.mo_energy = gw_diis.update(mo_energy)

        diff = abs(np.sum(1.0 / mo_energy[0] - 1.0 / mo_energy_prev[0])) / nmo / nmo / 2.0
        diff += abs(np.sum(1.0 / mo_energy[1] - 1.0 / mo_energy_prev[1])) / nmo / nmo / 2.0
        if diff < gw.conv_tol:
            conv = True

        logger.info(gw, 'evGW cycle= %d  |delta G|= %4.3g', cycle + 1, diff)
        with np.printoptions(threshold=nmo):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
        cycle += 1

    logger.debug(gw, 'UEVGWExact %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])

    return


class UEVGWExact(UGWExactDF):
    def __init__(self, mf, auxbasis=None):
        UGWExactDF.__init__(self, mf, auxbasis=auxbasis)

        # options
        self.W0 = False  # evGW0
        self.max_cycle = 30  # max evGW cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name
        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('broadening parameter = %.3e', self.eta)
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
        assert self.frozen is None or self.frozen == 0
        assert self.orbs is None
        assert self.outcore is False
        assert hasattr(self._scf, 'sigma') is False

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return
