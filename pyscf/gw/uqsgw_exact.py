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
Spin-unrestricted quasiparticle self-consistent GW method based on the fully analytic scheme.

References:
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
    J. Chem. Theory Comput. 9, 1, 232-246 (2013)
"""

import numpy as np
import scipy

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from pyscf.gw.ugw_ac import UGWAC
from pyscf.gw.ugw_exact_df import diagonalize_phrpa, get_transition_density, get_sigma


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    if (not isinstance(gw._scf, dft.uks.UKS)) and isinstance(gw._scf, scf.uhf.UHF):
        uhf = gw._scf
    else:
        uhf = scf.UHF(gw.mol.copy(deep=True))
        uhf.verbose = uhf.mol.verbose = 0
        uhf.mo_energy = np.array(mo_energy, copy=True)
        uhf.mo_coeff = np.array(mo_coeff, copy=True)
        uhf.mo_occ = uhf.get_occ()
        if mf._eri is not None:
            uhf._eri = mf._eri
    dm_iter = uhf.make_rdm1()

    gw_diis = scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        # update Lpq
        if gw.Lpq_ao is None:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(mo_coeff)
        else:
            gw.Lpq = einsum('sLmn,smp,snq->sLpq', gw.Lpq_ao, mo_coeff, mo_coeff)

        # diagonalize the RPA matrix
        gw.exci, _ = exci, xpy = diagonalize_phrpa(nocc=nocc, mo_energy=mo_energy, Lpq=gw.Lpq, RPAE=gw.RPAE)

        # calculate the transition density
        gw.rho = rho = get_transition_density(nocc=nocc, xpy=xpy, Lpq=gw.Lpq)

        # calculate the self-energy
        sigma = get_sigma(
            nocc=nocc, mo_energy=mo_energy, mo_energy_prev=mo_energy, exci=exci, rho=rho, eta=gw.eta, fullsigma=True,
            mode=gw.mode
        )

        # obtain static correlation energy in AO basis
        vsig = np.asarray([mo_coeff[s].T @ sigma[s] @ mo_coeff[s] for s in range(2)])

        # update veff
        if gw.vhf_df is False:
            veff = uhf.get_veff(dm=dm_iter)
        else:
            veff = np.zeros_like(dm_iter)
            for s1 in range(2):
                for s2 in range(2):
                    veff[s1] = einsum('tLii,Lpq->pq', gw.Lpq[s2, :, : nocc[s2], : nocc[s2]], gw.Lpq[s1])
                    if s1 == s2:
                        veff[s1] -= einsum('sLpi,sLiq->spq', gw.Lpq[s2, :, :, : nocc[s2]], gw.Lpq[s1, :, : nocc[1], :])
            for s in range(2):
                veff[s] = mo_coeff[s].T @ veff[s] @ mo_coeff[s]

        # complete Hamiltonian through DIIS
        ham = hcore[None, :, :] + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # diagonalize
        for s in range(2):
            gw.mo_energy[s], gw.mo_coeff[s] = mo_energy[s], mo_coeff[s] = scipy.linalg.eigh(ham[s], ovlp)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = uhf.make_rdm1(mo_coeff=mo_coeff)
        norm_dm = np.linalg.norm(dm_iter[s] - dm_old[s]) / nmo / 2.0
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'UQSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        with np.printoptions(threshold=nmo):
            logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
            logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
        cycle += 1

    logger.debug(gw, 'UQSGWExact %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
        cycle += 1

    return


class UQSGWExact(UGWAC):
    def __init__(self, mf, auxbasis=None):
        UGWAC.__init__(self, mf, frozen=None, auxbasis=auxbasis)
        # options
        self.max_cycle = 20  # max cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file
        self.mode = 'b'  # mode for off-diagonal elements of static correlation self-energy
        self.RPAE = False  # exchange in RPA response

        # matrices
        self.Lpq_ao = None  # three-center density-fitting matrix in AO, used for impurity solver
        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('density-fitting for exchange = %s', self.vhf_df)
        log.info('broadening parameter = %.3e', self.eta)
        log.info('static sigma mode = %s', self.mode)
        # qsGW parameters
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def kernel(self):
        assert self.frozen is None or self.frozen == 0
        assert self.orbs is None
        assert self.outcore is False
        assert hasattr(self._scf, 'sigma') is False
        assert self.nmo[0] == self.nmo[1]

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return
