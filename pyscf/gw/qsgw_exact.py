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
Spin-restricted quasiparticle self-consistent GW method based on the fully analytic scheme.

References:
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
    J. Chem. Theory Comput. 9, 1, 232-246 (2013)
"""

import numpy as np
import scipy

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from pyscf.gw.gw_ac import GWAC
from pyscf.gw.gw_exact_df import diagonalize_phrpa, get_transition_density, get_sigma


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    if (not isinstance(gw._scf, dft.rks.RKS)) and isinstance(gw._scf, scf.hf.RHF):
        rhf = gw._scf
    else:
        rhf = scf.RHF(gw.mol.copy(deep=True))
        rhf.verbose = rhf.mol.verbose = 0
        rhf.mo_energy = np.array(mo_energy, copy=True)
        rhf.mo_coeff = np.array(mo_coeff, copy=True)
        rhf.mo_occ = rhf.get_occ()
        if mf._eri is not None:
            rhf._eri = mf._eri
    dm_iter = rhf.make_rdm1()

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
            gw.Lpq = einsum('Lmn,mp,nq->Lpq', gw.Lpq_ao, mo_coeff, mo_coeff)

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
        CS = np.matmul(mo_coeff.T, ovlp)
        vsig = CS.T @ sigma @ CS

        # update veff
        if gw.vhf_df is False:
            veff = rhf.get_veff(dm=dm_iter)
        else:
            vj = 2.0 * einsum('Lii,Lpq->pq', gw.Lpq[:, :nocc, :nocc], gw.Lpq)
            vk = -einsum('Lpi,Liq->pq', gw.Lpq[:, :, :nocc], gw.Lpq[:, :nocc, :])
            veff = CS.T @ (vj + vk) @ CS

        # complete Hamiltonian through DIIS
        ham = hcore + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        # diagonalize
        gw.mo_energy, gw.mo_coeff = mo_energy, mo_coeff = scipy.linalg.eigh(ham, ovlp)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = rhf.make_rdm1(mo_coeff=mo_coeff)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        with np.printoptions(threshold=nmo):
            logger.debug(gw, '  mo_energy =\n%s', mo_energy)
        cycle += 1

    logger.debug(gw, 'QSGWExact %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)

    return


class QSGWExact(GWAC):
    def __init__(self, mf, auxbasis=None):
        GWAC.__init__(self, mf, frozen=None, auxbasis=auxbasis)
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
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
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

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return
