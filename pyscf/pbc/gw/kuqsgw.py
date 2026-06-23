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
Periodic spin-unrestricted quasiparticle self-consistent GW method based on the analytic continuation scheme.

References:
    J. Lei and T. Zhu, J. Chem. Phys. 157, 214114 (2022)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import h5py
import numpy as np
import scipy

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.pbc import df, scf
from pyscf.pbc.mp.kump2 import get_frozen_mask
from pyscf import scf as mol_scf

from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC
from pyscf.pbc.gw.kugw_ac import _mo_frozen, _mo_energy_frozen, _mo_occ_frozen, get_sigma, get_ef, set_frozen_orbs
from pyscf.pbc.gw.krqsgw import KRQSGW


def kernel(gw):
    mf = gw._scf
    eta = gw.eta
    nkpts = gw.nkpts
    nmo = gw.nmo[0]
    nocca_full = int(np.sum(mf.mo_occ[0][0]))
    noccb_full = int(np.sum(mf.mo_occ[1][0]))
    nocc_full = [nocca_full, noccb_full]

    # set frozen orbital list
    gw.set_frozen_orbs()
    orbs_frz = gw.orbs_frz

    if gw.load_chkfile:
        logger.info(gw, "Load chkfile from %s, QSGW previous cycle: ", gw.chkfile)
        with h5py.File(gw.chkfile, 'r') as fh5:
            gw.mo_energy = mo_energy_full = np.array(fh5['gw/mo_energy'])
            gw.mo_coeff = mo_coeff_full = np.array(fh5['gw/mo_coeff'])
            gw.mo_occ = mo_occ_full = np.array(fh5['gw/mo_occ'])
    else:
        mo_energy_full = np.array(gw.mo_energy, copy=True)
        mo_coeff_full = np.array(gw.mo_coeff, copy=True)
        mo_occ_full = np.array(gw.mo_occ, copy=True)

    mo_energy_frz = _mo_energy_frozen(gw, mo_energy_full)
    mo_coeff_frz = _mo_frozen(gw, mo_coeff_full)
    mo_occ_frz = _mo_occ_frozen(gw, mo_occ_full)

    # grids for integration on imaginary axis
    gw.freqs, gw.wts = freqs, wts = _get_scaled_legendre_roots(gw.nw)

    # set up Fermi level
    ef = gw.ef = get_ef(kmf=mf, mo_energy=mo_energy_full)

    # get hcore and ovlp
    with temporary_env(mf.mol, verbose=0):
        hcore = mf.get_hcore()
        ovlp = mf.get_ovlp()

    # set up UHF object
    if isinstance(mf.with_df, df.GDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).density_fit()
    elif isinstance(mf.with_df, df.RSDF):
        uhf = scf.KUHF(gw.mol.copy(deep=True), gw.kpts, exxdiv=None).rs_density_fit()
    if hasattr(mf, 'sigma'):
        uhf = scf.addons.smearing_(uhf, sigma=mf.sigma, method=mf.smearing_method)
    uhf.with_df = gw.with_df
    uhf.mo_energy = np.array(mo_energy_full, copy=True)
    uhf.mo_coeff = np.array(mo_coeff_full, copy=True)
    uhf.mo_occ = np.array(mo_occ_full, copy=True)
    uhf.verbose = uhf.mol.verbose = 0
    dm_iter = np.array(uhf.make_rdm1(), copy=True)
    dm_old = dm_iter.copy()

    # initialize DIIS
    gw_diis = mol_scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    cycle = 0
    while gw.converged is False and cycle < max(1, gw.max_cycle):
        # calculate self-energy on imaginary axis
        gw.sigmaI, gw.omega = sigmaI, omega = get_sigma(
            gw, freqs, wts, ef=ef, mo_energy=mo_energy_frz, orbs=orbs_frz, mo_coeff=mo_coeff_frz, mo_occ=mo_occ_frz,
            iw_cutoff=gw.ac_iw_cutoff, fullsigma=True)

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

        # real-axis sigma
        mode = gw.mode.strip().lower()
        sigma = np.zeros(shape=[2, nkpts, nmo, nmo], dtype=np.complex128)
        for s in range(2):
            for k in range(nkpts):
                for ip, p in enumerate(orbs_frz):
                    for iq, q in enumerate(orbs_frz):
                        if mode == 'b':
                            if p == q:
                                sigma[s, k, p, q] += 0.5 * acobj[s, k, ip, iq].ac_eval(
                                    mo_energy_frz[s][k][p] + 1j * eta
                                )
                                sigma[s, k, p, q] += 0.5 * acobj[s, k, ip, iq].ac_eval(
                                    mo_energy_frz[s][k][p] + 1j * eta
                                ).conj()
                            else:
                                sigma[s, k, p, q] += 0.5 * acobj[s, k, ip, iq].ac_eval(ef + 1j * eta)
                                sigma[s, k, p, q] += 0.5 * acobj[s, k, iq, ip].ac_eval(ef + 1j * eta).conj()
                        elif mode == 'a':
                            sigma[s, k, p, q] += 0.25 * acobj[s, k, ip, iq].ac_eval(
                                mo_energy_frz[s][k][p] + 1j * eta
                            )
                            sigma[s, k, p, q] += 0.25 * acobj[s, k, iq, ip].ac_eval(
                                mo_energy_frz[s][k][p] + 1j * eta
                            ).conj()
                            sigma[s, k, p, q] += 0.25 * acobj[s, k, ip, iq].ac_eval(
                                mo_energy_frz[s][k][q] + 1j * eta
                            )
                            sigma[s, k, p, q] += 0.25 * acobj[s, k, iq, ip].ac_eval(
                                mo_energy_frz[s][k][q] + 1j * eta
                            ).conj()
                        elif mode == 'c':
                            sigma[s, k, p, q] += 0.5 * acobj[s, k, ip, iq].ac_eval(ef + 1j * eta)
                            sigma[s, k, p, q] += 0.5 * acobj[s, k, iq, ip].ac_eval(ef + 1j * eta).conj()
                        else:
                            raise ValueError('Unknown QSGW mode %s' % gw.mode)

        # obtain static correlation self-energy in AO basis
        vsig = np.zeros_like(dm_iter, dtype=np.result_type(dm_iter, sigma))
        for s in range(2):
            for k in range(nkpts):
                CS = np.matmul(mo_coeff_frz[s][k].T.conj(), ovlp[k])
                vsig[s][k] = CS.T.conj() @ sigma[s, k] @ CS
        gw.vsig = vsig

        # update veff
        with temporary_env(uhf, verbose=0), temporary_env(uhf.with_df, verbose=0):
            veff = uhf.get_veff(dm_kpts=dm_iter)

        # finite size correction for exchange self-energy
        if gw.fc:
            vk_corr = -2.0 / np.pi * (6.0 * np.pi**2 / gw.mol.vol / nkpts) ** (1.0 / 3.0)
            for s in range(2):
                for k in range(nkpts):
                    veff[s][k] = mo_coeff_full[s][k].T.conj() @ veff[s][k] @ mo_coeff_full[s][k]

            for s in range(2):
                for k in range(nkpts):
                    for i in range(nocc_full[s]):
                        veff[s][k][i, i] = veff[s][k][i, i] + vk_corr

            for s in range(2):
                for k in range(nkpts):
                    CS = np.matmul(mo_coeff_full[s][k].T.conj(), ovlp[k])
                    veff[s][k] = CS.T.conj() @ veff[s][k] @ CS
        gw.veff = veff

        # complete Hamiltonian through DIIS
        ham = hcore + veff + vsig
        mo_energy_full = np.zeros_like(mf.mo_energy)
        mo_coeff_full = np.zeros_like(mf.mo_coeff)
        if cycle >= gw.diis_start_cycle:
            ham = gw_diis.update(ovlp, dm_iter, ham)
        # diagonalize
        for s in range(2):
            for k in range(nkpts):
                mo_energy_full[s][k], mo_coeff_full[s][k] = scipy.linalg.eigh(ham[s][k], ovlp[k])

        # update QSGW mean-field object
        uhf.mo_energy = np.array(mo_energy_full, copy=True)
        uhf.mo_coeff = np.array(mo_coeff_full, copy=True)
        mo_occ_full = uhf.get_occ(mo_energy_kpts=mo_energy_full, mo_coeff_kpts=mo_coeff_full)
        uhf.mo_occ = np.array(mo_occ_full, copy=True)

        # update attributes in the GW object
        gw.mo_energy = np.array(mo_energy_full, copy=True)
        gw.mo_coeff = np.array(mo_coeff_full, copy=True)
        gw.mo_occ = np.array(mo_occ_full, copy=True)
        ef = gw.ef = get_ef(mf, mo_energy=mo_energy_full)
        gw.acobj = acobj

        # update NON-FROZEN quantities
        mo_energy_frz = _mo_energy_frozen(gw, mo_energy_full)
        mo_coeff_frz = _mo_frozen(gw, mo_coeff_full)
        mo_occ_frz = _mo_occ_frozen(gw, mo_occ_full)

        # check density matrix convergence
        if cycle == 0:
            dm_old2 = dm_iter
        else:
            dm_old2 = dm_old
        dm_old = dm_iter
        # density matrix from updated QSGW density
        dm_iter = uhf.make_rdm1()
        norm_dm = np.linalg.norm(dm_iter - dm_old) / (nmo * nkpts * 2)
        norm_dm2 = np.linalg.norm(dm_old - dm_old2) / (nmo * nkpts * 2)
        if norm_dm < gw.conv_tol and norm_dm2 < gw.conv_tol and cycle > 0:
            gw.converged = True

        logger.info(gw, 'UQSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1

        if gw.chkfile:
            gw.dump_chk()

    if gw.writefile > 0:
        with temporary_env(mf, verbose=0), temporary_env(mf.mol, verbose=0), temporary_env(mf.with_df, verbose=0):
            veff_mf = mf.get_veff()

        with h5py.File('vxc.h5', 'w') as feri:
            feri['hcore'] = np.asarray(hcore)
            feri['veff_mf'] = np.asarray(veff_mf)
            feri['vhf_qp'] = np.asarray(gw.veff)
            feri['vc_qp'] = np.asarray(gw.vsig)
            feri['qp_energy'] = np.asarray(gw.mo_energy)
            feri['qp_coeff'] = np.asarray(gw.mo_coeff)
            feri['qp_occ'] = np.asarray(gw.mo_occ)

        acobj.save('ac_coeff.h5')

    return


class KUQSGW(KRQSGW):
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
        log.info('GW density matrix = %s', self.rdm)
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
        # qsGW settings
        log.info('off-diagonal self-energy mode = %s', self.mode)
        log.info('max cycle = %d', self.max_cycle)
        log.info('density matrix convergence tolerance = %.2e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS start cycle = %d', self.diis_start_cycle)
        log.info('DIIS file = %s', self.diis_file)
        log.info('load chkfile = %s', self.load_chkfile)
        if self.load_chkfile:
            log.info('chkfile path = %s', self.chkfile)
        log.info('')
        return

    def _finalize(self):
        """Hook for dumping results and clearing up the object."""
        if self.converged:
            logger.note(self, 'UQSGW converged.')
        else:
            logger.note(self, 'UQSGW not converged.')
        return

    @property
    def nocc(self):
        frozen_mask = get_frozen_mask(self)
        nkpts = len(self._scf.mo_energy[0])
        neleca = 0.0
        nelecb = 0.0
        for k in range(nkpts):
            neleca += np.sum(self._scf.mo_occ[0][k][frozen_mask[0][k]])
            nelecb += np.sum(self._scf.mo_occ[1][k][frozen_mask[1][k]])
        neleca = int(neleca / nkpts)
        nelecb = int(nelecb / nkpts)
        return (neleca, nelecb)

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        frozen_mask = get_frozen_mask(self)
        nmoa = len(self._scf.mo_energy[0][0][frozen_mask[0][0]])
        nmob = len(self._scf.mo_energy[1][0][frozen_mask[1][0]])
        return (nmoa, nmob)

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def kernel(self):
        """Run qsGW calculation."""
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

        if hasattr(self._scf, 'sigma'):
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
            self.fc = False

        nmo = self.nmo[0]
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (4 * nkpts * nmo**2 * naux) * 16 / 1e6
        mem_now = lib.current_memory()[0]
        if mem_incore + mem_now > 0.99 * self.max_memory:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'KUQSGW', *cput0)
        self._finalize()
        return

    set_frozen_orbs = set_frozen_orbs
