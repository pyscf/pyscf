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
Spin-restricted quasiparticle self-consistent GW method based on the analytic continuation scheme.

References:
    J. Lei and T. Zhu, J. Chem. Phys. 157, 214114 (2022)
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    Phys. Rev. B 76, 165106 (2007)
    J. Chem. Theory. Comput. 12, 2528-2541 (2016)
"""

import h5py
import numpy as np
import scipy

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from pyscf.gw.gw_ac import GWAC, set_frozen_orbs, _mo_energy_without_core, _mo_without_core, get_sigma, \
    get_sigma_outcore
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC


def kernel(gw):
    # local variables for convenience
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc
    eta = gw.eta

    if gw.load_chkfile:
        with h5py.File(gw.chkfile, 'r') as fh5:
            gw.mo_energy = np.array(fh5['gw/mo_energy'])
            gw.mo_coeff = np.array(fh5['gw/mo_coeff'])

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
    mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    # get hcore and ovlp
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    # keep this condition for embedding calculations
    gw.hf_obj = rhf = gw.set_hf_object()
    dm_iter = rhf.make_rdm1(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)

    gw_diis = scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    # set up Fermi level
    gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        # update Lpq
        if gw.Lpq_ao is None:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(gw.mo_coeff)
        else:
            gw.Lpq = einsum('Lmn,mp,nq->Lpq', gw.Lpq_ao, gw.mo_coeff, gw.mo_coeff)

        # compute full self-energy on imaginary axis
        if gw.outcore:
            sigmaI, omega = get_sigma_outcore(
                gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, mo_occ=mo_occ_frz,
                iw_cutoff=gw.ac_iw_cutoff,
                eval_freqs=eval_freqs_with_zero, fullsigma=True
            )
        else:
            sigmaI, omega = get_sigma(
                gw, orbs_frz, gw.Lpq, quad_freqs, quad_wts, ef, mo_energy_frz, mo_occ=mo_occ_frz,
                iw_cutoff=gw.ac_iw_cutoff,
                eval_freqs=eval_freqs_with_zero, fullsigma=True
            )

        # analytic continuation
        if gw.ac == 'twopole':
            acobj = TwoPoleAC(list(range(nmo)), nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
            if gw.ac_idx is not None:
                acobj.idx = gw.ac_idx
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)

        if gw.mode == 'b':
            sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
            nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
            acobj_diag = acobj.diagonal()
            for ip, p in enumerate(orbs_frz):
                nfrz_sigma[ip, ip] = acobj_diag[ip].ac_eval(mo_energy_frz[p] + 1j * eta).real
        elif gw.mode == 'a':
            nfrz_sigma = np.zeros(shape=[len(orbs_frz), len(orbs_frz)], dtype=np.complex128)
            for ip, p in enumerate(orbs_frz):
                for iq, q in enumerate(orbs_frz):
                    nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[p] + 1j * eta)
                    nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[p] + 1j * eta).conj()
                    nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[q] + 1j * eta)
                    nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[q] + 1j * eta).conj()
        elif gw.mode == 'c':
            sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
            nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
        else:
            raise ValueError('Unknown QSGW mode %s' % gw.mode)

        sigma = np.zeros(shape=[nmo, nmo], dtype=np.complex128)
        sigma[np.ix_(orbs_frz, orbs_frz)] = nfrz_sigma

        # get static correlation self-energy in AO basis
        CS = mo_coeff_frz.T @ ovlp
        vsig = CS.T @ sigma.real @ CS

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

        # update mo_energy, mo_coeff, mo_occ and ef
        gw.mo_energy, gw.mo_coeff = scipy.linalg.eigh(ham, ovlp)
        gw.mo_occ = rhf.get_occ(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)
        gw.ef = ef = gw.get_ef()
        gw.acobj = acobj

        # update non-frozen quantities
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
        mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

        # check density matrix convergence
        dm_old = dm_iter.copy()
        # update QSGW density matrix
        dm_iter = rhf.make_rdm1(mo_coeff=gw.mo_coeff, mo_occ=gw.mo_occ)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1
        with np.printoptions(threshold=len(gw.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

        if gw.chkfile:
            gw.dump_chk()

    logger.debug(gw, 'QSGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle + 1)

    with np.printoptions(threshold=len(gw.mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    return


class QSGW(GWAC):
    def __init__(self, mf, frozen=None, auxbasis=None):
        GWAC.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        # options
        self.mode = 'b'  # mode to evaluate off-diagonal self-energy
        self.max_cycle = 30  # max qsGW cycle
        self.conv_tol = 1.0e-6  # convergence tolerance
        self.diis_space = 10  # DIIS space
        self.diis_file = None  # DIIS file name
        self.chkfile = None  # name of check file
        self.load_chkfile = False  # load check file

        # matrices
        self.Lpq_ao = None  # three-center density-fitting matrix in AO, used for impurity solver
        self.vsig = None  # static correlation self-energy in AO
        self.hf_obj = None  # Hartree-Fock object for hcore, J and K
        return

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        log.info('frozen orbitals = %s', self.frozen)
        log.info('GW density matrix = %s', self.rdm)
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
        log.info('imaginary frequency cutoff = %s', str(self.ac_iw_cutoff))
        if self.ac == 'pade':
            log.info('Pade points = %d', self.ac_pade_npts)
            log.info('Pade step ratio = %.3f', self.ac_pade_step_ratio)
        # qsGW parameters
        log.info('off-diagonal mode = %s' % self.mode)
        log.info('max cycle = %d', self.max_cycle)
        log.info('convergence tolerance = %.3e', self.conv_tol)
        log.info('DIIS space = %d', self.diis_space)
        log.info('DIIS file = %s', self.diis_file)
        log.info('')
        return

    def dump_chk(self):
        if self.chkfile:
            with h5py.File(self.chkfile, 'w') as fh5:
                fh5['gw/mo_energy'] = self.mo_energy
                fh5['gw/mo_coeff'] = self.mo_coeff
                fh5['gw/mo_occ'] = self.mo_occ
        return

    def kernel(self):
        """Run a quasiparticle self-consistent GW calculation."""
        # smeared GW needs denser grids to be accurate
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0

        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    def set_hf_object(self):
        """Set Hartree-Fock object for hcore, J and K.
        This function should be overwritten for supercell.

        Returns
        -------
        rhf : pyscf.scf.rhf.RHF
            RHF object
        """
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = self._scf.to_hf()
            if self._scf._eri is not None:
                rhf._eri = self._scf._eri
        return rhf
