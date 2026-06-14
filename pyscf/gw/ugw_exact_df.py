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

'''
Spin-unrestricted G0W0 method based on the fully analytic scheme.
This implementation has N^6 scaling, and is accurate for all states.

Reference:
    J. Chem. Theory Comput. 9, 1, 232-246 (2013)
'''

import numpy as np
import scipy
import time

from pyscf import dft, scf
from pyscf.lib import einsum, logger, temporary_env

from pyscf.gw.ugw_ac import UGWAC, get_g0
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots


def kernel(gw):
    # local variables for convenience
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mf = gw._scf
    mo_energy = gw.mo_energy
    mo_coeff = gw.mo_coeff
    mf_mo_energy = gw._scf.mo_energy

    if gw.Lpq is None:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(gw.mo_coeff)

    # mean-field exchange-correlation matrix
    with temporary_env(mf, verbose=0):
        vxc_ao = mf.get_veff()
        vj_ao = mf.get_j()
    vxc_ao[0] = vxc_ao[0] - (vj_ao[0] + vj_ao[1])
    vxc_ao[1] = vxc_ao[1] - (vj_ao[0] + vj_ao[1])
    gw.vxc = vxc = np.asarray([mo_coeff[s].T @ vxc_ao[s] @ mo_coeff[s] for s in range(2)])

    # exchange self-energy
    if gw.vhf_df is False:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
        vk_ao = uhf.get_veff(dm=dm)
        vj_ao = uhf.get_j(dm=dm)
        vk_ao[0] = vk_ao[0] - (vj_ao[0] + vj_ao[1])
        vk_ao[1] = vk_ao[1] - (vj_ao[0] + vj_ao[1])
        vk = np.zeros(shape=[2, nmo, nmo], dtype=np.double)
        for s in range(2):
            vk[s] = mo_coeff[s].T @ vk_ao[s] @ mo_coeff[s]
    else:
        vk = np.zeros(shape=[2, nmo, nmo], dtype=np.double)
        for s in range(2):
            vk[s] = -einsum('Lpi,Liq->pq', gw.Lpq[s, :, :, : nocc[s]], gw.Lpq[s, :, : nocc[s], :])
    gw.vk = vk

    # diagonalize the RPA matrix
    gw.exci, _ = exci, xpy = diagonalize_phrpa(nocc=nocc, mo_energy=mo_energy, Lpq=gw.Lpq, RPAE=gw.RPAE)

    # calculate the transition density
    gw.rho = rho = get_transition_density(nocc=nocc, xpy=xpy, Lpq=gw.Lpq)

    # calculate the self-energy
    sigma = get_sigma(
        nocc=nocc, mo_energy=mo_energy, mo_energy_prev=mo_energy, exci=exci, rho=rho, eta=gw.eta, fullsigma=False
    )

    # quasiparticle equation
    if gw.qpe_linearized is True:
        derivative = get_sigma_derivative(
            nocc=nocc, mo_energy=mo_energy, mo_energy_prev=mo_energy, exci=exci, rho=rho, eta=gw.eta
        )
        z = 1.0 / (1.0 - derivative)
        if gw.qpe_linearized_range is not None:
            np.where((z < gw.qpe_linearized_range[0]) | (z > gw.qpe_linearized_range[1]), 1.0, z)
        mo_energy = mf_mo_energy + z * (vk + sigma - vxc).diagonal(axis1=1, axis2=2)
    else:
        def quasiparticle(qp_energy):
            sigma = get_sigma(
                nocc=nocc, mo_energy=qp_energy, mo_energy_prev=mf_mo_energy, exci=exci, rho=rho, eta=gw.eta,
                fullsigma=False)
            return qp_energy - mf_mo_energy - (sigma + vk - vxc).diagonal(axis1=1, axis2=2)

        try:
            mo_energy = scipy.optimize.newton(
                quasiparticle, mf_mo_energy, tol=gw.qpe_tol * nmo * 2.0, maxiter=gw.qpe_max_iter
            )
        except RuntimeError:
            logger.warn(gw, 'quasiparticle equation fails to converge!')

    gw.mo_energy = mo_energy
    with np.printoptions(threshold=nmo):
        logger.debug(gw, '  GW mo_energy spin-up   =\n%s', mo_energy[0])
        logger.debug(gw, '  GW mo_energy spin-down =\n%s', mo_energy[1])
    logger.warn(gw, 'GW QP energies may not be sorted from min to max')

    return


def diagonalize_phrpa(nocc, mo_energy, Lpq, RPAE=False):
    r"""Diagonalize particle-hole RPA matrix.
    The phRPA equation is solved for the response function for the screening interaction,
    which is defined in equation 66-68 in doi.org/10.1021/ct300648t
    The phRPA equation is reformulated as (A-B)(A+B)|X+Y>= \omega^2|X+Y>, which is equation 15 in 10.1063/1.477483
    RPAE equations are defined as equation 2 and 3 in 10.1103/PhysRevA.85.042507

    Parameters
    ----------
    nocc : int 1d array
        the number of occupied orbitals
    mo_energy : double 2d array
        orbital energy
    Lpq : double 4d array
        density-fitting matrix in the MO space
    RPAE : bool, optional
        add exchange in response, by default False

    Returns
    -------
    w : double 1d array
        excitation energy
    v : double 2d array
        X+Y eigenvector
    """
    nspin, nmo = mo_energy.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    rpa_dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = rpa_dim[0] + rpa_dim[1] if nspin == 2 else rpa_dim[0]

    # direct ERI contribution
    A = np.zeros(shape=[full_dim, full_dim], dtype=np.double)
    for i in range(nspin):
        for j in range(nspin):
            A[i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i], j * rpa_dim[0] : j * rpa_dim[0] + rpa_dim[j]] = einsum(
                'Lia,Ljb->iajb', Lpq[i, :, : nocc[i], nocc[i] :], Lpq[j, :, : nocc[j], nocc[j] :]
            ).reshape(rpa_dim[i], rpa_dim[j])
    B = np.array(A, copy=True)
    # exchange ERI contribution
    if RPAE is True:
        for i in range(nspin):
            A[i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i], i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i]] -= einsum(
                'Lij,Lab->iajb', Lpq[i, :, : nocc[i], nocc[i] :], Lpq[i, :, : nocc[i], nocc[i] :]
            ).reshape(rpa_dim[i], rpa_dim[i])
            B[i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i], i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i]] -= einsum(
                'Lib,Lja->iajb', Lpq[i, :, : nocc[i], nocc[i] :], Lpq[i, :, : nocc[i], nocc[i] :]
            ).reshape(rpa_dim[i], rpa_dim[i])

    # orbital energy contribution
    orb_diff = []
    for i in range(nspin):
        orb_diff.append(np.asarray(mo_energy[i][None, nocc[i]:] - mo_energy[i][:nocc[i], None]).reshape(-1))
    orb_diff = np.concatenate(orb_diff, axis=0)
    np.fill_diagonal(A, A.diagonal() + orb_diff)

    # A+B and A-B matrix
    apb = A + B
    amb = A - B

    # equation 15 in doi/10.1063/1.477483, solved by LAPACK function dspgv
    w, v = scipy.linalg.eigh(apb, amb, type=3)
    w = np.sqrt(w)
    v = v.T
    # (A-B)^{-1/2} |X+Y> to |X+Y>
    for i in range(w.shape[0]):
        v[i, :] /= np.sqrt(w[i])

    return w, v


def get_transition_density(nocc, xpy, Lpq):
    """Calculate the transition density.
    Equation 85 in doi/abs/10.1021/ct300648t

    Parameters
    ----------
    nocc : int 1d array
        the number of occupied orbitals
    xpy : double 2d array
        X+Y eigenvector
    Lpq : double 4d ndarray
        density-fitting matrix in the MO space

    Returns
    -------
    rho : double 4d array
        transition density
    """
    nspin, naux, nmo, _ = Lpq.shape
    nvir = [(nmo - nocc[i]) for i in range(nspin)]
    rpa_dim = [(nocc[i] * nvir[i]) for i in range(nspin)]
    full_dim = rpa_dim[0] + rpa_dim[1] if nspin == 2 else rpa_dim[0]
    rho = np.zeros(shape=[nspin, full_dim, nmo, nmo], dtype=np.double)
    t = np.zeros(shape=[full_dim, naux], dtype=np.double)
    for i in range(nspin):
        t += np.matmul(
            xpy[:, i * rpa_dim[0] : i * rpa_dim[0] + rpa_dim[i]], Lpq[i, :, : nocc[i], nocc[i] :].reshape(naux, -1).T
        )
    for i in range(nspin):
        rho[i] = np.matmul(t, Lpq[i].reshape(naux, -1)).reshape(-1, nmo, nmo)
    return rho


def get_sigma(nocc, mo_energy, mo_energy_prev, exci, rho, eta=1.0e-5, fullsigma=False, mode='b'):
    """Get the real part of the GW correlation self-energy.
    Equation 83 in doi.org/10.1021/ct300648t
    mode 'a' and 'b' correspond to equation 10 and 11 in doi.org/10.1103/PhysRevB.76.165106

    Parameters
    ----------
    nocc : int 1d array
        the number of occupied orbitals
    mo_energy : double 2d array
        orbital energy
    mo_energy_prev : double 2d array
        orbital energy in previous iteration
    exci : double 1d array
        phRPA excitation energy
    rho : double 4d array
        transition density
    eta : double, optional
        broadening parameter, by default 1.0e-5
    fullsigma : bool, optional
        calculate off-diagonal elements, by default False
    mode : str, optional
        mode for off-diagonal elements, by default "b"

    Returns
    -------
    sigma : double 3d array
        real part of the GW static correlation self-energy
    """
    nspin, nmo = mo_energy.shape
    eta2 = np.square(3.0 * eta)
    homo = max(mo_energy[0][nocc[0] - 1], mo_energy[1][nocc[1] - 1])
    lumo = min(mo_energy[0][nocc[0]], mo_energy[1][nocc[1]])
    ef = (homo + lumo) / 2.0
    sigma = np.zeros(shape=[nspin, nmo, nmo], dtype=np.double)
    if fullsigma is False:
        for s in range(nspin):
            energy_occ = mo_energy[s][:, None, None] - mo_energy_prev[s][None, : nocc[s], None] + exci[None, None, :]
            energy_vir = mo_energy[s][:, None, None] - mo_energy_prev[s][None, nocc[s] :, None] - exci[None, None, :]
            energy = np.concatenate([energy_occ, energy_vir], axis=1)
            energy = energy / (np.square(energy) + eta2)
            sigma[s][np.diag_indices(nmo)] = einsum('mpr,prm->p', np.square(rho[s]), energy)
    else:
        assert mode in ['a', 'b']
        for s in range(nspin):
            if mode == 'a':
                # mode A : off-diagonal evaluated as average, equation 11 in doi.org/10.1103/PhysRevB.76.165106
                energy_occ = mo_energy[s, :, None, None] - mo_energy_prev[s, None, :nocc[s], None] + exci[None, None, :]
                energy_vir = mo_energy[s, :, None, None] - mo_energy_prev[s, None, nocc[s]:, None] - exci[None, None, :]
                energy = np.concatenate([energy_occ, energy_vir], axis=1)
                energy = energy / (np.square(energy) + eta2)
                sigma[s] = einsum('mpr,mqr,prm->pq', rho[s], rho[s], energy)
                sigma[s] += einsum('mpr,mqr,qrm->pq', rho[s], rho[s], energy)
                sigma[s] *= 0.5
            elif mode == 'b':
                # mode B: off-diagonal evaluated at Fermi-level, equation 10 in doi.org/10.1103/PhysRevB.76.165106
                energy_occ = ef - mo_energy_prev[s][:nocc[s], None] + exci[None, :]
                energy_vir = ef - mo_energy_prev[s][nocc[s]:, None] - exci[None, :]
                energy = np.concatenate([energy_occ, energy_vir])
                energy = energy / (np.square(energy) + eta2)
                sigma[s] = einsum('mpr,mqr,rm->pq', rho[s], rho[s], energy)

                energy_occ = mo_energy[s, :, None, None] - mo_energy_prev[s, None, :nocc[s], None] + exci[None, None, :]
                energy_vir = mo_energy[s, :, None, None] - mo_energy_prev[s, None, nocc[s]:, None] - exci[None, None, :]
                energy = np.concatenate([energy_occ, energy_vir], axis=1)
                energy = energy / (np.square(energy) + eta2)
                sigma_diag = einsum('mpr,prm->p', np.square(rho[s]), energy)
                np.fill_diagonal(sigma[s], sigma_diag)

    return sigma


def get_sigma_derivative(nocc, mo_energy, mo_energy_prev, exci, rho, eta=1.0e-5):
    """Get the first-order derivative of the self-energy to the frequency.
    Equation.84 in doi.org/10.1021/ct300648t

    Parameters
    ----------
    nocc : int 1d array
        the number of occupied orbitals
    mo_energy : double 2d ndarray
        orbital energy
    mo_energy_prev : double 2d array
        orbital energy in previous iteration
    exci : double 1d array
        phRPA excitation energy
    rho : double 4d array
        transition density
    eta : double, optional
        broadening parameter, by default 1.0e-5

    Returns
    -------
    derivative : double 2d ndarray
        first-order derivative of the correlation self-energy to the frequency
    """
    nspin, nmo = mo_energy.shape
    eta2 = np.square(3.0 * eta)
    derivative = np.zeros(shape=[nspin, nmo], dtype=np.double)
    for s in range(nspin):
        # occupied part
        energy = mo_energy[s][:, None, None] - mo_energy_prev[s][None, : nocc[s], None] + exci[None, None, :]
        energy = np.square(energy)
        energy = (eta2 - energy) / (energy + eta2) ** 2
        derivative[s] += einsum('mpi,pim->p', np.square(rho[s][:, :, : nocc[s]]), energy)

        # virtual part
        energy = mo_energy[s][:, None, None] - mo_energy_prev[s][None, nocc[s] :, None] - exci[None, None, :]
        energy = np.square(energy)
        energy = (eta2 - energy) / (energy + eta2) ** 2
        derivative[s] += einsum('mpa,pam->p', np.square(rho[s][:, :, nocc[s] :]), energy)

    return derivative


def make_gf(gw, omega, eta, fullsigma=True, mode='linear'):
    """Get exact dynamical Green's function and self-energy.

    Dynamical self-energy is evaluated as equation 78 in doi.org/10.1021/ct300648t

    Two modes for solving Dyson equation
    "dyson" for using inverse Dyson equation.
    "linear" for G = G0 + G0 Sigma G0, as equation 16 in doi.org/10.1021/acs.jctc.0c01264

    Parameters
    ----------
    gw : UGWExact
        gw object, provides attributes: nocc, nmo, _scf.mo_energy, exci, rho, vk, vxc
    omega : double or complex array
        frequency grids
    eta : double
        broadening parameter
    fullsigma : bool, optional
        calculate off-diagonal elements, by default True
    mode : str, optional
        mode for Dyson equation, 'linear' or 'dyson', by default 'linear'

    Returns
    -------
    gf : complex 4d array
        GW Green's function
    gf0 : complex 4d array
        non-interacting Green's function
    sigma : complex 4d array
        self-energy
    """
    nmo = gw.nmo[0]
    nocc = gw.nocc
    exci = gw.exci
    mo_energy = np.asarray(gw._scf.mo_energy)

    # get self-energy
    # NOTE: this is for G0W0 Green's function, so mo_energy is from the SCF calculation
    sigma = np.zeros(shape=[2, nmo, nmo, len(omega)], dtype=np.complex128)
    for s in range(2):
        energy_occ = omega[:, None, None] - mo_energy[s, None, : nocc[s], None] + (exci[None, None, :] - 3.0 * 1j * eta)
        energy_vir = omega[:, None, None] - mo_energy[s, None, nocc[s] :, None] - (exci[None, None, :] - 3.0 * 1j * eta)
        energy = np.concatenate([energy_occ, energy_vir], axis=1)
        energy = 1.0 / energy

        if fullsigma is False:
            sigma_diag = einsum('mpr,wrm->pw', np.square(gw.rho[s]), energy)
            for iw in range(len(omega)):
                sigma[s, ..., iw] = np.diag(sigma_diag[:, iw])
        else:
            sigma[s] = einsum('mpr,mqr,wrm->pqw', gw.rho[s], gw.rho[s], energy)

    # Dyson equation for Green's function
    gf0 = get_g0(omega=omega, mo_energy=gw._scf.mo_energy, eta=eta)

    gf = np.zeros_like(gf0)
    sigma_diff = np.array(sigma, copy=True)
    if fullsigma is True:
        for iw in range(len(omega)):
            sigma_diff[:, :, :, iw] += gw.vk - gw.vxc
    else:
        for s in range(2):
            for iw in range(len(omega)):
                for i in range(len(gw._scf.mo_energy)):
                    sigma_diff[s, i, i, iw] += gw.vk[s, i, i] - gw.vxc[s, i, i]

    for s in range(2):
        for iw in range(len(omega)):
            if mode == 'linear':
                gf[s, :, :, iw] = gf0[s, :, :, iw] + gf0[s, :, :, iw] @ sigma_diff[s, :, :, iw] @ gf0[s, :, :, iw]
            elif mode == 'dyson':
                gf[s, :, :, iw] = np.linalg.inv(np.linalg.inv(gf0[s, :, :, iw]) - sigma_diff[s, :, :, iw])

    return gf, gf0, sigma


def make_diag_dos(gw, omega, eta):
    """Get density of states using diagonal self-energy.
    Equation 7 in doi.org/10.1021/acs.jctc.2c00617

    Parameters
    ----------
    gw : UGWExact
        gw object, provides attributes: nocc, nmo, _scf.mo_energy, exci, rho, vk, vxc
    omega : complex 1d array
        frequency grids
    eta : double
        broadening parameter

    Returns
    -------
    dos : double 3d array
        orbital-resolved density of states
    """
    nmo = gw.nmo[0]
    nocc = gw.nocc
    mo_energy = np.asarray(gw._scf.mo_energy)
    eta2 = np.square(3.0 * eta)
    sigma_real = np.zeros(shape=[2, nmo, len(omega)], dtype=np.double)
    sigma_imag = np.zeros_like(sigma_real)
    for s in range(2):
        energy_occ = omega[:, None, None] - mo_energy[s, None, : nocc[s], None] + gw.exci[None, None, :]
        energy_vir = omega[:, None, None] - mo_energy[s, None, nocc[s] :, None] - gw.exci[None, None, :]
        energy = np.concatenate([energy_occ, energy_vir], axis=1)
        energy_real = energy / (np.square(energy) + eta2)
        energy_imag = eta / (np.square(energy) + eta2)
        energy_imag[:, nocc[s] :, :] *= -1.0
        sigma_real[s] = einsum('mpr,wrm->pw', np.square(gw.rho[s]), energy_real) * 2.0
        sigma_imag[s] = einsum('mpr,wrm->pw', np.square(gw.rho[s]), energy_imag) * 2.0

    vk_minus_vxc = (gw.vk - gw.vxc).diagonal(axis1=1, axis2=2)
    ereal = omega[None, None, :] - mo_energy[:, :, None] - (sigma_real + vk_minus_vxc[:, :, None])
    dos = abs(sigma_imag) / ereal**2 + sigma_imag**2
    dos /= np.pi

    return dos


class UGWExactDF(UGWAC):
    def __init__(self, mf, auxbasis=None):
        UGWAC.__init__(self, mf, frozen=None, auxbasis=auxbasis)

        # options
        self.RPAE = False  # exchange in RPA response

        # matrices
        self.exci = None  # RPA excitation energy
        self.rho = None  # transition density

        return

    def dump_flags(self, verbose=None):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('GW nmo = %s', self.nmo[0])
        log.info('GW nocc = %s, nvir = %s', self.nocc, (self.nmo[s] - self.nocc[s] for s in range(2)))
        log.info('broadening parameter = %.3e', self.eta)
        log.info('RPA exchange response = %s', self.RPAE)
        log.info('use perturbative linearized QP eqn = %s', self.qpe_linearized)
        if self.qpe_linearized is True:
            log.info('linearized factor range = %s', self.qpe_linearized_range)
        else:
            log.info('QPE max iter = %d', self.qpe_max_iter)
            log.info('QPE tolerance = %.1e', self.qpe_tol)
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

        self.dump_flags()
        cput0 = (time.process_time(), time.perf_counter())
        kernel(self)
        logger.timer(self, 'GW', *cput0)
        return

    make_gf = make_gf
    make_diag_dos = make_diag_dos

    def make_rdm1(self, nw=60, mode='linear', ao_repr=False):
        r"""Get GW one-particle density matrix.
        rdm1 = G(it=0) = \int dw G(iw)

        Parameters
        ----------
        nw : int, optional
            number for imaginary frequency grids for integration, by default 60
        mode : str, optional
            mode for Dyson equation, by default "linear"
        ao_repr : bool, optional
            return density matrix in AO, by default False

        Returns
        -------
        rdm1 : double ndarray
            GW one-particle density matrix
        """
        freqs, wts = _get_scaled_legendre_roots(nw)
        homo = max(self._scf.mo_energy[0][self.nocc[0] - 1], self._scf.mo_energy[1][self.nocc[1] - 1])
        lumo = min(self._scf.mo_energy[0][self.nocc[0]], self._scf.mo_energy[1][self.nocc[1]])
        ef = (homo + lumo) / 2.0
        omega = 1j * freqs + ef
        gf = self.make_gf(omega=omega, eta=0, fullsigma=True, mode=mode)[0]

        rdm1 = 1.0 / np.pi * einsum('sijw,w->sij', gf, wts).real + np.eye(self.nmo[0])[None, :, :] * 0.5

        # symmetrize density matrix
        for s in range(2):
            rdm1[s] = 0.5 * (rdm1[s] + rdm1[s].T)
        nelec = np.trace(rdm1, axis1=1, axis2=2)
        logger.info(self, 'GW particle number up = %s, dn = %s, total = %s', nelec[0], nelec[1], nelec[0] + nelec[1])

        if ao_repr is True:
            for s in range(2):
                rdm1[s] = self._scf.mo_coeff[s] @ rdm1[s] @ self._scf.mo_coeff[s].T

        return rdm1

    def energy_tot(self, nw=60):
        """Calculate GW total energy using Galitskii-Migdal formula.
        V. M. Galitskii and A. B. Migdal, Zh. E ksp. Teor. Fiz. 34, 139~1958! @Sov. Phys. JETP 139, 96 ~1958!#
        Working equation: equation A5 in Phys. Rev. B 88, 075105

        Parameters
        ----------
        nw : int, optional
            number for imaginary frequency grids for integration, by default 60

        Returns
        -------
        e_tot : double
            GW total energy
        e_hf : double
            HF total energy
        e_c : double
            GW correlation energy
        """
        # get correlation self-energy on the imaginary axis
        freqs, wts = _get_scaled_legendre_roots(nw)
        omega = 1j * freqs
        # mode does not matter because gf is not used
        _, gf0, sigma = self.make_gf(omega=omega, eta=0, fullsigma=True)

        # GW correlation energy
        g0_sigma_target = 1.0 / 2.0 / np.pi * einsum('sijw,sjiw,w->', gf0, sigma, wts)
        e_c = g0_sigma_target.real

        dm = self._scf.make_rdm1()
        # keep this condition for embedding calculations
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(self._scf, scf.hf.RHF):
            uhf = self._scf
        else:
            uhf = scf.UHF(self.mol)
        with temporary_env(uhf, verbose=0):
            e_hf = uhf.energy_elec(dm=dm)[0]
            e_hf += self._scf.energy_nuc()
        e_tot = e_hf + e_c

        logger.info(self, 'HF energy@GW density  = %.8f', e_hf)
        logger.info(self, 'GW correlation energy = %.8f', e_c)
        logger.info(self, 'GW total energy       = %.8f', e_tot)

        return e_tot, e_hf, e_c

