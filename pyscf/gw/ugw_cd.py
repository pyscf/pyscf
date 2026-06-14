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
#

"""
Spin-unrestricted G0W0 method based the contour deformation scheme.
The scaling N^4 for valence and N^5 for core states.
GW-CD is particularly recommended for accurate core and high-energy states.

References:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    J. Chem. Theory Comput. 14, 4856-4869 (2018)
"""

from functools import reduce
import time
import numpy as np
import h5py
from scipy.optimize import newton
import scipy.linalg as sla

from pyscf import lib
from pyscf.lib import temporary_env
from pyscf.lib import logger
from pyscf import scf, dft

from pyscf.gw.ugw_ac import UGWAC, get_rho_response, set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.gw.utils.gw_np_helper import mkslice, get_id_minus_pi, get_id_minus_pi_inv_minus_id

einsum = lib.einsum


def kernel(gw, load_chk=None):
    """UGWCD kernel.

    Parameters
    ----------
    gw : UGWCD
        instance of UGWCD class
    load_chk : str, optional
        name of chkfile to load

    Returns
    -------
    tuple(bool, ndarray, ndarray)
        conv : bool
            True if converged
        conv_inds : ndarray
            conv_inds[p] = 1 if quasiparticle energy for orbital p is converged
        mo_energy : ndarray
            quasiparticle energies
    """
    # local variables for convenience
    mf = gw._scf
    nocc = gw.nocc

    # set frozen orbitals
    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    # get non-frozen quantities
    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and gw.outcore is False:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq

    nmo = gw.nmo
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    conv_inds = np.zeros(shape=mo_energy.shape, dtype=int)
    if load_chk is not None:
        with h5py.File(load_chk, 'r') as f:
            g = f['gwcd']
            mo_energy = g['mo_energy'][()]
            conv_inds = g['conv_inds'][()]
        if mo_energy.shape != (2, nmo):
            raise RuntimeError(f'{load_chk}: mo_energy shape mismatch')
        if conv_inds.shape != mo_energy.shape:
            raise RuntimeError(f'{load_chk}: conv_inds shape mismatch')

    chkfile = gw.chkfile
    if chkfile is not None:
        with h5py.File(chkfile, 'a') as f:
            if 'gwcd' in f:
                del f['gwcd']
            g = f.create_group('gwcd')
            g.create_dataset('mo_energy', data=mo_energy)
            g.create_dataset('conv_inds', data=conv_inds)

    # mean-field exchange-correlation
    v_mf_ao = mf.get_veff()
    vj_ao = mf.get_j()
    v_mf_ao[0] = v_mf_ao[0] - (vj_ao[0] + vj_ao[1])
    v_mf_ao[1] = v_mf_ao[1] - (vj_ao[0] + vj_ao[1])
    v_mf = np.asarray([reduce(np.matmul, (mo_coeff_frz[s].T, v_mf_ao[s], mo_coeff_frz[s])) for s in range(2)])
    gw.vxc = v_mf

    # exchange self-energy
    if gw.vhf_df is True and gw.outcore is False:
        vk = np.asarray([einsum('Lpi,Liq->pq', Lpq[s, :, :, : nocc[s]], Lpq[s, :, : nocc[s], :]) for s in range(2)])
    else:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.uks.UKS)) and isinstance(mf, scf.uhf.UHF):
            uhf = mf
        else:
            uhf = scf.UHF(gw.mol)
            if hasattr(gw._scf, 'sigma'):
                uhf = scf.addons.smearing_(uhf, sigma=gw._scf.sigma, method=gw._scf.smearing_method)
        vk_ao = uhf.get_veff(dm=dm)
        vj_ao = uhf.get_j(dm=dm)
        vk_ao[0] = vk_ao[0] - (vj_ao[0] + vj_ao[1])
        vk_ao[1] = vk_ao[1] - (vj_ao[0] + vj_ao[1])
        vk = np.asarray([reduce(np.matmul, (mo_coeff_frz[s].T, vk_ao[s], mo_coeff_frz[s])) for s in range(2)])
    gw.vk = vk

    # set up Fermi level
    gw.ef = gw.get_ef(mo_energy=mf.mo_energy)

    # grids for integration on imaginary axis
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)

    nocca, noccb = nocc

    # Compute Wmn(iw) on imaginary axis
    logger.debug(gw, 'Computing the imaginary part')
    Lia_a = np.ascontiguousarray(Lpq[0, :, :nocca, nocca:])
    Lia_b = np.ascontiguousarray(Lpq[1, :, :noccb, noccb:])

    Wmn = get_WmnI_diag(gw, orbs, Lpq[0], Lpq[1], Lia_a, Lia_b, quad_freqs)

    mo_energy = np.zeros_like(np.asarray(mf.mo_energy))
    conv_inds = np.zeros(shape=mo_energy.shape, dtype=int)

    conv = True
    for s in range(2):
        for p_in_frz, p_in_all in zip(orbs_frz, orbs):
            if gw.qpe_linearized:
                # FIXME
                logger.warn(gw, 'linearization with CD leads to wrong quasiparticle energy')
                raise NotImplementedError
            else:

                def quasiparticle(omega):
                    sigma = get_sigma_diag(
                        gw.ef,
                        omega,
                        s,
                        p_in_frz,
                        mo_energy_frz,
                        Lpq[0],
                        Lpq[1],
                        Lia_a,
                        Lia_b,
                        Wmn[s][:, p_in_frz],
                        quad_freqs,
                        quad_wts,
                        gw.eta,
                    ).real
                    return (
                        omega
                        - gw._scf.mo_energy[s][p_in_all]
                        - (sigma.real + vk[s, p_in_all, p_in_all] - v_mf[s, p_in_all, p_in_all])
                    )

                try:
                    if p_in_frz < nocc[s]:
                        delta = -1e-2
                    else:
                        delta = 1e-2
                    this_omega = gw._scf.mo_energy[s][p_in_frz] + delta
                    e, result = newton(
                        quasiparticle, this_omega, tol=gw.qpe_tol, maxiter=gw.qpe_max_iter, full_output=True
                    )
                    logger.debug(gw, f'QP energy for orb {p_in_all}: {e}')
                    logger.debug(gw, f'Number of iterations: {result.iterations}')
                    mo_energy[s][p_in_all] = e
                except RuntimeError:
                    conv = False

    if gw.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=1000)
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
        np.set_printoptions(threshold=1000)

    return conv, conv_inds, mo_energy


def get_sigma_diag(ef, ep, s, p, mo_energy, Lpqa, Lpqb, Lia_a, Lia_b, Wmn_s, freqs, wts, eta):
    """Compute self-energy on real axis using contour deformation.

    Parameters
    ----------
    ef : float
        Fermi level
    ep : float
        quasiparticle energy
    s : int
        spin index
    p : int
        orbital index
    mo_energy : ndarray
        orbital energies
    Lpqa : ndarray
        Lpq for alpha spin
    Lpqb : ndarray
        Lpq for beta spin
    Lia_a : ndarray
        Lpq for alpha spin, occupied-virtual block
    Lia_b : ndarray
        Lpq for beta spin, occupied-virtual block
    Wmn_s : ndarray
        Wmn on imaginary axis for spin s, shape (nw, norbs, nmo)
    freqs : ndarray
        frequencies on imaginary axis
    wts : ndarray
        quadrature weights on imaginary axis
    eta : float
        broadening parameter for poles on real axis

    Returns
    -------
    complex
        self-energy for orbital p and spin s at energy ep
    """
    sign = np.sign(ef - mo_energy)

    emo = ep - 1j * eta * sign - mo_energy
    g0 = wts[None, :] * emo[s][:, None] / ((emo[s] ** 2)[:, None] + (freqs**2)[None, :])
    sigmaI = -einsum('mw,wm', g0, Wmn_s) / np.pi

    sigmaR = get_sigmaR_diag(mo_energy, ep, s, p, ef, Lpqa, Lpqb, Lia_a, Lia_b, eta)
    return sigmaI + sigmaR


def get_sigmaR_diag(mo_energy, omega, s, orbp, ef, Lpqa, Lpqb, Lia_a, Lia_b, eta):
    """Compute self-energy for poles inside contour
    (more and more expensive away from Fermi surface).

    Parameters
    ----------
    mo_energy : ndarray
        orbital energies
    omega : float
        quasiparticle energy
    s : int
        spin index
    orbp : int
        orbital index
    ef : float
        Fermi level
    Lpqa : ndarray
        Lpq for alpha spin
    Lpqb : ndarray
        Lpq for beta spin
    Lia_a : ndarray
        Lpq for alpha spin, occupied-virtual block
    Lia_b : ndarray
        Lpq for beta spin, occupied-virtual block
    eta : float
        broadening parameter for poles on real axis

    Returns
    -------
    complex
        self-energy for orbital orbp and spin s at energy omega
    """
    Lpq = [Lpqa, Lpqb]

    if omega > ef:
        fm = 1.0
        idx = np.where((mo_energy[s] < omega) & (mo_energy[s] > ef))[0]
    else:
        fm = -1.0
        idx = np.where((mo_energy[s] > omega) & (mo_energy[s] < ef))[0]

    nocca, noccb = Lia_a.shape[1], Lia_b.shape[1]
    eia_a = mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:]
    eia_b = mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:]

    sigmaR = 0j
    if len(idx) > 0:
        for m in idx:
            em = mo_energy[s][m] - omega
            Pi = get_rho_response_R(eia_a, eia_b, abs(em), Lia_a, Lia_b, eta)
            Pi = get_id_minus_pi(Pi)
            vec = sla.solve(Pi.T, Lpq[s][:, orbp, m], check_finite=False, overwrite_a=True, assume_a='sym')
            vec -= Lpq[s][:, orbp, m]
            sigmaR += fm * np.dot(Lpq[s][:, m, orbp], vec)
    return sigmaR


def get_WmnI_diag(gw, orbs, Lpqa, Lpqb, Lia_a, Lia_b, freqs):
    """Compute W_mn(iw) on imaginary axis grids.

    Parameters
    ----------
    gw : UGWCD
        GW object
    orbs : list
        orbital indices
    Lpqa : ndarray
        Lpq for alpha spin
    Lpqb : ndarray
        Lpq for beta spin
    Lia_a : ndarray
        Lpq for alpha spin, occupied-virtual block
    Lia_b : ndarray
        Lpq for beta spin, occupied-virtual block
    freqs : ndarray
        frequencies on imaginary axis

    Returns
    -------
    Wmn : ndarray
        Wmn on imaginary axis, shape (s, Nmo, Norbs, Nw)
    """
    mo_energy = gw._scf.mo_energy
    nmo = gw.nmo
    nmoa, nmob = nmo
    nw = len(freqs)
    naux = Lpqa.shape[0]
    l_slice = [Lpqa[:, mkslice(orbs), :].reshape(naux, -1), Lpqb[:, mkslice(orbs), :].reshape(naux, -1)]

    naux_ones = np.ones((1, naux))

    norbs = len(orbs)
    Wmn = [np.empty((nw, norbs, nmoa)), np.empty((nw, norbs, nmob))]
    Pi = np.empty((naux, naux))

    logger.info(gw, f'Computing Wmn on imaginary axis ({nw} points)')

    for w in range(nw):
        if gw.verbose >= 4:
            gw.stdout.write(f'{w:4d} ')
            if w % 12 == 11:
                gw.stdout.write('\n')
            gw.stdout.flush()

        #  Pi_inv = (I - Pi)^-1 - I.
        Pi = get_rho_response(freqs[w], mo_energy, Lia_a, Lia_b)
        Pi_inv = get_id_minus_pi_inv_minus_id(Pi, overwrite_input=True)

        # These lines compute
        # Wmn = einsum('Pmn, Qmn, PQ -> mn', Lpq[:, :, mkslice(orbs)], Lpq[:,  :, mkslice(orbs)], Pi_inv)
        for s in range(2):
            l_slice_s = l_slice[s]
            Qmn_s = np.matmul(Pi_inv, l_slice_s)
            Qmn_s *= l_slice_s
            Wmn[s][w] = (naux_ones @ Qmn_s).reshape(norbs, nmo[s])

    gw.stdout.write('\n')
    gw.stdout.flush()

    # for w in range(nw):
    #     Pi = get_rho_response(freqs[w], mo_energy, Lpq[:,:nocc,nocc:])
    #     Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
    #     Qnm = einsum('Pnm,PQ->Qnm',Lpq[:,orbs,:],Pi_inv)
    #     Wmn[:,:,w] = einsum('Qnm,Qmn->mn',Qnm,Lpq[:,:,orbs])

    return Wmn


def get_rho_response_R(eia_a, eia_b, omega, Lia_a, Lia_b, eta):
    """Compute density response function in auxiliary basis at poles.

    Parameters
    ----------
    eia_a : ndarray
        Virtual-occupied energy differences for alpha spin
    eia_b : ndarray
        Virtual-occupied energy differences for beta spin
    omega : float
        Frequency
    Lia_a : ndarray
        Lpq for alpha spin, occupied-virtual block
    Lia_b : ndarray
        Lpq for beta spin, occupied-virtual block
    eta : float
        Broadening parameter for poles on real axis

    Returns
    -------
    Pi : ndarray
        Density response function in auxiliary basis at poles
    """
    # eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    # eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
    # #Pia = einsum('Pia,ia->Pia',Lia,eia)

    # # Response from both spin-up and spin-down density
    # Pi = 2. * einsum('Pia,Qia->PQ',Pia,Lia)
    # return Pi

    naux, _, _ = Lia_a.shape

    Pi = np.zeros((naux, naux), dtype=np.complex128)

    Lia = [Lia_a, Lia_b]
    eia = [eia_a, eia_b]

    for s in range(2):
        Lia_s = Lia[s]
        eia_s = eia[s]

        eia2 = 1.0 / (omega + eia_s + 2j * eta) + 1.0 / (-omega + eia_s)
        eiaR = np.ascontiguousarray(eia2.real)
        eiaI = np.ascontiguousarray(eia2.imag)

        # Response from both spin-up and spin-down density
        PiaR = Lia_s * (eiaR)
        Pi_R = PiaR.reshape(naux, -1) @ Lia_s.reshape(naux, -1).T
        del PiaR
        PiaI = Lia_s * (eiaI)
        Pi_I = PiaI.reshape(naux, -1) @ Lia_s.reshape(naux, -1).T
        del PiaI
        Pi += Pi_R + 1j * Pi_I
    return Pi


class UGWCD(UGWAC):
    def __init__(self, mf, frozen=None, auxbasis=None, chkfile=None):
        super().__init__(mf, frozen=frozen, auxbasis=auxbasis)
        self.chkfile = chkfile  # checkpoint file
        self.eta = 1e-3  # broadening parameter
        return

    def kernel(self):
        """Do one-shot GW calculation using contour deformation."""
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.conv_inds, self.mo_energy = kernel(self)
        logger.timer(self, 'UGWCD', *cput0)
        return
