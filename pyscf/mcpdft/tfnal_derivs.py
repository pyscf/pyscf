#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
import numpy as np
from scipy import linalg
from pyscf.lib import logger


def _reshape_vxc_sigma(vxc0, dens_deriv):
    # d/drho, d/dsigma -> d/drho, d/drho'
    vrho = vxc0[0]
    vxc1 = list(vrho.T)
    if dens_deriv:
        vsigma = vxc0[1]
        vxc1 = vxc1 + list(vsigma.T)
    else:
        vxc1 = [vxc1[0][None, :], vxc1[1][None, :]]
    return vxc1


def _unpack_vxc_sigma(vxc0, rho, dens_deriv):
    # d/drho, d/dsigma -> d/drho, d/drho'
    vrho = vxc0[0]
    vxc1 = list(vrho.T)
    if dens_deriv:
        vsigma = vxc0[1]
        vxc1 = vxc1 + list(vsigma.T)
        vxc1 = _unpack_sigma_vector(vxc1, rho[0][1:4], rho[1][1:4])
    else:
        vxc1 = [vxc1[0][None, :], vxc1[1][None, :]]
    return vxc1


def _pack_fxc_ltri(fxc0, dens_deriv):
    # d2/drho2, d2/drhodsigma, d2/dsigma2
    # -> lower-triangular Hessian matrix
    frho = fxc0[0].T
    fxc1 = [frho[0], ]
    fxc1 += [frho[1], frho[2], ]
    if dens_deriv:
        frhosigma, fsigma = fxc0[1].T, fxc0[2].T
        fxc1 += [frhosigma[0], frhosigma[3], fsigma[0], ]
        fxc1 += [frhosigma[1], frhosigma[4], fsigma[1], fsigma[3], ]
        fxc1 += [frhosigma[2], frhosigma[5], fsigma[2], fsigma[4], fsigma[5]]
    return fxc1


def eval_ot(otfnal, rho, Pi, dderiv=1, weights=None, _unpack_vot=True):
    r'''get the integrand of the on-top xc energy and its functional
    derivatives wrt rho and Pi

    Args:
        rho : ndarray of shape (2,*,ngrids)
            containing spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]

    Kwargs:
        dderiv : integer
            Order of derivatives to return
        weights : ndarray of shape (ngrids)
            used ONLY for debugging the total number of ``translated''
            electrons in the calculation of rho_t
            Not multiplied into anything!
        _unpack_vot : logical
            If True, derivatives with respect to density gradients are
            reported as de/drho' and de/dPi'; otherwise, they are
            reported as de/d|rho'|^2, de/d(rho'.Pi'), and de/d|Pi'|^2

    Returns:
        eot : ndarray of shape (ngrids)
            integrand of the on-top exchange-correlation energy
        vot : ndarrays of shape (*,ngrids) or None
            first functional derivative of Eot wrt (density, pair
            density) and their derivatives. If _unpack_vot = True, shape
            and format is ([a, ngrids], [b, ngrids]) : (vrho, vPi);
            otherwise, [c, ngrids] : [rho,Pi,|rho'|^2,tau,rho'.Pi',|Pi'|^2]
            ftGGA: a=4, b=4, c=5 (drop tau)
            tmGGA: a=5, b=1, c=4 (drop Pi')
            tGGA: a=4, b=1, c=3 (drop Pi', tau)
            *tLDA: a=1, b=1, c=2 (drop rho', tau)
        fot : ndarray of shape (*,ngrids) or None
            second functional derivative of Eot wrt density, pair
            density, and derivatives; first dimension is lower-
            triangular matrix elements corresponding to the basis
            (rho, Pi, |rho'|^2, rho'.Pi', |Pi'|^2)
            stopping at Pi (3 elements) for *tLDA and |rho'|^2 (6
            elements) for tGGA.
    '''
    if dderiv > 2:
        raise NotImplementedError("Translation of density derivatives of "
                                  "higher order than 2")
    if rho.ndim == 2: rho = rho[:, None, :]
    if Pi.ndim == 1: Pi = Pi[None, :]
    assert (rho.shape[0] == 2)
    assert (rho.shape[1] <= 6), "Undefined behavior for this function"

    nderiv = rho.shape[1]
    nderiv_Pi = Pi.shape[0]

    rho_t = otfnal.get_rho_translated(Pi, rho)
    # LDA in libxc has a special numerical problem with zero-valued densities
    # in one spin
    if nderiv == 0:
        idx = (rho_t[0, 0] > 1e-15) & (rho_t[1, 0] < 1e-15)
        rho_t[1, 0, idx] = 1e-15
        idx = (rho_t[0, 0] < 1e-15) & (rho_t[1, 0] > 1e-15)
        rho_t[0, 0, idx] = 1e-15

    # mGGA in libxc has special numerical problem with zero-valued densities in
    # one spin!
    if nderiv == 5:
        idx = (rho_t[0, 4] > 1e-15) & (rho_t[1, 4] < 1e-15)
        rho_t[1, 4, idx] = 1e-15
        idx = (rho_t[0, 4] < 1e-15) & (rho_t[1, 4] > 1e-15)
        rho_t[0, 4, idx] = 1e-15

    rho_tot = rho.sum(0)

    if nderiv > 4 and dderiv > 1:
        raise NotImplementedError("Meta-GGA functional Hessians")

    if 1 < nderiv <= 4:
        rho_deriv = rho_tot[1:4, :]
    elif 4 < nderiv <= 5:
        rho_deriv = rho_tot[1:5, :]
    else:
        rho_deriv = None

    Pi_deriv = Pi[1:4, :] if nderiv_Pi > 1 else None
    xc_grid = otfnal._numint.eval_xc(otfnal.otxc, (rho_t[0, :, :],
                                                   rho_t[1, :, :]), spin=1, relativity=0, deriv=dderiv,
                                     verbose=otfnal.verbose)[:dderiv + 1]
    eot = xc_grid[0] * rho_t[:, 0, :].sum(0)
    if (weights is not None) and otfnal.verbose >= logger.DEBUG:
        nelec = rho_t[0, 0].dot(weights) + rho_t[1, 0].dot(weights)
        logger.debug(otfnal, ('MC-PDFT: Total number of electrons in (this '
                              'chunk of) the total density = %s'), nelec)
        ms = (rho_t[0, 0].dot(weights) - rho_t[1, 0].dot(weights)) / 2.0
        logger.debug(otfnal, ('MC-PDFT: Total ms = (neleca - nelecb) / 2 in '
                              '(this chunk of) the translated density = %s'), ms)
    vot = fot = None
    if dderiv > 0:
        # vrho, vsigma = xc_grid[1][:2]
        vxc = list(xc_grid[1][0].T)
        if otfnal.dens_deriv > 0:
            vxc = vxc + list(xc_grid[1][1].T)

        # vrho, vsigma, vlapl, vtau = xc_grid[1][:4]
        if otfnal.dens_deriv > 1:
            # we might get a None for one of the derivatives..
            # we get None for the laplacian derivative
            if xc_grid[1][2] is not None:
                raise NotImplementedError("laplacian translated meta-GGA functionals")

            # Here is the tau term
            vxc = vxc + list(xc_grid[1][3].T)

        vot = otfnal.jT_op(vxc, rho, Pi)
        if _unpack_vot: vot = _unpack_sigma_vector(vot,
                                                   deriv1=rho_deriv, deriv2=Pi_deriv)
    if dderiv > 1:
        # I should implement this entirely in terms of the gradient norm, since
        # that reduces the number of grid columns from 25 to 9 for t-GGA and
        # from 64 to 25 for ft-GGA (and steps around the need to "unpack"
        # fsigma and frhosigma entirely).
        fxc = _pack_fxc_ltri(xc_grid[2], otfnal.dens_deriv)
        # First pass: fxc
        fot = _jT_f_j(fxc, otfnal.jT_op, rho, Pi, rec=otfnal)
        # Second pass: translation derivatives
        fot_d_jT = otfnal.d_jT_op(vxc, rho, Pi)
        fot[:fot_d_jT.shape[0]] += fot_d_jT
    return eot, vot, fot


def unpack_vot(packed, rho, Pi):
    if rho.ndim == 2: rho = rho[:, None, :]
    if Pi.ndim == 1: Pi = Pi[None, :]
    assert (rho.shape[0] == 2)

    nderiv = rho.shape[1]
    nderiv_Pi = Pi.shape[0]

    rho_tot = rho.sum(0)
    rho_deriv = rho_tot[1:4, :] if nderiv > 1 else None
    Pi_deriv = Pi[1:4, :] if nderiv_Pi > 1 else None
    return _unpack_sigma_vector(packed, deriv1=rho_deriv, deriv2=Pi_deriv)


def _unpack_sigma_vector(packed, deriv1=None, deriv2=None):
    # For GGAs, libxc differentiates with respect to
    #   sigma[0] = nabla^2 rhoa
    #   sigma[1] = nabla rhoa . nabla rhob
    #   sigma[2] = nabla^2 rhob
    # So we have to multiply the Jacobian to obtain the requested derivatives:
    #   J[0,nabla rhoa] = 2 * nabla rhoa
    #   J[0,nabla rhob] = 0
    #   J[1,nabla rhoa] = nabla rhob
    #   J[1,nabla rhob] = nabla rhoa
    #   J[2,nabla rhoa] = 0
    #   J[2,nabla rhob] = 2 * nabla rhob
    if len(packed) > 5:
        raise RuntimeError("{} {}".format(len(packed), [p.shape for p in packed[:5]]))
    ncol1 = 1
    if deriv1 is not None and len(packed) > 2:
        ncol1 += deriv1.shape[0]
    ncol2 = 1 + 3 * int((deriv2 is not None) and len(packed) > 3)
    ngrid = packed[0].shape[-1]  # Don't assume it's an ndarray
    unp1 = np.empty((ncol1, ngrid), dtype=packed[0].dtype)
    unp2 = np.empty((ncol2, ngrid), dtype=packed[0].dtype)
    unp1[0] = packed[0]
    unp2[0] = packed[1]
    if ncol1 > 1:
        unp1[1:4] = 2 * deriv1[:3] * packed[2]
        if ncol1 > 4:
            # Deal with the tau term
            unp1[4:5] = packed[3:4]
        if ncol2 > 1:
            unp1[1:4] += deriv2 * packed[-2]
            unp2[1:4] = (2 * deriv2 * packed[-1]) + (deriv1[:3] * packed[-2])
    return unp1, unp2


def contract_vot(vot, rho, Pi):
    '''Evalute the product of unpacked vot with perturbed density, pair density, and derivatives.

        Args:
            vot : (ndarray of shape (*,ngrids), ndarray of shape (*, ngrids))
                format is ([a, ngrids], [b, ngrids]) : (vrho, vPi)
                ftGGA: a=4, b=4
                tGGA: a=4, b=1
                *tLDA: a=1, b=1
            rho : ndarray of shape (*,ngrids)
                containing density [and derivatives]
                the density contracted with vot
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]
                the density contracted with vot

        Returns:
            cvot : ndarray of shape (ngrids)
                product of vot wrt (density, pair density) and their derivatives
        '''
    vrho, vPi = vot
    if rho.shape[0] == 2: rho = rho.sum(0)
    if rho.ndim == 1: rho = rho[None, :]
    if Pi.ndim == 1: Pi = Pi[None, :]

    cvot = vrho[0] * rho[0] + vPi[0] * Pi[0]
    if len(vrho) > 1:
        cvot += (vrho[1:4,:] * rho[1:4, :]).sum(0)

    if len(vPi) > 1:
        cvot += (vPi[1:4, :] * Pi[1:4, :]).sum(0)

    return cvot


def contract_fot(otfnal, fot, rho0, Pi0, rho1, Pi1, unpack=True,
                 vot_packed=None):
    r''' Evaluate the product of a packed lower-triangular matrix
    with perturbed density, pair density, and derivatives.

    Args:
        fot : ndarray of shape (*,ngrids)
            Lower-triangular matrix elements corresponding to the basis
            (rho, Pi, |drho|^2, drho'.dPi, |dPi|^2) stopping at Pi (3
            elements) for *tLDA and |drho|^2 (6 elements) for tGGA.
        rho0 : ndarray of shape (2,*,ngrids)
            containing density [and derivatives]
            the density at which fot was evaluated
        Pi0 : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
            the density at which fot was evaluated
        rho1 : ndarray of shape (2,*,ngrids)
            containing density [and derivatives]
            the density contracted with fot
        Pi1 : ndarray with shape (*,ngrids)
            containing on-top pair density [and derivatives]
            the density contracted with fot

    Kwargs:
        unpack : logical
            If True, returns vot1 in unpacked shape:
                (ndarray of shape (*,ngrids),
                 ndarray of shape (*,ngrids))
            corresponding to (density, pair density) and their
            derivatives. This requires vot_packed for *tGGA functionals
            Otherwise, returns vot1 in packed shape:
                (rho, Pi, |rho'|^2, rho'.Pi', |Pi'|^2)
            stopping at Pi (3 elements) for *tLDA and |rho'|^2 (6
            elements) for tGGA.
        vot_packed : ndarray of shape (*,ngrids)
            Vector elements corresponding to the basis
            (rho, Pi, |drho|^2, drho'.dPi, |dPi|^2) stopping at Pi (2
            elements) for *tLDA and |drho|^2 (3 elements) for tGGA.
            Required if unpack == True for *tGGA functionals
            (because vot_|drho|^2 contributes to fot_rho',rho', etc.)

    Returns:
        vot1 : (ndarray of shape (*,ngrids),
                ndarray of shape (*,ngrids))
            product of fot wrt (density, pair density)
            and their derivatives
    '''
    if rho0.shape[0] == 2: rho0 = rho0.sum(0)  # Never has exactly 1 derivative
    if rho0.ndim == 1: rho0 = rho0[None, :]
    if Pi0.ndim == 1: Pi0 = Pi0[None, :]
    if rho1.shape[0] == 2: rho1 = rho1.sum(0)  # Never has exactly 1 derivative
    if rho1.ndim == 1: rho1 = rho1[None, :]
    if Pi1.ndim == 1: Pi1 = Pi1[None, :]

    ngrids = fot[0].shape[-1]
    vrho1 = np.zeros(ngrids, dtype=fot[0].dtype)
    vPi1 = np.zeros(ngrids, dtype=fot[2].dtype)
    vrho1, vPi1 = np.zeros_like(rho1), np.zeros_like(Pi1)

    # TODO: dspmv implementation
    vrho1 = fot[0] * rho1[0] + fot[1] * Pi1[0]
    vPi1 = fot[2] * Pi1[0] + fot[1] * rho1[0]

    rho0p = Pi0p = rho1p = Pi1p = None
    v1 = [vrho1, vPi1]
    if len(fot) > 3:
        srr = 2 * (rho0[1:4, :] * rho1[1:4, :]).sum(0)
        vrho1 += fot[3] * srr
        vPi1 += fot[4] * srr
        vrr = fot[3] * rho1[0] + fot[4] * Pi1[0] + fot[5] * srr
        rho0p = rho0[1:4]
        rho1p = rho1[1:4]
        v1 = [vrho1, vPi1, vrr]
    if len(fot) > 6:
        srP = ((rho0[1:4, :] * Pi1[1:4, :]).sum(0)
               + (rho1[1:4, :] * Pi0[1:4, :]).sum(0))
        sPP = 2 * (Pi0[1:4, :] * Pi1[1:4, :]).sum(0)
        vrho1 += fot[6] * srP + fot[10] * sPP
        vPi1 += fot[7] * srP + fot[11] * sPP
        vrr += fot[8] * srP + fot[12] * sPP
        vrP = (fot[6] * rho1[0] + fot[7] * Pi1[0]
               + fot[8] * srr + fot[9] * srP + fot[13] * sPP)
        vPP = (fot[10] * rho1[0] + fot[11] * Pi1[0]
               + fot[12] * srr + fot[13] * srP + fot[14] * sPP)
        Pi0p = Pi0[1:4]
        Pi1p = Pi1[1:4]
        v1 = [vrho1, vPi1, vrr, vrP, vPP]

    ggrad = (rho0p is not None) or (Pi0p is not None)
    if unpack:
        vrho1, vPi1 = _unpack_sigma_vector(v1, rho0p, Pi0p)
        if ggrad:
            if vot_packed is None:
                raise RuntimeError("Cannot evaluate fot.x in terms of "
                                    "unpacked density gradients without vot_packed")
            vrho2, vPi2 = _unpack_sigma_vector(vot_packed, rho1p, Pi1p)
            if vrho1.shape[0] > 1: vrho1[1:4, :] += vrho2[1:4, :]
            if vPi1.shape[0] > 1:  vPi1[1:4, :] += vPi2[1:4, :]
        v1 = (vrho1, vPi1)

    return v1


def _jT_f_j(frr, jT_op, *args, **kwargs):
    r''' Apply a jacobian function taking *args to the lower-triangular
    second-derivative array frr'''
    nel = len(frr)
    nr = int(round(np.sqrt(1 + 8 * nel) - 1)) // 2
    rec = kwargs.get('rec', None)
    ngrids = frr[0].shape[-1]

    # build square-matrix index array to address packed matrix frr w/o copying
    ltri_ix = np.tril_indices(nr)
    idx_arr = np.zeros((nr, nr), dtype=np.int32)
    idx_arr[ltri_ix] = range(nel)
    idx_arr += idx_arr.T
    diag_ix = np.diag_indices(nr)
    idx_arr[diag_ix] = idx_arr[diag_ix] // 2

    # first pass: jT . frr -> fcr
    fcr = np.stack([jT_op([frr[i] for i in ix_row], *args)
                    for ix_row in idx_arr], axis=1)

    # second pass. fcr is a rectangular matrix (unavoidably)
    nc = fcr.shape[0]
    if getattr(rec, 'verbose', 0) < logger.DEBUG:
        fcc = np.empty((nc * (nc + 1) // 2, ngrids), dtype=fcr.dtype)
        i = 0
        for ix_row, fc_row in enumerate(fcr):
            di = ix_row + 1
            j = i + di
            fcc[i:j] = jT_op(fc_row, *args)[:di]
            i = j
    else:
        fcc = np.empty((nc, nc, ngrids), dtype=fcr.dtype)
        for fcc_row, fcr_row in zip(fcc, fcr):
            fcc_row[:] = jT_op(fcr_row, *args)
        for i in range(1, nc):
            for j in range(i):
                scale = (fcc[i, j] + fcc[j, i]) / 2
                scale[scale == 0] = 1
                logger.debug(rec, 'MC-PDFT jT_f_j symmetry check %d,%d: %e',
                             i, j, linalg.norm((fcc[i, j] - fcc[j, i]) / scale))
        ltri_ix = np.tril_indices(nc)
        fcc = fcc[ltri_ix]

    return fcc


def _gentLDA_jT_op(x, rho, Pi, R, zeta):
    # On a grid, multiply the transpose of the Jacobian
    #     d(trhoa,trhob) [fictitous densities]
    # J = ______________
    #     d(rho,Pi)      [real densities]
    # by a vector x_(trhoa,trhob)
    ngrid = rho.shape[-1]
    if R.ndim > 1: R = R[0]

    # ab -> cs coordinate transformation
    xc = (x[0] + x[1]) / 2.0
    xm = (x[0] - x[1]) / 2.0

    # Charge sector has no explicit rho denominator
    # and so does not require indexing to avoid
    # division by zero
    jTx = np.zeros((2, ngrid), dtype=x[0].dtype)
    jTx[0] = xc + xm * (zeta[0] - (2 * R * zeta[1]))

    # Spin sector has a rho denominator
    idx = (rho[0] > 1e-15)
    zeta = zeta[1, idx]
    rho = rho[0, idx]
    xm = xm[idx]
    jTx[1, idx] = 4 * xm * zeta / rho

    return jTx


def _tGGA_jT_op(x, rho, Pi, R, zeta):
    # On a grid, multiply the transpose of the Jacobian
    #     d(trho?'.trho?') [fictitous density gradients]
    # J = ______________
    #     d(rho,Pi,|rho'|) [real densities and gradients]
    # by a vector x_(|trho*'|^2) in the context of tGGAs
    ngrid = rho.shape[-1]
    jTx = np.zeros((3, ngrid), dtype=x[0].dtype)
    if R.ndim > 1: R = R[0]

    # ab -> cs coordinate transformation
    xcc = (x[2] + x[4] + x[3]) / 4.0
    xcm = (x[2] - x[4]) / 2.0
    xmm = (x[2] + x[4] - x[3]) / 4.0

    # Gradient-gradient sector
    idx = (zeta[0]!=1)
    jTx[2] = x[2]
    jTx[2,idx] = (xcc + xcm * zeta[0] + xmm * zeta[0] * zeta[0])[idx]

    # Finite-precision safety

    # Density-gradient sector
    idx = (rho[0] > 1e-15)
    sigma_fac = ((rho[1:4].conj() * rho[1:4]).sum(0) * zeta[1])
    sigma_fac = ((xcm + 2 * zeta[0] * xmm) * sigma_fac)[idx]
    rho = rho[0, idx]
    R = R[idx]
    sigma_fac = -2 * sigma_fac / rho
    jTx[0, idx] = R * sigma_fac
    jTx[1, idx] = -2 * sigma_fac / rho

    return jTx


def _tmetaGGA_jT_op(x, rho, Pi, R, zeta):
    # output ordering is
    # ordering: rho, Pi, |rho'|^2, tau
    ngrid = rho.shape[-1]
    jTx = np.zeros((4, ngrid), dtype=x[0].dtype)
    if R.ndim > 1:
        R = R[0]

    # ab -> cs coordinate transformation
    xc = (x[5] + x[6]) / 2.0
    xm = (x[5] - x[6]) / 2.0

    # easy part
    jTx[3] = xc + zeta[0] * xm

    tau_lapl_factor = zeta[1] * rho[4] * xm
    idx = rho[0] > 1e-15
    rho = rho[0, idx]
    R = R[idx]

    tau_lapl_factor = 2 * tau_lapl_factor[idx] / rho

    jTx[0, idx] = -R * tau_lapl_factor
    jTx[1, idx] = 2 * tau_lapl_factor / rho

    return jTx


def _tGGA_jT_op_m2z(x, rho, zeta, srr):
    # cs -> rho,zeta step of _tGGA_jT_op above
    # unused; for contemplative purposes only
    jTx = np.empty_like(x)
    jTx[0] = x[0] + zeta[0] * x[1]
    jTx[1] = x[1] * rho + (x[3] + 2 * zeta[0] * x[4]) * srr
    jTx[2] = x[2] + x[3] * zeta[0] + x[4] * zeta[0] * zeta[0]
    jTx[3] = 0
    jTx[4] = 0
    return jTx


def _ftGGA_jT_op_m2z(x, rho, zeta, srz, szz):
    # cs -> rho,zeta step of _ftGGA_jT_op below
    jTx = np.empty_like(x)
    jTx[0] = 2 * x[4] * (zeta[0] * srz + rho * szz) + x[3] * srz
    jTx[1] = 2 * x[4] * rho * srz
    jTx[2] = 0
    jTx[3] = (x[3] + 2 * x[4] * zeta[0]) * rho
    jTx[4] = x[4] * rho * rho
    return jTx


def _ftGGA_jT_op_z2R(x, zeta, srP, sPP):
    # rho,zeta -> rho,R step of _ftGGA_jT_op below
    jTx = np.empty_like(x)
    jTx[0] = x[0]
    jTx[1] = (x[1] * zeta[1] + x[3] * srP * zeta[2] +
              2 * x[4] * sPP * zeta[1] * zeta[2])
    jTx[2] = x[2]
    jTx[3] = x[3] * zeta[1]
    jTx[4] = x[4] * zeta[1] * zeta[1]
    return jTx


def _ftGGA_jT_op_R2Pi(x, rho, R, srr, srP, sPP):
    # rho,R -> rho,Pi step of _ftGGA_jT_op below
    if rho.ndim > 1: rho = rho[0]
    if R.ndim > 1: R = R[0]
    jTx = np.empty_like(x)
    ri = np.empty_like(x)
    ri[0, :] = 0.0
    idx = rho > 1e-15
    ri[0, idx] = 1.0 / rho[idx]
    for i in range(4):
        ri[i + 1] = ri[i] * ri[0]

    jTx[0] = (x[0] - 2 * R * x[1] * ri[0]
              + x[3] * (6 * R * ri[1] * srr - 8 * srP * ri[2])
              + x[4] * (-24 * R * R * ri[2] * srr + 80 * R * ri[3] * srP
                        - 64 * ri[4] * sPP))
    jTx[1] = (4 * x[1] * ri[1] - 8 * x[3] * ri[2] * srr
              + x[4] * (32 * R * ri[3] * srr - 64 * ri[4] * srP))
    jTx[2] = x[2] - 2 * R * x[3] * ri[0] + 4 * x[4] * R * R * ri[1]
    jTx[3] = 4 * x[3] * ri[1] - 16 * x[4] * R * ri[2]
    jTx[4] = 16 * x[4] * ri[3]
    return jTx


def _ftGGA_jT_op(x, rho, Pi, R, zeta):
    # On a grid, evaluate the contribution to the matrix-vector product
    # of the transpose of the Jacobian
    #     d(trho?'.trho?') [fictitous density gradients]
    # J = ______________
    #     d(rho,Pi,|rho'|) [real densities and gradients]
    # with a vector x_(|trho*'|^2), which is present in ftGGAs and
    # missing in tGGAs
    ngrid = rho.shape[-1]
    jTx = np.zeros((5, ngrid), dtype=x[0].dtype)

    # ab -> cs step
    jTx[2] = (x[2] + x[4] + x[3]) / 4.0
    jTx[3] = (x[2] - x[4]) / 2.0
    jTx[4] = (x[2] + x[4] - x[3]) / 4.0
    x = jTx

    # Intermediates
    srr = (rho[1:4, :] * rho[1:4, :]).sum(0)
    srP = (rho[1:4, :] * R[1:4, :]).sum(0)
    sPP = (R[1:4, :] * R[1:4, :]).sum(0)
    srz = srP * zeta[1]
    szz = sPP * zeta[1] * zeta[1]

    # cs -> rho,zeta step
    x = _ftGGA_jT_op_m2z(x, rho[0], zeta, srz, szz)

    # rho,zeta -> rho,R step
    x = _ftGGA_jT_op_z2R(x, zeta, srP, sPP)

    # rho,R -> rho,Pi step
    srP = (rho[1:4, :] * Pi[1:4, :]).sum(0)
    sPP = (Pi[1:4, :] * Pi[1:4, :]).sum(0)
    jTx = _ftGGA_jT_op_R2Pi(x, rho, R, srr, srP, sPP)

    return jTx


def _gentLDA_d_jT_op(x, rho, Pi, R, zeta):
    # On a grid, differentiate once the Jacobian
    #     d(trhoa,trhob) [fictitous densities]
    # J = ______________
    #     d(rho,Pi)      [real densities]
    # and multiply by x_(trhoa,trhob) so as to compute the nonlinear-
    # translation contribution to the second functional derivatives of
    # the on-top energy in tLDA and ftLDA.
    rho = rho[0]
    Pi = Pi[0]
    R = R[0]
    ngrid = rho.shape[-1]
    f = np.zeros((3, ngrid), dtype=x[0].dtype)

    # ab -> cs
    xm = (x[0] - x[1]) / 2.0

    # Indexing
    idx = rho > 1e-15
    rho = rho[idx]
    Pi = Pi[idx]
    xm = xm[idx]
    R = R[idx]
    zeta = zeta[:, idx]

    # Intermediates
    # R = otfnal.get_ratio (Pi, rho/2)
    # zeta = otfnal.get_zeta (R, fn_deriv=2)[1:]
    xmw = 2 * xm / rho
    z1 = xmw * (zeta[1] + 2 * R * zeta[2])

    # without further ceremony
    f[0, idx] = R * z1
    f[1, idx] = -2 * z1 / rho
    f[2, idx] = xmw * 8 * zeta[2] / rho / rho

    return f


def _tGGA_d_jT_op(x, rho, Pi, R, zeta):
    # On a grid, differentiate once the Jacobian
    #     d(trho?'.trho?') [fictitous density gradients]
    # J = ______________
    #     d(rho,Pi,|rho'|) [real densities and gradients]
    # and multiply by x_(trho?'.trho?') so as to compute the nonlinear-
    # translation contribution to the second functional derivatives of
    # the on-top energy in tGGAs.

    # Generates contributions to the first five elements
    # of the lower-triangular packed Hessian
    ngrid = rho.shape[-1]
    f = np.zeros((5, ngrid), dtype=x[0].dtype)

    # Indexing
    idx = rho[0] > 1e-15
    rho = rho[0:4, idx]
    Pi = Pi[0:1, idx]
    x = [xi[idx] for xi in x]
    R = R[0, idx]
    zeta = zeta[:, idx]

    # ab -> cs
    xcm = (x[2] - x[4]) / 2.0
    xmm = (x[2] + x[4] - x[3]) / 4.0

    # Intermediates
    sigma = (rho[1:4] * rho[1:4]).sum(0)
    rho = rho[0]
    rho2 = rho * rho
    rho3 = rho2 * rho
    rho4 = rho3 * rho

    # coefficient of dsigma dz
    xcm += 2 * zeta[0] * xmm
    f[3, idx] = -2 * xcm * R * zeta[1] / rho
    f[4, idx] = 4 * xcm * zeta[1] / rho2

    # coefficient of d^2 z
    xcm *= sigma
    f[0, idx] = 2 * xcm * R * (3 * zeta[1] + 2 * R * zeta[2]) / rho2
    f[1, idx] = -8 * xcm * (zeta[1] + R * zeta[2]) / rho3
    f[2, idx] = 16 * xcm * zeta[2] / rho4

    # coefficient of dz dz
    xmm *= 8 * sigma * zeta[1] * zeta[1] / rho2
    f[0, idx] += xmm * R * R
    f[1, idx] -= 2 * xmm * R / rho
    f[2, idx] += 4 * xmm / rho2

    return f


#   r,r
#   1,r   1,1
#   srr,r srr,1 srr,srr
#   sr1,r sr1,1 sr1,srr sr1,sr1
#   s11,r s11,1 s11,srr s11,sr1 s11,s11

def _ftGGA_d_jT_op_m2z(v, rho, zeta, srz, szz):
    # srm += srz*r
    # smm += 2srz*r*z + szz*r*r
    # 0  : r, r
    # 1  : r, z
    # 6  : srz, r
    # 7  : srz, z
    # 10 : szz, r
    ngrids = v.shape[1]
    f = np.zeros((15, ngrids), dtype=v.dtype)
    f[0] = 2 * v[4] * szz
    f[1] = 2 * v[4] * srz
    f[6] = v[3] + 2 * v[4] * zeta[0]
    f[7] = 2 * v[4] * rho
    f[10] = 2 * v[4] * rho
    assert (tuple(f[0].shape) == tuple(f[1].shape))
    assert (tuple(f[0].shape) == tuple(f[6].shape))
    assert (tuple(f[0].shape) == tuple(f[10].shape))
    return f


def _ftGGA_d_jT_op_z2R(v, zeta, srP, sPP):
    # z = z[0]
    # srz = srP*z[1]
    # szz = sPP*z[1]**2
    # 2  : P, P
    # 7  : srP, P
    # 11 : sPP, P
    ngrids = v.shape[1]
    f = np.zeros((15, ngrids), dtype=v.dtype)
    f[2] = 2 * v[4] * sPP * (zeta[3] * zeta[1] + zeta[2] * zeta[2])
    f[2] += v[1] * zeta[2] + v[3] * srP * zeta[3]
    f[7] = v[3] * zeta[2]
    f[11] = 2 * v[4] * zeta[1] * zeta[2]
    assert (tuple(f[2].shape) == tuple(f[7].shape))
    assert (tuple(f[2].shape) == tuple(f[11].shape))
    return f


def _ftGGA_d_jT_op_R2Pi(v, rho, Pi, srr, srP, sPP):
    # R = 4Pi/(r**2) = Pi*d[0]
    # srR = srP*d[0] + srr*Pi*d[1]
    # sRR = sPP*d[0]**2 + 2*Pi*d[1]*srP*d[0] + srr*(Pi*d[1])**2
    #        d^n(d[0])
    # d[n] = ---------
    #          dr^n
    ngrids = v.shape[-1]
    f = np.zeros((15, ngrids), dtype=v.dtype)
    d = np.zeros((4, ngrids), dtype=v.dtype)
    idx = np.abs(rho) > 1e-15
    d[0, idx] = 4 / rho[idx] / rho[idx]
    d[1, idx] = -2 * d[0, idx] / rho[idx]
    d[2, idx] = -3 * d[1, idx] / rho[idx]
    d[3, idx] = -4 * d[2, idx] / rho[idx]
    # rho, rho
    f[0] = v[1] * Pi * d[2]
    f[0] += v[3] * (srP * d[2] + srr * Pi * d[3])
    f[0] += 2 * v[4] * sPP * (d[2] * d[0] + d[1] * d[1])
    f[0] += 2 * v[4] * srP * Pi * (3 * d[2] * d[1] + d[3] * d[0])
    f[0] += 2 * v[4] * srr * Pi * Pi * (d[3] * d[1] + d[2] * d[2])
    # rho, Pi
    f[1] = v[1] * d[1] + v[3] * srr * d[2]
    f[1] += 2 * v[4] * srP * (d[2] * d[0] + d[1] * d[1])
    f[1] += 4 * v[4] * srr * Pi * d[2] * d[1]
    # Pi, Pi
    f[2] = 2 * v[4] * srr * d[1] * d[1]
    # rho, rr
    f[3] = v[3] * Pi * d[2] + 2 * v[4] * Pi * Pi * d[2] * d[1]
    # Pi, rr
    f[4] = v[3] * d[1] + 2 * v[4] * Pi * d[1] * d[1]
    # rho, rP
    f[6] = v[3] * d[1] + 2 * v[4] * Pi * (d[2] * d[0] + d[1] * d[1])
    # Pi, rP
    f[7] = 2 * v[4] * d[0] * d[1]
    # rho, PP
    f[10] = 2 * v[4] * d[0] * d[1]
    for row in f[1:]:
        if hasattr(row, 'shape'):
            assert (tuple(row.shape) == tuple(f[0].shape))
    return f


def _ftGGA_d_jT_op(v, rho, Pi, R, zeta):
    # raise NotImplementedError ("Second density derivatives for fully-"
    #    "translated GGA functionals")
    # Generates contributions to the first five elements,
    # then 6,7, then 10,11
    # of the lower-triangular packed Hessian
    # (I.E., no double gradient derivatives)
    # for the terms added in the fully-translated extension of tGGA

    # ab -> cs
    vcm = (v[2] - v[4]) / 2.0
    vmm = (v[2] + v[4] - v[3]) / 4.0
    v[3] = vcm
    v[4] = vmm
    v = np.asarray(v)

    # Intermediates
    srr = (rho[1:4, :] * rho[1:4, :]).sum(0)
    srP = (rho[1:4, :] * R[1:4, :]).sum(0)
    sPP = (R[1:4, :] * R[1:4, :]).sum(0)
    srz = srP * zeta[1]
    szz = sPP * zeta[1] * zeta[1]

    # cs -> rho, zeta
    f = _ftGGA_d_jT_op_m2z(v, rho[0], zeta, srz, szz)
    v = _ftGGA_jT_op_m2z(v, rho[0], zeta, srz, szz)

    # rho, zeta -> rho, R
    # The for loops here are because I'm guessing that repeated
    # initialization of large arrays to zero is slower than a short,
    # shallow Python loop
    f = _jT_f_j(f, _ftGGA_jT_op_z2R, zeta, srP, sPP)
    f += _ftGGA_d_jT_op_z2R(v, zeta, srP, sPP)
    v = _ftGGA_jT_op_z2R(v, zeta, srP, sPP)

    # rho, R -> rho, Pi
    srP = (rho[1:4, :] * Pi[1:4, :]).sum(0)
    sPP = (Pi[1:4, :] * Pi[1:4, :]).sum(0)
    f = _jT_f_j(f, _ftGGA_jT_op_R2Pi, rho, R, srr, srP, sPP)
    f += _ftGGA_d_jT_op_R2Pi(v, rho[0], Pi[0], srr, srP, sPP)

    return f
