#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#         Gengzhi Yang <genzyang17@gmail.com>
#


import numpy

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

from pyscf.lo.stability import stability_newton
from pyscf.pbc.lo.base import get_kmesh
from pyscf.pbc.tools import k2gamma


def pipek_stability_jacobi(mlo, verbose=None, return_status=False, Rmax=10):
    r'''Jacobi-sweep stability check for k-point PMWFs.

    This routine tests whether any translationally symmetric 2×2 Jacobi
    rotations can increase the PM objective. Specifically, it considers
    real-valued pairwise rotations between Wannier functions
    :math:`(w_{\mathbf{0}i}, w_{\mathbf{R}j})`, where the lattice vector
    :math:`\mathbf{R}` is restricted to a finite real-space range.

    For each orbital pair ``(i,j)`` and each allowed ``R``, the function
    searches over a discrete set of candidate rotation angles
    (``π/4, π/2, 3π/4``) and applies any rotation that yields an
    increase in the objective greater than ``mlo.conv_tol``.

    Kwargs:
        Rmax : float or None
            Real-space cutoff that controls the range of pairwise rotations
            tested in the Jacobi sweep. Only lattice vectors satisfying
            :math:`|\mathbf{R}| < R_{\mathrm{max}}` are included.
            Unit: Bohr. Default is 10. If set to ``None``, all R vectors
            in the BvK will be used.
    '''
    log = logger.new_logger(mlo, verbose)
    exponent = mlo.exponent

    cell = mlo.cell
    kpts = mlo.kpts
    norb = mlo.norb
    Nk = len(kpts)

    tril_ijdx = numpy.tril_indices(norb, k=-1)
    tril_idx, tril_jdx = tril_ijdx
    thetapool = numpy.asarray([1,2,3])*0.25*numpy.pi

    kmesh = get_kmesh(cell, kpts)
    Rs_bvk = k2gamma.translation_vectors_for_kmesh(cell, kmesh)
    if Rmax is None:
        Rs = Rs_bvk
    else:
        Rs = Rs_bvk[numpy.linalg.norm(Rs_bvk, axis=1) < Rmax]
    log.debug('Checking pairwise stability for WF pairs (0i, Rj) for Rvec:\n%s', Rs)
    phase = numpy.exp(1j * numpy.dot(Rs, kpts.T))
    phase_bvk = numpy.exp(1j * numpy.dot(Rs_bvk, kpts.T))

    latvec_bvk = lib.einsum('ix,i->ix', cell.lattice_vectors(), kmesh)
    def get_shift_map(Rs, R0, logacc=8):
        ''' Mapping Rs-R0 back to Rs using BvK lattice vectors

            Returns:
                shift_map : array
                    Rs[i] - R0 = Rs[shift_map[i]]
        '''
        Ts = numpy.linalg.solve(latvec_bvk.T, Rs.T).T
        T0 = numpy.linalg.solve(latvec_bvk.T, R0)
        Ts1 = Ts - T0
        # shift Ts back to [0,1)^3
        Ts = numpy.round(Ts, logacc+1)%1
        Ts1 = numpy.round(Ts1, logacc+1)%1
        shift_map = numpy.where(abs(Ts1[:,None,:]-Ts[None,:,:]).max(axis=2)<10**-logacc)[1]
        return shift_map

    shift_map = [get_shift_map(Rs_bvk, R) for R in Rs]  # NR x Nk

    def update_rotation_local_(u, theta, i, j, expikR):
        for k,x in enumerate(u):
            xi = x[:,i].copy()
            xj = x[:,j].copy()
            x[:,i] = xi*numpy.cos(theta) + xj*numpy.sin(theta)*expikR[k]
            x[:,j] = -xi*numpy.sin(theta)*expikR[k].conj() + xj*numpy.cos(theta)

    u = mlo.identity_rotation()
    stable = True
    while True:
        mo_coeff = mlo.rotate_orb(u)
        pop0 = mlo.atomic_pops(cell, mo_coeff, mode='00').real
        pop0 = numpy.ascontiguousarray(lib.einsum('xii->ix', pop0))
        proj0k = mlo.atomic_pops(cell, mo_coeff, mode='0k')
        proj0R = numpy.ascontiguousarray(lib.einsum('xikj,Rk->Rijx', proj0k, phase_bvk.conj()).real)
        proj0k = None

        pop0exp = pop0**exponent
        Lij = (pop0exp[tril_idx] + pop0exp[tril_jdx]).sum(axis=1)

        dLij = numpy.zeros_like(Lij)
        thetaij = numpy.zeros_like(Lij)
        Ridxij = numpy.zeros(Lij.shape, dtype=int)

        for iR,R in enumerate(Rs):
            # Q_{0,TA} -> Q_{R,TA}
            popR = pop0.reshape(norb,Nk,-1)[:,shift_map[iR]].reshape(norb,-1)

            Qi_i = pop0[tril_idx]
            Qi_j = popR[tril_jdx]
            Pij_ij = proj0R[iR, tril_idx, tril_jdx]

            for theta in thetapool:
                c = numpy.cos(theta)
                s = numpy.sin(theta)
                c2 = c**2
                s2 = s**2
                cs2 = c * s * 2

                # Population after rotations
                Qitild = Qi_i * c2 + Qi_j * s2 + cs2 * Pij_ij
                Qjtild = Qi_i * s2 + Qi_j * c2 - cs2 * Pij_ij

                # Population change
                dL = (Qitild**exponent + Qjtild**exponent).sum(axis=1) - Lij
                Qitild = Qjtild = None

                # Find theta that increases the PM objective
                mask = dL > (dLij + mlo.conv_tol)
                if numpy.any(mask):
                    dLij[mask] = dL[mask]
                    thetaij[mask] = theta
                    Ridxij[mask] = iR

            Qi_i = Qi_j = Pij_ij = None

        idxs = numpy.where(dLij > mlo.conv_tol)[0]
        if idxs.size == 0:
            break

        # Sort idxs in decreasing order
        idxs = idxs[numpy.argsort(dLij[idxs])[::-1]]

        # Remove overlapping pairs using a greedy algorithm
        stable = False
        done = numpy.zeros(mlo.norb, dtype=bool)

        for idx in idxs:
            i, j = tril_idx[idx], tril_jdx[idx]
            if done[i] or done[j]:
                continue
            done[i] = done[j] = True

            theta = thetaij[idx]
            iR = Ridxij[idx]
            R = Rs[iR]
            log.info('Rotating orbital pair (%d,%d) by %.2f Pi for Rvec %s. delta_f= %.14g',
                     i, j, theta/numpy.pi, R, dLij[idx])

            update_rotation_local_(u, theta, i, j, phase[iR])

    if stable:
        log.info(f'{mlo.__class__.__name__} is stable in the Jacobi stability analysis')
        mo_coeff = mlo.mo_coeff
    else:
        mo_coeff = mlo.rotate_orb(u)

    if return_status:
        return mo_coeff, stable
    else:
        return mo_coeff
