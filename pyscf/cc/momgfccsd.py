#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Oliver Backhouse <olbackhouse@gmail.com>
#

"""
GF-CCSD solver via moment constraints.

See reference: Backhouse, Booth, arXiv:2206.13198 (2022).
"""

from collections import defaultdict

import numpy as np
import scipy.linalg

from pyscf import lib, cc, ao2mo
from pyscf.lib import logger
from pyscf.agf2 import mpi_helper


def kernel(
        gfccsd,
        hole_moments=None,
        part_moments=None,
        t1=None,
        t2=None,
        l1=None,
        l2=None,
        eris=None,
        imds=None,
        verbose=None,
):
    if gfccsd.verbose >= logger.WARN:
        gfccsd.check_sanity()
    gfccsd.dump_flags()

    log = logger.new_logger(gfccsd, verbose)

    if (l1 is None and gfccsd._cc.l1 is None) or (l2 is None and gfccsd._cc.l2 is None):
        raise ValueError(
                "Lambda amplitudes must be set for %s. This "
                "can be done by calling solve_lambda on the "
                "CC method or by setting l1, l2 attributes. "
                % gfccsd.__class__.__name__
        )

    if (hole_moments is None or part_moments is None) and imds is None:
        ip = hole_moments is None
        ea = part_moments is None
        imds = gfccsd.make_imds(eris=eris, ip=ip, ea=ea)

    if hole_moments is None:
        log.info("Building hole moments:")
        hole_moments = gfccsd.build_hole_moments(t1=t1, t2=t2, l1=l1, l2=l2, imds=imds)
    else:
        log.info("Hole moments passed by argument.")

    if part_moments is None:
        log.info("Building particle moments:")
        part_moments = gfccsd.build_part_moments(t1=t1, t2=t2, l1=l1, l2=l2, imds=imds)
    else:
        log.info("Particle moments passed by argument.")

    if gfccsd.hermi_moments:
        hole_moments = 0.5 * (hole_moments + hole_moments.swapaxes(1, 2).conj())
        part_moments = 0.5 * (part_moments + part_moments.swapaxes(1, 2).conj())

    if gfccsd.hermi_solver:
        solver = block_lanczos_symm
        eig = eigh_block_tridiagonal
    else:
        solver = block_lanczos_nosymm
        eig = eig_block_tridiagonal

    log.info("Solving for the hole moments.")
    blocks = solver(gfccsd, hole_moments)
    orth = mat_sqrt(hole_moments[0], hermi=gfccsd.hermi_solver)
    eh, vh = eig(gfccsd, *blocks, orth=orth)

    log.info("Solving for the particle moment.")
    blocks = solver(gfccsd, part_moments)
    orth = mat_sqrt(part_moments[0], hermi=gfccsd.hermi_solver)
    ep, vp = eig(gfccsd, *blocks, orth=orth)

    # Check the moments
    if gfccsd.niter[0] is not None:
        for n in range(2*gfccsd.niter[0]+2):
            a = lib.einsum("xk,yk,k->xy", vh[0], vh[1].conj(), eh**n)
            a /= np.max(np.abs(a))
            b = hole_moments[n] / np.max(np.abs(hole_moments[n]))
            err = np.max(np.abs(a - b))
            (logger.debug1 if err < 1e-8 else logger.warn)(
                    gfccsd, "Error in hole moment %d:  %10.6g", n, err)
    if gfccsd.niter[0] is not None:
        for n in range(2*gfccsd.niter[1]+2):
            a = lib.einsum("xk,yk,k->xy", vp[0], vp[1].conj(), ep**n)
            a /= np.max(np.abs(a))
            b = part_moments[n] / np.max(np.abs(part_moments[n]))
            err = np.max(np.abs(a - b))
            (logger.debug1 if err < 1e-8 else logger.warn)(
                    gfccsd, "Error in particle moment %d:  %10.6g", n, err)

    mask = np.argsort(eh.real)
    eh, vh = eh[mask], (vh[0][:, mask], vh[1][:, mask])
    mask = np.argsort(ep.real)
    ep, vp = ep[mask], (vp[0][:, mask], vp[1][:, mask])

    return eh, vh, ep, vp


def mat_sqrt(m, hermi=False):
    """Return the square root of a matrix.
    """

    if hermi:
        w, v = np.linalg.eigh(m)
        mask = w >= 0
        w, v = w[mask], v[:, mask]
        out = np.dot(v * w[None]**0.5, v.T.conj())

    else:
        w, v = np.linalg.eig(m)
        out = np.dot(v * w[None]**(0.5+0j), np.linalg.inv(v))

    return out


def mat_isqrt(m, tol=1e-16, hermi=False):
    """Return the inverse square root of a matrix.
    """

    if hermi:
        w, v = np.linalg.eigh(m)
        mask = w > tol
        w, v = w[mask], v[:, mask]
        out = np.dot(v * w[None]**-0.5, v.T.conj())

    else:
        w, v = np.linalg.eig(m)
        mask = np.abs(w) >= tol
        vinv = np.linalg.inv(v)[mask]
        w, v = w[mask], v[:, mask]
        out = np.dot(v * w[None]**(-0.5+0j), vinv)

    return out


def build_block_tridiagonal(a, b, c=None):
    """Construct a block tridiagonal matrix from a list of on-diagonal
    and off-diagonal blocks.
    """

    z = np.zeros_like(a[0], dtype=a[0].dtype)

    if c is None:
        c = [x.T.conj() for x in b]

    h = np.block([[
        a[i] if i == j else
        b[j] if j == i-1 else
        c[i] if i == j-1 else z
        for j in range(len(a))]
        for i in range(len(a))]
    )

    return h


def eig_block_tridiagonal(gfccsd, a, b, c, orth=None):
    """Diagonalise a non-Hermitian block-tridiagonal Hamiltonian and
    transform its eigenvectors appropriately.
    """

    h_tri = build_block_tridiagonal(a, b, c)

    e, u = np.linalg.eig(h_tri)

    if orth is not None:
        vl = np.dot(orth, u[:gfccsd.nmo])
        vr = np.dot(np.linalg.inv(u)[:, :gfccsd.nmo], orth).T.conj()
    else:
        vl = u[:gfccsd.nmo]
        vr = np.linalg.inv(u)[:, :gfccsd.nmo].T.conj()

    return e, (vl, vr)


def eigh_block_tridiagonal(gfccsd, a, b, orth=None):
    """Diagonalise a Hermitian block-tridiagonal Hamiltonian and
    transform its eigenvectors appropriately.
    """

    h_tri = build_block_tridiagonal(a, b)

    e, u = np.linalg.eigh(h_tri)

    if orth is not None:
        v = np.dot(orth, u[:gfccsd.nmo])
    else:
        v = u[:gfccsd.nmo]

    return e, (v, v)


def _matrix_info(x, hermi=False):
    norm = np.abs(np.einsum("pq,qp->", x, x))
    eigvals = np.linalg.eigvals(x)
    mineig = np.min(np.abs(eigvals))
    maxeig = np.max(np.abs(eigvals))
    return norm, mineig, maxeig


def block_lanczos_symm(gfccsd, moments, verbose=None):
    """Hermitian block Lanczos solver, returns a set of poles that
    best reproduce the inputted moments.

    Args:
        gfccsd : MomGFCCSD
            GF-CCSD object
        moments : ndarray (2*niter+2, n, n)
            Array of moments with which the resulting poles should
            be consistent with.

    Kwargs:
        verbose : int
            Level of verbosity.

    Returns:
        a : ndarray (niter+1, n, n)
            On-diagonal blocks of the block tridiagonal Hamiltonian.
        b : ndarray (niter, n, n)
            Off-diagonal blocks of the block tridiagonal Hamiltonian.
    """

    log = logger.new_logger(gfccsd, verbose)
    log.debug1("block_lanczos_symm: %d moments", len(moments))

    nmo = gfccsd.nmo
    niter = (len(moments) - 2) // 2
    dtype = np.complex128

    a = np.zeros((niter+1, nmo, nmo), dtype=dtype)
    b = np.zeros((niter, nmo, nmo), dtype=dtype)
    t = np.zeros((len(moments), nmo, nmo), dtype=dtype)

    v = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
    v[0, 0] = np.eye(nmo).astype(dtype)

    orth = mat_isqrt(moments[0], hermi=True)
    for i in range(len(moments)):
        t[i] = np.linalg.multi_dot((orth, moments[i], orth))

    a[0] = t[1]

    log.debug1("Raw moments:")
    log.debug1("  %4s %12s %12s %12s", "N", "norm", "min(|eig|)", "max(|eig|)")
    for i in range(len(moments)):
        log.debug1("  %4d %12.6g %12.6g %12.6g", i, *_matrix_info(moments[i], hermi=True))

    log.debug1("Orthogonalised moments:")
    log.debug1("  %4s %12s %12s %12s", "N", "norm", "min(|eig|)", "max(|eig|)")
    for i in range(len(moments)):
        log.debug1("  %4d %12.6g %12.6g %12.6g", i, *_matrix_info(t[i], hermi=True))

    for i in range(niter):
        log.info("Iteration %d", i)

        b2 = np.zeros((nmo, nmo), dtype=dtype)
        for j in range(i+2):
            for l in range(i+1):
                b2 += np.linalg.multi_dot((v[i, l].T.conj(), t[j+l+1], v[i, j-1]))

        b2 -= np.dot(a[i], a[i])
        if i:
            b2 -= np.dot(b[i-1], b[i-1])

        b[i] = mat_sqrt(b2, hermi=True)
        binv = mat_isqrt(b2, hermi=True)

        for j in range(i+2):
            r = (
                    + v[i, j-1]
                    - np.dot(v[i, j], a[i])
                    - np.dot(v[i-1, j], b[i-1])
            )
            v[i+1, j] = np.dot(r, binv)

        for j in range(i+2):
            for l in range(i+2):
                a[i+1] += np.linalg.multi_dot((v[i+1, l].T.conj(), t[j+l+1], v[i+1, j]))

        log.debug1("  %4s %12s %12s %12s", "mat", "norm", "min(|eig|)", "max(|eig|)")
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B^2", *_matrix_info(b2))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B", *_matrix_info(b[i]))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B^-1", *_matrix_info(binv))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "A", *_matrix_info(a[i+1]))

        biorth_error = 0.0
        for j in range(i+2):
            x = np.zeros_like(v[0, 0])
            for k in range(i+2):
                for l in range(i+2):
                    x += np.linalg.multi_dot((v[i+1, l].T.conj(), t[k+l], v[j, k]))
            biorth_error = max(biorth_error, np.max(np.abs(x - np.eye(nmo)*((i+1)==j))))
        log.info("  Error in biorthogonality:  %12.6g", biorth_error)

    return a, b


def block_lanczos_nosymm(gfccsd, moments, verbose=None):
    """Non-Hermitian block Lanczos solver, returns a set of poles that
    best reproduce the inputted moments.

    Args:
        gfccsd : MomGFCCSD
            GF-CCSD object
        moments : ndarray (2*niter+2, n, n)
            Array of moments with which the resulting poles should
            be consistent with.

    Kwargs:
        verbose : int
            Level of verbosity.

    Returns:
        a : ndarray (niter+1, n, n)
            On-diagonal blocks of the block tridiagonal Hamiltonian.
        b : ndarray (niter, n, n)
            Upper off-diagonal blocks of the block tridiagonal
            Hamiltonian.
        c : ndarray (niter, n, n)
            Lower off-diagonal blocks of the block tridiagonal
            Hamiltonian.
    """

    log = logger.new_logger(gfccsd, verbose)
    log.debug1("block_lanczos_nosymm: %d moments", len(moments))

    nmo = gfccsd.nmo
    niter = (len(moments) - 2) // 2
    dtype = np.complex128

    a = np.zeros((niter+1, nmo, nmo), dtype=dtype)
    b = np.zeros((niter, nmo, nmo), dtype=dtype)
    c = np.zeros((niter, nmo, nmo), dtype=dtype)
    t = np.zeros((len(moments), nmo, nmo), dtype=dtype)

    v = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
    w = defaultdict(lambda: np.zeros((nmo, nmo), dtype=dtype))
    v[0, 0] = np.eye(nmo).astype(dtype)
    w[0, 0] = np.eye(nmo).astype(dtype)

    orth = mat_isqrt(moments[0])
    for i in range(len(moments)):
        t[i] = np.linalg.multi_dot((orth, moments[i], orth))

    a[0] = t[1]

    log.debug1("Raw moments:")
    log.debug1("  %4s %12s %12s %12s", "N", "norm", "min(|eig|)", "max(|eig|)")
    for i in range(len(moments)):
        log.debug1("  %4d %12.6g %12.6g %12.6g", i, *_matrix_info(moments[i]))

    log.debug1("Orthogonalised moments:")
    log.debug1("  %4s %12s %12s %12s", "N", "norm", "min(|eig|)", "max(|eig|)")
    for i in range(len(moments)):
        log.debug1("  %4d %12.6g %12.6g %12.6g", i, *_matrix_info(t[i]))

    for i in range(niter):
        log.info("Iteration %d", i)

        b2 = np.zeros((nmo, nmo), dtype=dtype)
        c2 = np.zeros((nmo, nmo), dtype=dtype)

        for j in range(i+2):
            for l in range(i+1):
                b2 += np.linalg.multi_dot((w[i, l], t[j+l+1], v[i, j-1]))
                c2 += np.linalg.multi_dot((w[i, j-1], t[j+l+1], v[i, l]))

        b2 -= np.dot(a[i], a[i])
        c2 -= np.dot(a[i], a[i])
        if i:
            b2 -= np.dot(c[i-1], c[i-1])
            c2 -= np.dot(b[i-1], b[i-1])

        b[i] = mat_sqrt(b2)
        c[i] = mat_sqrt(c2)

        binv = mat_isqrt(b2)
        cinv = mat_isqrt(c2)

        for j in range(i+2):
            r = (
                    + v[i, j-1]
                    - np.dot(v[i, j], a[i])
                    - np.dot(v[i-1, j], b[i-1])
            )
            v[i+1, j] = np.dot(r, cinv)

            s = (
                    + w[i, j-1]
                    - np.dot(a[i], w[i, j])
                    - np.dot(c[i-1], w[i-1, j])
            )
            w[i+1, j] = np.dot(binv, s)

        for j in range(i+2):
            for l in range(i+2):
                a[i+1] += np.linalg.multi_dot((w[i+1, l], t[j+l+1], v[i+1, j]))

        log.debug1("  %4s %12s %12s %12s", "mat", "norm", "min(|eig|)", "max(|eig|)")
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B^2", *_matrix_info(b2))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B", *_matrix_info(b[i]))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "B^-1", *_matrix_info(binv))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "C^2", *_matrix_info(c2))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "C", *_matrix_info(c[i]))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "C^-1", *_matrix_info(cinv))
        log.debug1("  %4s %12.6g %12.6g %12.6g", "A", *_matrix_info(a[i+1]))

        biorth_error = 0.0
        for j in range(i+2):
            x = np.zeros_like(v[0, 0])
            y = np.zeros_like(v[0, 0])
            for k in range(i+2):
                for l in range(i+2):
                    x += np.linalg.multi_dot((w[i+1, l], t[k+l], v[j, k]))
                    y += np.linalg.multi_dot((w[j, l], t[k+l], v[i+1, k]))
            biorth_error = max(biorth_error, np.max(np.abs(x - np.eye(nmo)*((i+1)==j))))
            biorth_error = max(biorth_error, np.max(np.abs(y - np.eye(nmo)*((i+1)==j))))
        log.info("  Error in biorthogonality:  %12.6g", biorth_error)

    return a, b, c


def _kd(n, i):
    v = np.zeros((n,))
    v[i] = 1.0
    return v


def contract_ket_hole(gfccsd, eom, t1, t2, v, orb):
    r"""Contract a vector with \bar{a}^\dagger_p |\Psi>.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        return v[orb]
    else:
        b1 = t1[:, orb-nocc]
        b2 = t2[:, :, orb-nocc]
        b = eom.amplitudes_to_vector(b1, b2)
        return np.dot(v, b)


def build_ket_hole(gfccsd, eom, t1, t2, orb):
    r"""Build \bar{a}^\dagger_p |\Psi>.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        b1 = np.eye(nocc)[orb]
        b2 = np.zeros((nocc, nocc, nvir))
    else:
        b1 = t1[:, orb-nocc]
        b2 = t2[:, :, orb-nocc]

    return eom.amplitudes_to_vector(b1, b2)


def build_bra_hole(gfccsd, eom, t1, t2, l1, l2, orb):
    """Get the first- and second-order contributions to the left-hand
    transformed vector for a given orbital for the hole part of the
    Green's function.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        e1 = _kd(nocc, orb)
        e1 -= lib.einsum("ie,e->i", l1, t1[orb])
        tmp = t2[orb] * 2.0
        tmp -= t2[orb].swapaxes(1, 2)
        e1 -= lib.einsum("imef,mef->i", l2, tmp)

        tmp = -lib.einsum("ijea,e->ija", l2, t1[orb])
        e2 = 2.0 * tmp
        e2 -= tmp.swapaxes(0, 1)
        tmp = lib.einsum("ja,i->ija", l1, _kd(nocc, orb))
        e2 += tmp * 2.0
        e2 -= tmp.swapaxes(0, 1)

    else:
        e1 = l1[:, orb-nocc].copy()
        e2 = l2[:, :, orb-nocc] * 2.0
        e2 -= l2[:, :, :, orb-nocc]

    return eom.amplitudes_to_vector(e1, e2)


def contract_ket_part(gfccsd, eom, t1, t2, v, orb):
    r"""Contract a vector with \bar{a}_p |\Psi>.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        b1 = t1[orb]
        b2 = t2[orb]
        b = eom.amplitudes_to_vector(b1, b2)
        return np.dot(v, b)
    else:
        return -v[orb-nocc]


def build_ket_part(gfccsd, eom, t1, t2, orb):
    r"""Build \bar{a}_p |\Psi>.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        b1 = t1[orb]
        b2 = t2[orb]
    else:
        b1 = -np.eye(nvir)[orb-nocc]
        b2 = np.zeros((nocc, nvir, nvir))

    return eom.amplitudes_to_vector(b1, b2)


def build_bra_part(gfccsd, eom, t1, t2, l1, l2, orb):
    """Get the first- and second-order contributions to the left-hand
    transformed vector for a given orbital for the particle part of the
    Green's function.
    """

    nocc, nvir = t1.shape

    if orb < nocc:
        e1 = -l1[orb]
        e2 = -l2[orb] * 2.0
        e2 += l2[:, orb]

    else:
        e1 = _kd(nvir, orb-nocc)
        e1 -= lib.einsum("mb,m->b", l1, t1[:, orb-nocc])
        tmp = t2[:, :, :, orb-nocc] * 2.0
        tmp -= t2[:, :, orb-nocc]
        e1 -= lib.einsum("kmeb,kme->b", l2, tmp)

        tmp = -lib.einsum("ikba,k->iab", l2, t1[:, orb-nocc])
        e2 = tmp * 2.0
        e2 -= tmp.swapaxes(1, 2)
        tmp = lib.einsum("ib,a->iab", l1, _kd(nvir, orb-nocc))
        e2 += tmp * 2.0
        e2 -= tmp.swapaxes(1, 2)

    return eom.amplitudes_to_vector(e1, e2)


class MomGFCCSD(lib.StreamObject):
    """Green's function coupled cluster singles and doubles using the
    moment-resolved solver.

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`.
        niter : tuple of (int, int)
            Number of block Lanczos iterations for occupied and virtual
            sectors. If either are `None` then said sector will not be
            computed.
        weight_tol : float
            Threshold for weight in the physical space to consider a
            pole an ionisation or removal event. Default value is 1e-1.
        hermi_moments : bool
            Whether to Hermitise the moments, default value is False.
        hermi_solver : obol
            Whether to use the real-valued, symmetric block Lanczos
            solver, default value is False.

    Results:
        eh : ndarray
            Energies of the compressed hole Green's function
        vh : tuple of ndarray
            Left- and right-hand transition amplitudes of the compressed
            hole Green's function
        ep : ndarray
            Energies of the compressed particle Green's function
        vp : tuple of ndarray
            Left- and right-hand transition amplitudes of the compressed
            particle Green's function
    """

    _keys = {
        'verbose', 'stdout', 'niter', 'weight_tol',
        'hermi_moments', 'hermi_solver', 'eh', 'ep', 'vh', 'vp', 'chkfile',
    }

    def __init__(self, mycc, niter=(2, 2)):
        self._cc = mycc
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout

        if isinstance(mycc, cc.uccsd.UCCSD):
            raise NotImplementedError("MomGFCCSD for unrestricted CCSD")

        if isinstance(niter, int):
            self.niter = (niter, niter)
        else:
            self.niter = niter
        self.weight_tol = 1e-1
        self.hermi_moments = False
        self.hermi_solver = False
        self.eh = None
        self.ep = None
        self.vh = None
        self.vp = None
        self._t1 = None
        self._t2 = None
        self._l1 = None
        self._l2 = None
        self.chkfile = self._cc.chkfile

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("niter = %s", self.niter)
        log.info("nmo = %s", self.nmo)
        log.info("nocc = %s", self.nocc)
        log.info("weight_tol = %s", self.weight_tol)
        log.info("hermi_moments = %s", self.hermi_moments)
        log.info("hermi_solver = %s", self.hermi_solver)
        log.info("chkfile = %s", self.chkfile)

    def _finalize(self):
        self.ipgfccsd()
        self.eagfccsd()
        return self

    def reset(self, mol=None):
        self._cc.reset(mol)
        return self

    @property
    def eomip_method(self):
        return self._cc.eomip_method()

    @property
    def eomea_method(self):
        return self._cc.eomea_method()

    build_bra_hole = build_bra_hole
    build_bra_part = build_bra_part
    contract_ket_hole = contract_ket_hole
    contract_ket_part = contract_ket_part

    def make_imds(self, eris=None, ip=True, ea=True):
        """Build EOM intermediates.
        """

        imds = cc.eom_rccsd._IMDS(self._cc, eris=eris)

        if ip:
            imds.make_ip()
        if ea:
            imds.make_ea()

        return imds

    def build_hole_moments(self, t1=None, t2=None, l1=None, l2=None, imds=None, niter=None):
        """Build moments of the hole (IP-EOM-CCSD) Green's function.
        """

        if t1 is None:
            t1 = self._cc.t1
        if t2 is None:
            t2 = self._cc.t2
        if l1 is None:
            l1 = self._cc.l1
        if l2 is None:
            l2 = self._cc.l2

        if niter is None:
            niter = self.niter[0]
        nmom = 2 * niter + 2
        moments = np.zeros((nmom, self.nmo, self.nmo))

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomip_method()
        if imds is None:
            imds = self.make_imds(ea=False)
        diag = eom.get_diag(imds)

        for p in mpi_helper.nrange(self.nmo):
            ket = self.build_bra_hole(eom, t1, t2, l1, l2, p)
            for n in range(nmom):
                for q in range(self.nmo):
                    moments[n, q, p] += self.contract_ket_hole(eom, t1, t2, ket, q)
                if (n+1) != nmom:
                    ket = -eom.l_matvec(ket, imds, diag)

        mpi_helper.barrier()
        moments = mpi_helper.allreduce(moments)

        logger.timer(self, "IP-EOM-CCSD moments", *cput0)

        return moments

    def build_part_moments(self, t1=None, t2=None, l1=None, l2=None, imds=None, niter=None):
        """Build moments of the particle (EA-EOM-CCSD) Green's function.
        """

        if t1 is None:
            t1 = self._cc.t1
        if t2 is None:
            t2 = self._cc.t2
        if l1 is None:
            l1 = self._cc.l1
        if l2 is None:
            l2 = self._cc.l2

        if niter is None:
            niter = self.niter[1]
        nmom = 2 * niter + 2
        moments = np.zeros((nmom, self.nmo, self.nmo))

        cput0 = (logger.process_clock(), logger.perf_counter())

        eom = self.eomea_method()
        if imds is None:
            imds = self.make_imds(ip=False)
        diag = eom.get_diag(imds)

        for p in mpi_helper.nrange(self.nmo):
            ket = self.build_bra_part(eom, t1, t2, l1, l2, p)
            for n in range(nmom):
                for q in range(self.nmo):
                    moments[n, q, p] -= self.contract_ket_part(eom, t1, t2, ket, q)
                if (n+1) != nmom:
                    ket = eom.l_matvec(ket, imds, diag)

        mpi_helper.barrier()
        moments = mpi_helper.allreduce(moments)

        logger.timer(self, "EA-EOM-CCSD moments", *cput0)

        return moments

    def make_rdm1(self, ao_repr=False, eris=None, imds=None):
        """Build the first-order reduced density matrix at the CCSD
        level using the zeroth-order moment of the hole part of the
        CCSD Green's function.
        """

        if imds is None:
            imds = self.make_imds(eris=eris, ea=False)

        dm1 = self.build_hole_moments(imds=imds, niter=0)[0]
        dm1 = dm1 + dm1.T.conj()

        if ao_repr:
            mo = self._cc.mo_coeff
            dm1 = np.linalg.multi_dot((mo, dm1, mo.T.conj()))

        return dm1

    def kernel(self, **kwargs):
        eh, vh, ep, vp = kernel(self, **kwargs)

        self.eh = eh
        self.vh = vh
        self.ep = ep
        self.vp = vp

        self._finalize()

        return eh, vh, ep, vp

    def dump_chk(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile

        lib.chkfile.dump(chkfile, key+"/eh", self.eh)
        lib.chkfile.dump(chkfile, key+"/vh_left", self.vh[0])
        lib.chkfile.dump(chkfile, key+"/vh_right", self.vh[1])
        lib.chkfile.dump(chkfile, key+"/ep", self.ep)
        lib.chkfile.dump(chkfile, key+"/vp_left", self.vp[0])
        lib.chkfile.dump(chkfile, key+"/vp_right", self.vp[1])
        lib.chkfile.dump(chkfile, key+"/niter", np.array(self.niter))

        return self

    def update_from_chk_(self, chkfile=None, key="gfccsd"):
        if chkfile is None:
            chkfile = self.chkfile

        self.eh = lib.chkfile.load(chkfile, key+"/eh")
        self.vh = (
                lib.chkfile.load(chkfile, key+"/vh_left"),
                lib.chkfile.load(chkfile, key+"/vh_right"),
        )
        self.ep = lib.chkfile.load(chkfile, key+"/ep")
        self.vp = (
                lib.chkfile.load(chkfile, key+"/vp_left"),
                lib.chkfile.load(chkfile, key+"/vp_right"),
        )
        self.niter = tuple(lib.chkfile.load(chkfile, key+"/niter"))

    update = update_from_chk = update_from_chk_

    def ipgfccsd(self, nroots=5):
        """Print and return ionisation potentials.
        """

        eh, (vh, uh) = self.eh, self.vh

        mask = np.abs(np.sum(vh * uh.conj(), axis=0)) > self.weight_tol
        mask = np.arange(mask.size)[mask][::-1]
        e_ip = -eh[mask]
        v_ip, u_ip = vh[:, mask], uh[:, mask]

        nroots = min(nroots, len(e_ip))
        logger.note(self, "  %s %s %16s %10s", "", "", "Energy", "Weight")
        for n in range(nroots):
            qpwt = np.abs(np.sum(v_ip[:, n] * u_ip[:, n].conj())).real
            warn = ""
            if np.abs(e_ip[n].imag) > 1e-8:
                warn += "(Warning: imag part: %.6g)" % e_ip[n].imag
            logger.note(self, "  %2s %2d %16.10g %10.6g %s" % ("IP", n, e_ip[n].real, qpwt, warn))

        if nroots == 1:
            return e_ip[0].real, v_ip[:, 0], u_ip[:, 0]
        else:
            return e_ip.real, v_ip, u_ip

    def eagfccsd(self, nroots=5):
        """Print and return electron affinities.
        """

        ep, (vp, up) = self.ep, self.vp

        mask = np.abs(np.sum(vp * up.conj(), axis=0)) > self.weight_tol
        e_ea = ep[mask]
        v_ea, u_ea = vp[:, mask], up[:, mask]

        nroots = min(nroots, len(e_ea))
        logger.note(self, "  %s %s %16s %10s", "", "", "Energy", "Weight")
        for n in range(nroots):
            qpwt = np.abs(np.sum(v_ea[:, n] * u_ea[:, n].conj())).real
            warn = ""
            if np.abs(e_ea[n].imag) > 1e-8:
                warn += "(Warning: imag part: %.6g)" % e_ea[n].imag
            logger.note(self, "  %2s %2d %16.10g %10.6g %s" % ("EA", n, e_ea[n].real, qpwt, warn))

        if nroots == 1:
            return e_ea[0].real, v_ea[:, 0], u_ea[:, 0]
        else:
            return e_ea.real, v_ea, u_ea

    @property
    def nmo(self):
        return self._cc.nmo

    @property
    def nocc(self):
        return self._cc.nocc


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(
            #atom="O 0 0 0; O 0 0 1",
            atom="N 0 0 0; N 0 0 1",
            basis="cc-pvdz",
            verbose=0,
    )
    mf = scf.RHF(mol)
    mf = mf.run()
    ccsd = cc.CCSD(mf)
    ccsd = ccsd.run()
    ccsd.solve_lambda()

    niter = 5

    gfcc = MomGFCCSD(ccsd, (niter, niter))
    gfcc.kernel()

    ip1, vip1 = ccsd.ipccsd(nroots=8)
    ip2, vip2, uip2 = gfcc.ipgfccsd(nroots=8)

    ea1, vea1 = ccsd.eaccsd(nroots=8)
    ea2, vea2, uea2 = gfcc.eagfccsd(nroots=8)

    print("    %12s %12s %12s" % ("EOM", "GF", "Error"))
    print("IP1 %12.8f %12.8f %12.8f" % (ip1[0],ip2[0],np.abs(ip1[0]-ip2[0])))
    print("IP2 %12.8f %12.8f %12.8f" % (ip1[1],ip2[1],np.abs(ip1[1]-ip2[1])))
    print("EA1 %12.8f %12.8f %12.8f" % (ea1[0],ea2[0],np.abs(ea1[0]-ea2[0])))
    print("EA2 %12.8f %12.8f %12.8f" % (ea1[1],ea2[1],np.abs(ea1[1]-ea2[1])))
