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
# Author: Xubo Wang <wangxubo0201@gmail.com>
#

'''
Eigenvalue problems of Hermitian matrices with time-reversal (Kramers)
symmetry.

In a Kramers basis (phi_1, ..., phi_n, T phi_1, ..., T phi_n) such a matrix
has the quaternion structure

    H = [[A, B], [-B^*, A^*]],    A = A^+,  B = -B^T,

and the time reversal acts on a vector [p; q] as T [p; q] = [-q^*; p^*].
The eigenvalues are doubly degenerate.  The eigenvectors are returned in
Kramers pairs (u, Tu) in consecutive columns; the pairing is exact by
construction.

The standard problem is solved with zquatev (https://github.com/sunqm/zquatev)
if it is installed, otherwise with the quaternion Householder
tridiagonalization implemented here (blocked as LAPACK zhetrd/zlatrd, the
back-transformation in the compact WY form).  The generalized problem
HC = SCE is reduced to a standard one with the Cholesky factor of S, which
has the quaternion structure as well; when S is numerically not positive
definite, canonical orthogonalization is used instead.

The backend can be chosen with the argument ``backend`` or with
``lib_kramers_eigh_backend`` in the pyscf config ('auto', 'zquatev' or
'numpy').
'''

import numpy
import scipy.linalg
from pyscf import __config__

zquatev = None
try:
    # Install zquatev with
    # pip install git+https://github.com/sunqm/zquatev
    import zquatev
except ImportError:
    pass

BACKEND = getattr(__config__, 'lib_kramers_eigh_backend', 'auto')

# number of Householder steps per panel
BLOCK_SIZE = getattr(__config__, 'lib_kramers_eigh_block_size', 32)


def eigh(h, s=None, backend=None, lindep=1e-14, iop=0):
    '''Eigenvalues and eigenvectors of a Hermitian matrix with the quaternion
    structure, HC = CE, or of the generalized problem HC = SCE.

    Args:
        h : (2n, 2n) array
            Hermitian, in the Kramers basis (phi_1..phi_n, T phi_1..T phi_n).
        s : (2n, 2n) array, optional
            Hermitian positive (semi)definite, in the same basis.

    Kwargs:
        backend : str
            'zquatev', 'numpy' or 'auto' (zquatev when installed).  Default
            from the config ``lib_kramers_eigh_backend``.
        lindep : float
            For the generalized problem with an S that is not positive
            definite: eigenvectors of S with eigenvalues below
            lindep * max(eig(S)) are removed.
        iop : int
            Order of the eigenvectors, as in zquatev.eigh.  0: the Kramers
            basis order of the input, (u_1..u_m, Tu_1..Tu_m), so that the
            vectors can serve as a Kramers basis themselves.  1: pairs in
            consecutive columns, (u_1, Tu_1, u_2, Tu_2, ...).

    Returns:
        e : (2m,) array
            Eigenvalues, each one twice, ascending in u_k (m = n, or less
            when linearly dependent vectors of S were removed).
        c : (2n, 2m) array
            Eigenvectors, orthonormal (in the metric S for the generalized
            problem), Tu_k exactly the time reversal of u_k.
    '''
    h = numpy.asarray(h)
    n2 = h.shape[0]
    if n2 % 2:
        raise ValueError('Matrix dimension %d is not even' % n2)
    inter = _interleave_index(n2 // 2)
    hi = _symmetrize(h[inter[:, None], inter])
    if s is None:
        e, c = _eigh_standard(hi, backend)
    else:
        s = numpy.asarray(s)
        si = _symmetrize(s[inter[:, None], inter])
        e, c = _eigh_generalized(hi, si, backend, lindep)
    out = numpy.empty_like(c)
    out[inter] = c
    if iop == 0:
        e = numpy.hstack((e[0::2], e[1::2]))
        out = numpy.hstack((out[:,0::2], out[:,1::2]))
    elif iop != 1:
        raise ValueError('iop must be 0 or 1')
    return e, out


def _interleave_index(n):
    '''Block order (phi_1..phi_n, T phi_1..T phi_n) -> interleaved order
    (phi_1, T phi_1, phi_2, ...): interleaved index j is block index idx[j].'''
    return numpy.ravel(numpy.column_stack((numpy.arange(n), numpy.arange(n) + n)))


def _tr(c):
    '''Time reversal in the interleaved basis, (p_k, q_k) -> (-q_k^*, p_k^*).'''
    y = numpy.empty_like(c)
    y[0::2] = -c[1::2].conj()
    y[1::2] = c[0::2].conj()
    return y


def _symmetrize(h):
    '''The nearest Hermitian matrix with the quaternion structure, in the
    interleaved basis.'''
    h = numpy.asarray(h, dtype=numpy.complex128)
    a = .5 * (h[0::2, 0::2] + h[1::2, 1::2].conj())
    b = .5 * (h[0::2, 1::2] - h[1::2, 0::2].conj())
    a = .5 * (a + a.conj().T)
    b = .5 * (b - b.T)
    out = numpy.empty_like(h)
    out[0::2, 0::2] = a
    out[1::2, 1::2] = a.conj()
    out[0::2, 1::2] = b
    out[1::2, 0::2] = -b.conj()
    return out


def _get_backend(backend):
    if backend is None:
        backend = BACKEND
    if backend == 'auto':
        backend = 'numpy' if zquatev is None else 'zquatev'
    if backend == 'zquatev' and zquatev is None:
        raise RuntimeError('zquatev is not installed')
    if backend not in ('zquatev', 'numpy'):
        raise ValueError('Unknown backend %s' % backend)
    return backend


def _eigh_standard(h, backend=None):
    '''HC = CE, h in the interleaved basis (structured).'''
    if _get_backend(backend) == 'zquatev':
        n2 = h.shape[0]
        inter = _interleave_index(n2 // 2)
        hb = numpy.empty_like(h)
        hb[inter[:, None], inter] = h
        e, c = zquatev.eigh(hb, iop=1)
        return e, c[inter]
    return _eigh_householder(h)


def _eigh_generalized(h, s, backend=None, lindep=1e-14):
    '''HC = SCE, h and s in the interleaved basis (structured).'''
    try:
        l = scipy.linalg.cholesky(s, lower=True)
    except numpy.linalg.LinAlgError:
        l = None

    if l is not None:
        # The Cholesky factor has the quaternion structure: the diagonal 2x2
        # blocks of S and of its Schur complements are real multiples of the
        # unit matrix.  Impose it exactly.
        l[1::2, 1::2] = l[0::2, 0::2].conj()
        l[0::2, 1::2] = -l[1::2, 0::2].conj()
        tmp = scipy.linalg.solve_triangular(l, h, lower=True)
        hp = scipy.linalg.solve_triangular(l, tmp.conj().T, lower=True).conj().T
        e, cp = _eigh_standard(_symmetrize(hp), backend)
        c = numpy.empty_like(cp)
        c[:, 0::2] = scipy.linalg.solve_triangular(l, cp[:, 0::2], lower=True, trans='C')
    else:
        # canonical orthogonalization
        es, vs = _eigh_standard(s, backend)
        mask = es > lindep * es.max()
        x = vs[:, mask] / numpy.sqrt(es[mask])
        hx = x.conj().T.dot(h).dot(x)
        e, cp = _eigh_standard(_symmetrize(hx), backend)
        c = numpy.empty((x.shape[0], cp.shape[1]), dtype=numpy.complex128)
        c[:, 0::2] = x.dot(cp[:, 0::2])
    c[:, 1::2] = _tr(c[:, 0::2])
    return e, c


def _householder_vector(x):
    '''Quaternion Householder reflection W = 1 - 2 V V^+, V = [v, Tv], with
    W x = y, y nonzero only in its first quaternion element (x interleaved).
    Returns V and y, or None if x is zero.'''
    xnorm = numpy.linalg.norm(x)
    if xnorm == 0:
        return None
    a, b = x[0], x[1]
    x1 = numpy.sqrt(abs(a)**2 + abs(b)**2)
    # y = c (a, b) with c real, |y| = |x|; the sign avoids the cancellation
    y = numpy.zeros_like(x)
    if x1 > 0:
        y[0] = -xnorm / x1 * a
        y[1] = -xnorm / x1 * b
    else:
        y[0] = -xnorm
    v = x - y
    vnorm = numpy.linalg.norm(v)
    if vnorm == 0:
        return None
    v /= vnorm
    V = numpy.empty((x.size, 2), dtype=numpy.complex128)
    V[:, 0] = v
    V[:, 1] = _tr(v)
    return V, y


def _eigh_householder(h, nb=None):
    '''Eigenvalues and Kramers-paired eigenvectors of a structured Hermitian
    matrix in the interleaved basis.

    1. Tridiagonalization with quaternion Householder reflections, which
       commute with T; the updates of the trailing matrix are deferred over
       panels of nb steps and applied with one matrix product (as zlatrd).
    2. The quaternion tridiagonal matrix is made real by SU(2) phases.
    3. Real symmetric tridiagonal eigenvalue problem.
    4. Back-transformation in the compact WY form.
    '''
    if nb is None:
        nb = BLOCK_SIZE
    h = numpy.array(h, dtype=numpy.complex128, order='C', copy=True)
    n2 = h.shape[0]
    n = n2 // 2

    # Matrix products on the strided views h[r:, r:] use the @ operator, which
    # calls BLAS with the strides (ndarray.dot copies the view first).
    vs = []
    nsteps = n - 2
    for k0 in range(0, max(nsteps, 0), nb):
        k1 = min(k0 + nb, nsteps)
        Vp = numpy.zeros((n2, 2*(k1-k0)), dtype=numpy.complex128)
        Qp = numpy.zeros_like(Vp)
        for k in range(k0, k1):
            j = 2 * (k - k0)
            c0 = 2 * k
            r = c0 + 2
            if j:
                # columns (2k, 2k+1) with the pending updates of this panel
                col = h[c0:, c0:c0+2] - 2 * (Vp[c0:, :j] @ Qp[c0:c0+2, :j].conj().T
                                             + Qp[c0:, :j] @ Vp[c0:c0+2, :j].conj().T)
                h[c0:, c0:c0+2] = col
                h[c0:c0+2, c0:] = col.conj().T
            refl = _householder_vector(h[r:, c0].copy())
            if refl is None:
                vs.append(None)
                continue
            V, y = refl
            # p = H V for the current trailing matrix
            p = h[r:, r:] @ V
            if j:
                p -= 2 * (Vp[r:, :j] @ (Qp[r:, :j].conj().T @ V)
                          + Qp[r:, :j] @ (Vp[r:, :j].conj().T @ V))
            # W H W = H - 2 (V Q^+ + Q V^+),  Q = p - V V^+ p
            q = p - V @ (V.conj().T @ p)
            Vp[r:, j:j+2] = V
            Qp[r:, j:j+2] = q
            ty = _tr(y)
            h[r:, c0] = y
            h[r:, c0+1] = ty
            h[c0, r:] = y.conj()
            h[c0+1, r:] = ty.conj()
            vs.append(V)
        r = 2 * k1
        if r < n2:
            U = numpy.hstack((Vp[r:], Qp[r:]))
            Z = numpy.hstack((Qp[r:], Vp[r:]))
            h[r:, r:] -= 2 * (U @ Z.conj().T)

    # Real tridiagonal matrix by SU(2) phases D_k (D_0 = 1): with the
    # sub-diagonal quaternion blocks O_k, D_k+1 = O_k D_k / |O_k|.  O_k is built
    # from its first column so that D stays in SU(2) when |O_k| is tiny.
    d = h[numpy.arange(0, n2, 2), numpy.arange(0, n2, 2)].real
    off = numpy.empty(max(n-1, 0))
    D = numpy.empty((n, 2, 2), dtype=numpy.complex128)
    D[0] = numpy.eye(2)
    for k in range(n - 1):
        a, b = h[2*k+2, 2*k], h[2*k+3, 2*k]
        o = numpy.array([[a, -b.conjugate()], [b, a.conjugate()]])
        onorm = numpy.sqrt(abs(a)**2 + abs(b)**2)
        off[k] = onorm
        D[k+1] = o.dot(D[k]) / onorm if onorm > 0 else numpy.eye(2)
    if n == 1:
        e, z = d, numpy.ones((1, 1))
    else:
        # stev (implicit QL/QR): orthogonal eigenvectors also for the tight
        # clusters of degenerate eigenvalues, where stemr may lose orthogonality
        e, z = scipy.linalg.eigh_tridiagonal(d, off, lapack_driver='stev')

    # eigenvectors u_j = sum_k z_kj D_k[:,0]; the partners follow by T
    c = numpy.empty((n2, n), dtype=numpy.complex128)
    c[0::2] = D[:, 0, 0, None] * z
    c[1::2] = D[:, 1, 0, None] * z
    for j0 in reversed(range(0, len(vs), nb)):
        blk = vs[j0:j0+nb]
        r0 = 2 * j0 + 2
        m = 2 * len(blk)
        Y = numpy.zeros((n2 - r0, m), dtype=numpy.complex128)
        tau = numpy.zeros(m)
        for i, V in enumerate(blk):
            if V is not None:
                Y[2*i:, 2*i:2*i+2] = V
                tau[2*i:2*i+2] = 2
        # W_j0 ... W_j0+m/2-1 = 1 - Y T Y^+ (T upper triangular)
        yy = Y.conj().T @ Y
        T = numpy.zeros((m, m), dtype=numpy.complex128)
        for i in range(0, m, 2):
            T[i:i+2, i:i+2] = numpy.diag(tau[i:i+2])
            if i:
                T[:i, i:i+2] = -T[:i, :i] @ yy[:i, i:i+2] * tau[i:i+2]
        cs = c[r0:]
        cs -= Y @ (T @ (Y.conj().T @ cs))

    out = numpy.empty((n2, n2), dtype=numpy.complex128)
    out[:, 0::2] = c
    out[:, 1::2] = _tr(c)
    return numpy.repeat(e, 2), out
