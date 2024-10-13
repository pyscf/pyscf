#!/usr/bin/env python
# Copyright 2024 The PySCF Developers. All Rights Reserved.
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

import sys
import numpy as np
import scipy.linalg
from pyscf.lib.parameters import MAX_MEMORY
from pyscf.lib import logger
from pyscf.lib.linalg_helper import _sort_elast, _outprod_to_subspace

# Add at most 20 states in each iteration
MAX_SPACE_INC = 20

def eigh(aop, x0, precond, tol_residual=1e-5, lindep=1e-12, nroots=1,
         x0sym=None, pick=None, max_cycle=50, max_memory=MAX_MEMORY,
         verbose=logger.WARN):
    '''
    Solve symmetric eigenvalues.

    This solver is similar to the `linalg_helper.davidson` solver, with
    optimizations for performance and wavefunction symmetry, specifically
    tailored for linear response methods.

    Args:
        aop : function(x) => array_like_x
            The matrix-vector product operator.
        x0 : 1D array or a list of 1D array
            Initial guess.
        precond : function(dx, e) => array_like_dx
            Preconditioner to generate new trial vector. The argument dx is a
            residual vector ``A*x0-e*x0``; e is the eigenvalue.

    Kwargs:
        tol_residual : float
            Convergence tolerance for the norm of residual vector ``A*x0-e*x0``.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        nroots : int
            Number of eigenvalues to be computed.
        x0sym:
            The symmetry label for each initial guess vectors.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        max_cycle : int
            max number of iterations.
        max_memory : int or float
            Allowed memory in MB.

    Returns:
        e : list of floats
            Eigenvalues.
        c : list of 1D arrays
            Eigenvectors.
    '''

    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)

    space_inc = nroots if MAX_SPACE_INC is None else MAX_SPACE_INC
    x0_size = x0.shape[1]
    max_space = int(max_memory*1e6/8/x0_size / 2 - nroots - space_inc)
    if max_space < nroots * 2 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eigh')
    max_space = max(max_space, nroots * 2)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    xs = np.zeros((0, x0_size))
    ax = np.zeros((0, x0_size))
    e = w = v = None
    conv_last = conv = np.zeros(nroots, dtype=bool)
    xt = x0

    if x0sym is not None:
        xt_ir = np.asarray(x0sym)
        xs_ir = np.array([], dtype=xt_ir.dtype)

    for icyc in range(max_cycle):
        xt, xt_idx = _qr(xt, lindep)
        # Generate at most space_inc trial vectors
        xt = xt[:space_inc]
        xt_idx = xt_idx[:space_inc]

        row0 = len(xs)
        axt = aop(xt)
        xs = np.vstack([xs, xt])
        ax = np.vstack([ax, axt])
        if x0sym is not None:
            xs_ir = np.hstack([xs_ir, xt_ir[xt_idx]])

        # Compute heff = xs.conj().dot(ax.T)
        if w is None:
            heff = xs.conj().dot(ax.T)
        else:
            hsub = xt.conj().dot(ax.T)
            heff = np.block([[np.diag(w), hsub[:,:row0].conj().T],
                             [hsub[:,:row0], hsub[:,row0:]]])

        if x0sym is None:
            w, v = scipy.linalg.eigh(heff)
        else:
            # Diagonalize within eash symmetry sectors
            row1 = len(xs)
            w = np.empty(row1)
            v = np.zeros((row1, row1))
            v_ir = []
            i1 = 0
            for ir in set(xs_ir):
                idx = np.where(xs_ir == ir)[0]
                i0, i1 = i1, i1 + idx.size
                w_sub, v_sub = scipy.linalg.eigh(heff[idx[:,None],idx])
                w[i0:i1] = w_sub
                v[idx,i0:i1] = v_sub
                v_ir.append([ir] * idx.size)
            w_idx = np.argsort(w)
            w = w[w_idx]
            v = v[:,w_idx]
            xs_ir = np.hstack(v_ir)[w_idx]

        w, v, idx = pick(w, v, nroots, locals())
        if x0sym is not None:
            xs_ir = xs_ir[idx]
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        e, elast = w[:nroots], e
        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            # mapping to previous eigenvectors
            vlast = np.eye(nroots)
            elast, conv_last = _sort_elast(elast, conv, vlast,
                                           v[:nroots,:nroots], log)
            de = e - elast

        xs = v.T.dot(xs)
        ax = v.T.dot(ax)
        if len(xs) * 2 > max_space:
            row0 = max(nroots, max_space-space_inc)
            xs = xs[:row0]
            ax = ax[:row0]
            w = w[:row0]
            if x0sym is not None:
                xs_ir = xs_ir[:row0]

        t_size = max(nroots, max_space-len(xs))
        xt = -w[:t_size,None] * xs[:t_size]
        xt += ax[:t_size]
        if x0sym is not None:
            xt_ir = xs_ir[:t_size]

        dx_norm = np.linalg.norm(xt, axis=1)
        max_dx_norm = max(dx_norm[:nroots])
        conv = dx_norm[:nroots] < tol_residual
        for k, ek in enumerate(e[:nroots]):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, len(xs), max_dx_norm, e, de[ide])
            break

        # remove subspace linear dependency
        for k, xk in enumerate(xt):
            if dx_norm[k] > tol_residual:
                xt[k] = precond(xk, e[0])
        xt -= xs.conj().dot(xt.T).T.dot(xs)
        xt_norm = np.linalg.norm(xt, axis=1)

        remaining = []
        for k, xk in enumerate(xt):
            if dx_norm[k] > tol_residual and xt_norm[k]**2 > lindep:
                xt[k] /= xt_norm[k]
                remaining.append(k)
        if len(remaining) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {dx_norm}')
            break

        xt = xt[remaining]
        if x0sym is not None:
            xt_ir = xt_ir[remaining]
        norm_min = xt_norm[remaining].min()
        log.debug('davidson %d %d |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, len(xs), max_dx_norm, e, de[ide], norm_min)

    x0 = xs[:nroots]
    # Check whether the solver finds enough eigenvectors.
    if len(x0) < min(x0_size, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return conv, e, x0

def eig(aop, x0, precond, tol_residual=1e-5, nroots=1, x0sym=None, pick=None,
        max_cycle=50, max_memory=MAX_MEMORY, lindep=1e-12, verbose=logger.WARN):
    '''
    Solver for linear response eigenvalues
    [ A    B] [X] = w [X]
    [-B* -A*] [Y]     [Y]

    subject to normalization X^2 - Y^2 = 1

    Reference:
      Olsen, Jensen, and Jorgenson, J Comput Phys, 74, 265,
      DOI: 10.1016/0021-9991(88)90081-2

    Args:
        aop : function(x) => array_like_x
            The matrix-vector product operator.
        x0 : 1D array or a list of 1D array
            Initial guess.
        precond : function(dx, e) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``A*x0-e*x0``; e is the eigenvalue.

    Kwargs:
        tol_residual : float
            Convergence tolerance for the norm of residual vector ``A*x0-e*x0``.
        lindep : float
            Linear dependency threshold.
        nroots : int
            Number of eigenvalues to be computed.
        x0sym:
            The symmetry label for each initial guess vectors.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        max_cycle : int
            max number of iterations.
        max_memory : int or float
            Allowed memory in MB.

    Returns:
        e : list of floats
            Eigenvalues.
        c : list of 1D arrays
            Eigenvectors.
    '''

    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)
    space_inc = nroots + 2

    x0_size = x0.shape[1]
    max_space = int(max_memory*1e6/8/(2*x0_size) / 2 - space_inc)
    if max_space < nroots * 2 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eig')
        max_space = space_inc * 2
    max_space = max(max_space, nroots * 2)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}')

    heff = None
    e = None
    v = None
    conv_last = conv = np.zeros(nroots, dtype=bool)

    if x0sym is not None:
        x0_ir = np.asarray(x0sym)

    half_size = x0[0].size // 2
    fresh_start = True
    for icyc in range(max_cycle):
        if fresh_start:
            xs = np.zeros((0, x0_size))
            ax = np.zeros((0, x0_size))
            row1 = 0
            xt = x0
            if x0sym is not None:
                xs_ir = []
                xt_ir = x0_ir

        if x0sym is None:
            xt = _symmetric_orth(xt)
        else:
            xt_orth = []
            xt_orth_ir = []
            for ir in set(xt_ir):
                idx = np.where(xt_ir == ir)[0]
                xt_sub = _symmetric_orth(xt[idx])
                xt_orth.append(xt_sub)
                xt_orth_ir.append([ir] * len(xt_sub))
            xt = np.vstack(xt_orth)
            xs_ir = np.hstack([xs_ir, *xt_orth_ir])
            xt_orth = xt_orth_ir = None

        axt = aop(xt)
        xs = np.vstack([xs, xt])
        ax = np.vstack([ax, axt])
        row0, row1 = row1, row1+len(xt)

        if heff is None:
            dtype = np.result_type(axt, xt)
            heff = np.empty((max_space*2,max_space*2), dtype=dtype)

        h11 = xs[:row0].conj().dot(axt.T)
        h21 = _conj_dot(xs[:row0], axt)
        heff[0:row0*2:2, row0*2+0:row1*2:2] = h11
        heff[1:row0*2:2, row0*2+0:row1*2:2] = h21
        heff[0:row0*2:2, row0*2+1:row1*2:2] = -h21.conj()
        heff[1:row0*2:2, row0*2+1:row1*2:2] = -h11.conj()

        h11 = xt.conj().dot(ax.T)
        h21 = _conj_dot(xt, ax)
        heff[row0*2+0:row1*2:2, 0:row1*2:2] = h11
        heff[row0*2+1:row1*2:2, 0:row1*2:2] = h21
        heff[row0*2+0:row1*2:2, 1:row1*2:2] = -h21.conj()
        heff[row0*2+1:row1*2:2, 1:row1*2:2] = -h11.conj()

        if x0sym is None:
            w, v = scipy.linalg.eig(heff[:row1*2,:row1*2])
        else:
            # Diagonalize within eash symmetry sectors
            xs_ir2 = np.repeat(xs_ir, 2)
            w = np.empty(row1*2, dtype=np.complex128)
            v = np.zeros((row1*2, row1*2), dtype=np.complex128)
            v_ir = []
            i1 = 0
            for ir in set(xs_ir):
                idx = np.where(xs_ir2 == ir)[0]
                i0, i1 = i1, i1 + idx.size
                w_sub, v_sub = scipy.linalg.eig(heff[idx[:,None],idx])
                w[i0:i1] = w_sub
                v[idx,i0:i1] = v_sub
                v_ir.append([ir] * idx.size)
            v_ir = np.hstack(v_ir)

        w, v, idx = pick(w, v, nroots, locals())
        if x0sym is not None:
            v_ir = v_ir[idx]
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        w, e, elast = w[:space_inc], w[:nroots], e
        v, vlast = v[:,:space_inc], v[:,:nroots]
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv, vlast, v[:,:nroots], log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x0 = _gen_x0(v, xs)
        ax0 = xt = _gen_ax0(v, ax)
        xt -= w[:,None] * x0
        ax0 = None
        if x0sym is not None:
            xt_ir = x0_ir = v_ir[:space_inc]

        dx_norm = np.linalg.norm(xt, axis=1)
        max_dx_norm = max(dx_norm[:nroots])
        conv = dx_norm[:nroots] < tol_residual
        for k, ek in enumerate(e[:nroots]):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, len(xs), max_dx_norm, e, de[ide])
            break

        # remove subspace linear dependency
        for k, xk in enumerate(xt):
            if dx_norm[k] > tol_residual:
                xt[k] = precond(xk, e[0])

        xt -= xs.conj().dot(xt.T).T.dot(xs)
        c = _conj_dot(xs, xt)
        xt[:,:half_size] -= c.T.dot(xs[:,half_size:].conj())
        xt[:,half_size:] -= c.T.dot(xs[:,:half_size].conj())
        xt_norm = np.linalg.norm(xt, axis=1)

        remaining = []
        for k, xk in enumerate(xt):
            if (dx_norm[k] > tol_residual and
                xt_norm[k] > tol_residual and xt_norm[k]**2 > lindep):
                xt[k] /= xt_norm[k]
                remaining.append(k)
        if len(remaining) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {dx_norm}')
            break

        xt = xt[remaining]
        if x0sym is not None:
            xt_ir = xt_ir[remaining]
        norm_min = xt_norm[remaining].min()
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, len(xs), max_dx_norm, e, de[ide], norm_min)

        fresh_start = len(xs)+space_inc > max_space

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return conv[:nroots], e[:nroots], x0[:nroots]

def _gen_x0(v, xs):
    out = _outprod_to_subspace(v[::2], xs)
    out_conj = _outprod_to_subspace(v[1::2].conj(), xs)
    n = out.shape[1] // 2
    # v[1::2] * xs.conj() = (v[1::2].conj() * xs).conj()
    out[:,:n] += out_conj[:,n:].conj()
    out[:,n:] += out_conj[:,:n].conj()
    return out

def _gen_ax0(v, xs):
    out = _outprod_to_subspace(v[::2], xs)
    out_conj = _outprod_to_subspace(v[1::2].conj(), xs)
    n = out.shape[1] // 2
    out[:,:n] -= out_conj[:,n:].conj()
    out[:,n:] -= out_conj[:,:n].conj()
    return out

def _conj_dot(xi, xj):
    '''Dot product between the conjugated basis of xi and xj.
    The conjugated basis of xi is np.hstack([xi[half:], xi[:half]]).conj()
    '''
    n = xi.shape[-1] // 2
    return xi[:,n:].dot(xj[:,:n].T) + xi[:,:n].dot(xj[:,n:].T)

def _qr(xs, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nv = 0
    idx = []
    for i, xi in enumerate(xs):
        for j in range(nv):
            prod = xs[j].conj().dot(xi)
            xi -= xs[j] * prod
        norm = np.linalg.norm(xi)
        if norm**2 > lindep:
            xs[nv] = xi/norm
            nv += 1
            idx.append(i)
    return xs[:nv], idx

def _symmetric_orth(xt, lindep=1e-6):
    xt = np.asarray(xt)
    n, m = xt.shape
    if n == 0:
        raise RuntimeError('Linear dependency in trial bases')
    m = m // 2
    # The conjugated basis np.hstack([xt[:,m:], xt[:,:m]]).conj()
    s11 = xt.conj().dot(xt.T)
    s21 = _conj_dot(xt, xt)
    s = np.block([[s11, s21.conj().T],
                  [s21, s11.conj()  ]])
    e, c = scipy.linalg.eigh(s)
    if e[0] < lindep:
        if n == 1:
            return xt
        return _symmetric_orth(xt[:-1], lindep)

    c_orth = (c * e**-.5).dot(c[:n].conj().T)
    x_orth = c_orth[:n].T.dot(xt)
    # Contribution from the conjugated basis
    x_orth[:,:m] += c_orth[n:].T.dot(xt[:,m:].conj())
    x_orth[:,m:] += c_orth[n:].T.dot(xt[:,:m].conj())
    return x_orth
