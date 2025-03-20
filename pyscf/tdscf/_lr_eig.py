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

import numpy as np
import scipy.linalg
from pyscf.lib.parameters import MAX_MEMORY
from pyscf.lib import logger
from pyscf.lib.exceptions import LinearDependencyError
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

    log = logger.new_logger(verbose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)

    x0_size = x0.shape[1]
    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        # Adding too many trial bases in each iteration may cause larger errors
        space_inc = max(nroots, min(MAX_SPACE_INC, x0_size//2))

    max_space = int(max_memory*1e6/8/x0_size / 2 - nroots - space_inc)
    if max_space < nroots * 4 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eigh')
    max_space = max(max_space, nroots * 4)
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
            else:
                log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
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
        log.debug1('Generate %d trial vectors. Drop %d vectors',
                   len(xt), dx_norm.size - len(xt))

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
        conv : list of booleans
            Whether each root is converged.
        e : list of floats
            Eigenvalues.
        c : list of 1D arrays
            Eigenvectors.
    '''

    assert callable(pick)
    assert callable(precond)

    log = logger.new_logger(verbose)

    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)
    x0_size = x0.shape[1]
    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        # Adding too many trial bases in each iteration may introduce more errors
        space_inc = max(nroots, min(MAX_SPACE_INC, x0_size//4))

    max_space = int(max_memory*1e6/8/(2*x0_size) / 2 - space_inc)
    if max_space < nroots * 4 < x0_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eig')
        max_space = space_inc * 2
    max_space = max(max_space, nroots * 4)
    max_space = min(max_space, x0_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    if x0sym is None:
        x0 = _symmetric_orth(x0)
    else:
        x0_ir = np.asarray(x0sym)
        x0_orth = []
        x0_orth_ir = []
        for ir in set(x0_ir):
            idx = np.where(x0_ir == ir)[0]
            xt_sub = _symmetric_orth(x0[idx])
            x0_orth.append(xt_sub)
            x0_orth_ir.append([ir] * len(xt_sub))
        if x0_orth:
            x0 = np.vstack(x0_orth)
            x0_ir = np.hstack(x0_orth_ir)
        else:
            x0 = []
        x0_orth = x0_orth_ir = xt_sub = None
    if len(x0) == 0:
        raise LinearDependencyError('Empty initial guess')

    heff = None
    e = None
    v = None
    vlast = None
    conv_last = conv = np.zeros(nroots, dtype=bool)

    half_size = x0[0].size // 2
    fresh_start = True
    for icyc in range(max_cycle):
        if fresh_start:
            vlast = None
            conv_last = conv = np.zeros(nroots, dtype=bool)
            xs = np.zeros((0, x0_size))
            ax = np.zeros((0, x0_size))
            row1 = 0
            xt = x0
            if x0sym is not None:
                xs_ir = x0_ir

        axt = aop(xt)
        xs = np.vstack([xs, xt])
        ax = np.vstack([ax, axt])
        row0, row1 = row1, row1+len(xt)

        if heff is None:
            dtype = np.result_type(axt, xt)
            heff = np.empty((max_space*2,max_space*2), dtype=dtype)

        h11 = xs[:row0].conj().dot(axt.T).astype(dtype)
        h21 = _conj_dot(xs[:row0], axt).astype(dtype)
        heff[0:row0*2:2, row0*2+0:row1*2:2] = h11
        heff[1:row0*2:2, row0*2+0:row1*2:2] = h21
        heff[0:row0*2:2, row0*2+1:row1*2:2] = -h21.conj()
        heff[1:row0*2:2, row0*2+1:row1*2:2] = -h11.conj()

        h11 = xt.conj().dot(ax.T).astype(dtype)
        h21 = _conj_dot(xt, ax).astype(dtype)
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
        v = v[:,:space_inc]
        if vlast is not None:
            elast, conv_last = _sort_elast(elast, conv, vlast, v[:,:nroots], log)
        vlast = v[:,:nroots]

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
            xt_ir = v_ir[:space_inc]
            x0_ir = v_ir[:nroots]

        dx_norm = np.linalg.norm(xt, axis=1)
        max_dx_norm = max(dx_norm[:nroots])
        conv = dx_norm[:nroots] < tol_residual
        for k, ek in enumerate(e[:nroots]):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
            else:
                log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
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

        # Remove quasi linearly dependent bases, as they cause more numerical
        # errors in _symmetric_orth
        xt_norm = np.linalg.norm(xt, axis=1)
        xt_to_keep = (dx_norm > tol_residual) & (xt_norm > max(lindep**.5, tol_residual))
        xt = xt[xt_to_keep]
        if len(xt) > 0:
            xt /= xt_norm[xt_to_keep, None]
            if x0sym is None:
                xt = _symmetric_orth(xt)
            else:
                xt_ir = xt_ir[xt_to_keep]
                xt_orth = []
                xt_orth_ir = []
                for ir in set(xt_ir):
                    idx = np.where(xt_ir == ir)[0]
                    xt_sub = _symmetric_orth(xt[idx])
                    xt_orth.append(xt_sub)
                    xt_orth_ir.append([ir] * len(xt_sub))
                if xt_orth:
                    xt = np.vstack(xt_orth)
                    xs_ir = np.hstack([xs_ir, *xt_orth_ir])
                else:
                    xt = []
                xt_orth = xt_orth_ir = xt_sub = None

        if len(xt) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {dx_norm}')
            break
        log.debug1('Generate %d trial vectors. Drop %d vectors',
                   len(xt), dx_norm.size - len(xt))

        xt_norm = np.linalg.norm(xt, axis=1)
        norm_min = xt_norm.min()
        log.debug('lr_eig %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, len(xs), max_dx_norm, e, de[ide], norm_min)

        fresh_start = len(xs)+space_inc > max_space

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return conv[:nroots], e[:nroots], x0[:nroots]

def real_eig(aop, x0, precond, tol_residual=1e-5, nroots=1, x0sym=None, pick=None,
             max_cycle=50, max_memory=MAX_MEMORY, lindep=1e-12, verbose=logger.WARN):
    '''
    Solve linear response eigenvalues for real A and B matrices
    [ A  B] [X] = w [X]
    [-B -A] [Y]     [Y]

    subject to normalization X^2 - Y^2 = 1 . This function is based on the
    algorithm implemented in https://github.com/John-zzh/improved-Davidson-Algorithm

    Args:
        aop : function(x) => array_like_x
            The matrix-vector product operator.
        x0 : 1D array or a list of 1D array
            Initial guess.
        precond : function(dx, e) => array_like_dx
            Preconditioner to generate new trial vector.

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
        conv : list of booleans
            Whether each root is converged.
        e : list of floats
            Eigenvalues.
        c : list of 1D arrays
            Eigenvectors.
    '''

    assert pick is None
    assert callable(precond)

    log = logger.new_logger(verbose)

    assert x0.ndim == 2
    A_size = x0.shape[1] // 2
    V = x0[:,:A_size]
    W = x0[:,A_size:]
    x0 = (V, W)
    if MAX_SPACE_INC is None:
        space_inc = nroots
    else:
        # Adding too many trial bases in each iteration may cause larger errors
        space_inc = max(nroots, min(MAX_SPACE_INC, A_size//2))

    max_space = int(max_memory*1e6/8/(4*A_size) / 2 - space_inc)
    if max_space < nroots * 4 < A_size:
        log.warn('Not enough memory to store trial space in _lr_eig.eig')
        max_space = space_inc * 2
    max_space = max(max_space, nroots * 4)
    max_space = min(max_space, A_size)
    log.debug(f'Set max_space {max_space}, space_inc {space_inc}')

    if x0sym is not None:
        x0_ir = np.asarray(x0sym)

    '''U1 = AV + BW
       U2 = AW + BV'''
    V_holder = np.empty((A_size, max_space), order='F')
    W_holder = np.empty_like(V_holder)
    U1_holder = np.empty_like(V_holder)
    U2_holder = np.empty_like(V_holder)

    a = np.empty((max_space*2,max_space*2))
    b = np.empty_like(a)
    sigma = np.empty_like(a)
    pi = np.empty_like(a)
    e = None
    v_sub = None
    vlast = None
    conv_last = conv = np.zeros(nroots, dtype=bool)

    fresh_start = True
    for icyc in range(max_cycle):
        if fresh_start:
            m0 = m1 = 0
            V, W = x0
            if x0sym is not None:
                xs_ir = xt_ir = x0_ir

        axt = aop(np.hstack([V, W]))
        U1 =  axt[:,:A_size]
        U2 = -axt[:,A_size:]
        m0, m1 = m1, m1+len(U1)
        V_holder [:,m0:m1] = V.T
        W_holder [:,m0:m1] = W.T
        U1_holder[:,m0:m1] = U1.T
        U2_holder[:,m0:m1] = U2.T

        '''
        a = np.dot(V.T, U1)
        a += np.dot(W.T, U2)
        b = np.dot(V.T, U2)
        b += np.dot(W.T, U1)
        sigma = np.dot(V.T, V)
        sigma -= np.dot(W.T, W)
        pi = np.dot(V.T, W)
        pi -= np.dot(W.T, V)
        a = (a + a.T) / 2
        b = (b + b.T) / 2
        sigma = (sigma + sigma.T) / 2
        pi = (pi - pi.T) / 2
        '''
        a_block  = _sym_dot(V_holder, U1_holder, m0, m1)
        a_block += _sym_dot(W_holder, U2_holder, m0, m1)
        b_block  = _sym_dot(V_holder, U2_holder, m0, m1)
        b_block += _sym_dot(W_holder, U1_holder, m0, m1)
        sigma_block  = _sym_dot(V_holder, V_holder, m0, m1)
        sigma_block -= _sym_dot(W_holder, W_holder, m0, m1)
        pi_block  = _asym_dot(V_holder, W_holder, m0, m1)
        pi_block -= _asym_dot(W_holder, V_holder, m0, m1)
        a[:m1,m0:m1] = a_block.T
        a[m0:m1,:m1] = a_block
        b[:m1,m0:m1] = b_block.T
        b[m0:m1,:m1] = b_block
        sigma[:m1,m0:m1] = sigma_block.T
        sigma[m0:m1,:m1] = sigma_block
        pi[:m1,m0:m1] = -pi_block.T
        pi[m0:m1,:m1] = pi_block

        if x0sym is None:
            omega, x, y = TDDFT_subspace_eigen_solver(
                a[:m1,:m1], b[:m1,:m1], sigma[:m1,:m1], pi[:m1,:m1], space_inc)
        else:
            # Diagonalize within eash symmetry sectors
            omega = np.empty(m1)
            x = np.zeros((m1, m1))
            y = np.zeros_like(x)
            v_ir = []
            i1 = 0
            for ir in set(xs_ir):
                idx = np.nonzero(xs_ir[:m1] == ir)[0]
                _w, _x, _y = TDDFT_subspace_eigen_solver(
                    a[idx[:,None],idx], b[idx[:,None],idx],
                    sigma[idx[:,None],idx], pi[idx[:,None],idx], idx.size)
                i0, i1 = i1, i1 + idx.size
                omega[i0:i1] = _w
                x[idx,i0:i1] = _x
                y[idx,i0:i1] = _y
                v_ir.append([ir] * _w.size)
            idx = np.argsort(omega)
            omega = omega[idx]
            v_ir = np.hstack(v_ir)[idx]
            x = x[:,idx]
            y = y[:,idx]

        w, e, elast = omega[:space_inc], omega[:nroots], e
        v_sub = x[:,:space_inc]
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv, vlast, v_sub[:,:nroots], log)
        vlast = v_sub[:,:nroots]

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x = x[:,:space_inc]
        y = y[:,:space_inc]
        X_full  = V_holder[:,:m1].dot(x)
        X_full += W_holder[:,:m1].dot(y)
        Y_full  = W_holder[:,:m1].dot(x)
        Y_full += V_holder[:,:m1].dot(y)
        x0 = (X_full[:,:nroots].T, Y_full[:,:nroots].T)
        # residuals
        R_x  = U1_holder[:,:m1].dot(x)
        R_x += U2_holder[:,:m1].dot(y)
        R_x -= X_full * w
        R_y  = U2_holder[:,:m1].dot(x)
        R_y += U1_holder[:,:m1].dot(y)
        R_y += Y_full * w

        r_norms  = np.linalg.norm(R_x, axis=0) ** 2
        r_norms += np.linalg.norm(R_y, axis=0) ** 2
        r_norms = r_norms ** .5

        if x0sym is not None:
            xt_ir = v_ir[:space_inc]
            x0_ir = v_ir[:nroots]

        max_r_norm = max(r_norms[:nroots])
        conv = r_norms[:nroots] <= tol_residual
        for k, ek in enumerate(e[:nroots]):
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, r_norms[k], ek, de[k])
            else:
                log.debug1('root %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, r_norms[k], ek, de[k])
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, len(conv), max_r_norm, e, de[ide])
            break

        r_index = r_norms > tol_residual
        XY_new = precond(np.vstack([R_x[:,r_index], R_y[:,r_index]]).T, w[r_index])
        X_new = XY_new[:,:A_size].T
        Y_new = XY_new[:,A_size:].T
        if x0sym is None:
            V, W = VW_Gram_Schmidt_fill_holder(
                V_holder[:,:m1], W_holder[:,:m1], X_new, Y_new, lindep)
        else:
            xt_ir = xt_ir[r_index]
            xt_orth_ir = []
            V = []
            W = []
            for ir in set(xt_ir):
                idx = np.nonzero(xt_ir == ir)[0]
                _V, _W = VW_Gram_Schmidt_fill_holder(
                    V_holder[:,:m1], W_holder[:,:m1], X_new[:,idx], Y_new[:,idx], lindep)
                V.append(_V)
                W.append(_W)
                xt_orth_ir.append([ir] * len(_V))
            if len(V) > 0:
                V = np.vstack(V)
                W = np.vstack(W)
                xt_ir = np.hstack(xt_orth_ir)
                xs_ir = np.hstack([xs_ir, xt_ir])

        if len(V) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {r_norms}')
            break

        log.debug1('Generate %d trial vectors. Drop %d vectors',
                   len(V), r_norms.size - len(V))
        X_new = Y_new = R_x = R_y = None

        xy_norms  = np.linalg.norm(V, axis=0) ** 2
        xy_norms += np.linalg.norm(W, axis=0) ** 2
        norm_min = (xy_norms ** .5).min()
        log.debug('real_lr_eig %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, m1, max_r_norm, e, de[ide], norm_min)

        fresh_start = m1 + len(V) > max_space

    # Check whether the solver finds enough eigenvectors.
    if len(x0[0]) < min(A_size, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0[0])}, nroots={nroots})')

    return conv[:nroots], e[:nroots], np.hstack(x0)

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
    if xt.dtype == np.float64:
        return _symmetric_orth_real(xt, lindep)
    else:
        return _symmetric_orth_cmplx(xt, lindep)

def _symmetric_orth_real(xt, lindep=1e-6):
    '''
    Symmetric orthogonalization for xt = {[X, Y]},
    and its dual basis vectors {[Y, X]}
    '''
    x0_size = xt.shape[1]
    s11 = xt.dot(xt.T)
    s21 = _conj_dot(xt, xt)
    # Symmetric orthogonalize s, where
    # s = [[s11, s21.conj().T],
    #      [s21, s11.conj()  ]]
    e, c = np.linalg.eigh(s11)
    mask = e > lindep**2
    e = e[mask]
    if e.size == 0:
        return np.zeros([0, x0_size], dtype=xt.dtype)
    c = c[:,mask] * e**-.5

    # c22 = c.conj()
    # s21 -> c22.conj().T.dot(s21).dot(c11)
    csc = c.T.dot(s21).dot(c)
    n = csc.shape[0]
    for i in range(n):
        _s21 = csc[i:,i:]
        # s21 is symmetric for real vectors
        w, u = np.linalg.eigh(_s21)
        mask = 1 - abs(w) > lindep
        if np.any(mask):
            c = c[:,i:]
            break
    else:
        return np.zeros([0, x0_size], dtype=xt.dtype)
    w = w[mask]
    u = u[:,mask]
    c_orth = c.dot(u)

    if e[0] < 1e-6 or 1-abs(w[0]) < 1e-3:
        # Rerun the orthogonalization to reduce numerical errors
        e, c = np.linalg.eigh(c_orth.T.dot(s11).dot(c_orth))
        c *= e**-.5
        c_orth = c_orth.dot(c)
        csc = c_orth.T.dot(s21).dot(c_orth)
        w, u = np.linalg.eigh(csc)
        c_orth = c_orth.dot(u)

    # Symmetric diagonalize
    # [1 w.conj()] => c = [a b]
    # [w 1       ]        [b a]
    # where
    # a = ((1+w)**-.5 + (1-w)**-.5)/2
    # b = (phase*(1+w)**-.5 - phase*(1-w)**-.5)/2
    a1 = (1 + w)**-.5
    a2 = (1 - w)**-.5
    a = (a1 + a2) / 2
    b = (a1 - a2) / 2

    m = xt.shape[1] // 2
    x_orth = (c_orth * a).T.dot(xt)
    # Contribution from the conjugated basis
    x_orth[:,:m] += (c_orth * b).T.dot(xt[:,m:])
    x_orth[:,m:] += (c_orth * b).T.dot(xt[:,:m])
    return x_orth

def _symmetric_orth_cmplx(xt, lindep=1e-6):
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
        return _symmetric_orth_cmplx(xt[:-1], lindep)

    c_orth = (c * e**-.5).dot(c[:n].conj().T)
    x_orth = c_orth[:n].T.dot(xt)
    # Contribution from the conjugated basis
    x_orth[:,:m] += c_orth[n:].T.dot(xt[:,m:].conj())
    x_orth[:,m:] += c_orth[n:].T.dot(xt[:,:m].conj())
    return x_orth

def _sym_dot(V, U1, m0, m1):
    '''(V*U1 + U1.T*V.T)[m0:m1,:m1]'''
    a  = V [:,m0:m1].T.dot(U1[:,:m1])
    a += U1[:,m0:m1].T.dot(V [:,:m1])
    a *= .5
    return a

def _asym_dot(V, U1, m0, m1):
    '''(V*U1 - U1.T*V.T)[m0:m1,:m1]'''
    a  = V [:,m0:m1].T.dot(U1[:,:m1])
    a -= U1[:,m0:m1].T.dot(V [:,:m1])
    a *= .5
    return a

def TDDFT_subspace_eigen_solver(a, b, sigma, pi, nroots):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    s_m_p = np.einsum('i,ij,j->ij', d_mh, sigma - pi, d_mh)

    '''LU = d^−1/2 (σ − π) d^−1/2'''
    ''' A = LU '''
    L, U = scipy.linalg.lu(s_m_p, permute_l=True)
    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    '''U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    d_amb_d = np.einsum('i,ij,j->ij', d_mh, a-b, d_mh)
    GGT = np.linalg.multi_dot([U_inv.T, d_amb_d, U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    d_apb_d = np.einsum('i,ij,j->ij', d_mh, a+b, d_mh)
    M = np.linalg.multi_dot([G.T, L_inv, d_apb_d, L_inv.T, G])

    omega2, Z = np.linalg.eigh(M)
    if np.any(omega2 <= 0):
        idx = np.nonzero(omega2 > 0)[0]
        omega2 = omega2[idx[:nroots]]
        Z = Z[:,idx[:nroots]]
    else:
        omega2 = omega2[:nroots]
        Z = Z[:,:nroots]
    omega = omega2**0.5

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''
    x_p_y = np.einsum('i,ik,k->ik', d_mh, L_inv.T.dot(G.dot(Z)), omega**-0.5)
    x_m_y = np.einsum('i,ik,k->ik', d_mh, U_inv.dot(G_inv.T.dot(Z)), omega**0.5)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x
    return omega, x, y

def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, X_new, Y_new, lindep=1e-12):
    '''
    QR orthogonalization for (X_new, Y_new) basis vectors, then apply symmetric
    orthogonalization for {[X, Y]}, and its dual basis vectors {[Y, X]}
    '''
    _x  = V_holder.T.dot(X_new)
    _x += W_holder.T.dot(Y_new)
    _y  = V_holder.T.dot(Y_new)
    _y += W_holder.T.dot(X_new)
    X_new -= V_holder.dot(_x)
    X_new -= W_holder.dot(_y)
    Y_new -= W_holder.dot(_x)
    Y_new -= V_holder.dot(_y)
    x0_size = X_new.shape[0]

    s11  = X_new.T.dot(X_new)
    s11 += Y_new.T.dot(Y_new)
    # s21 is symmetric
    s21  = X_new.T.dot(Y_new)
    s21 += Y_new.T.dot(X_new)
    e, c = np.linalg.eigh(s11)
    mask = e > lindep
    e = e[mask]
    if e.size == 0:
        return (np.zeros([0, x0_size], dtype=X_new.dtype),
                np.zeros([0, x0_size], dtype=Y_new.dtype))
    c = c[:,mask] * e**-.5

    csc = c.T.dot(s21).dot(c)
    n = csc.shape[0]
    lindep_sqrt = lindep**.5
    for i in range(n):
        w, u = np.linalg.eigh(csc[i:,i:])
        mask = 1 - abs(w) > lindep_sqrt
        if np.any(mask):
            c = c[:,i:]
            break
    else:
        return (np.zeros([0, x0_size], dtype=X_new.dtype),
                np.zeros([0, x0_size], dtype=Y_new.dtype))
    w = w[mask]
    u = u[:,mask]
    c_orth = c.dot(u)

    if e[0] < lindep_sqrt or any(abs(w)> 1-1e-3):
        # Rerun the orthogonalization to reduce numerical errors.
        # When w~=1-1e-3, errors in the orthogonalization (off-diagonal terms)
        # is near 1e-6.
        e, c = np.linalg.eigh(c_orth.T.dot(s11).dot(c_orth))
        c *= e**-.5
        c_orth = c_orth.dot(c)
        csc = c_orth.T.dot(s21).dot(c_orth)
        w, u = np.linalg.eigh(csc)

        mask = 1 - abs(w) > lindep_sqrt
        w = w[mask]
        u = u[:,mask]
        c_orth = c_orth.dot(u)

    # Symmetric diagonalize
    # [1 w] => c = [a b]
    # [w 1]        [b a]
    # where
    # a = ((1+w)**-.5 + (1-w)**-.5)/2
    # b = ((1+w)**-.5 - (1-w)**-.5)/2
    a1 = (1 + w)**-.5
    a2 = (1 - w)**-.5
    a = (a1 + a2) / 2
    b = (a1 - a2) / 2

    x_orth  = X_new.dot(c_orth * a)
    x_orth += Y_new.dot(c_orth * b)
    y_orth  = Y_new.dot(c_orth * a)
    y_orth += X_new.dot(c_orth * b)
    return x_orth.T, y_orth.T
