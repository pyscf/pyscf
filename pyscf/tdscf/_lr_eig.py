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
from pyscf.lib import logger
from pyscf.lib.linalg_helper import _sort_elast, _outprod_to_subspace

def eig(aop, x0, precond, tol_residual=1e-5, max_cycle=50, max_space=12,
        lindep=1e-12, nroots=1, pick=None, verbose=logger.WARN):
    '''
    Solver for linear response eigenvalues
    [ A    B] [X] = w [X]
    [-B* -A*] [Y]     [Y]

    subject to normalization X^2 - Y^2 = 1

    Reference:
      Olsen, Jensen, and Jorgenson, J Comput Phys, 74, 265,
      DOI: 10.1016/0021-9991(88)90081-2
    '''

    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if callable(x0):
        x0 = x0()
    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)

    max_space = max_space + (nroots-1) * 6
    heff = None
    e = None
    v = None
    conv = np.zeros(nroots, dtype=bool)

    dot = np.dot
    half_size = x0[0].size // 2
    fresh_start = True

    for icyc in range(max_cycle):
        if fresh_start:
            xs = []
            ax = []
            row1 = 0
            xt = x0
        xt = _symmetric_orth(xt)
        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        row0, row1 = row1, row1+len(xt)
        space = row1

        if heff is None:
            dtype = np.result_type(*axt, *xt)
            heff = np.empty((max_space*2,max_space*2), dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        for i in range(row1):
            for jp, j in enumerate(range(row0, row1)):
                h11 = xs[i].conj().dot(axt[jp])
                h21 = _conj_dot(xs[i], axt[jp])
                heff[i*2  ,j*2  ] = h11
                heff[i*2+1,j*2  ] = h21
                heff[i*2  ,j*2+1] = -h21.conj()
                heff[i*2+1,j*2+1] = -h11.conj()

                if i < row0:
                    h11 = xt[jp].conj().dot(ax[i])
                    h21 = _conj_dot(xt[jp], ax[i])
                    heff[j*2  ,i*2  ] = h11
                    heff[j*2+1,i*2  ] = h21
                    heff[j*2  ,i*2+1] = -h21.conj()
                    heff[j*2+1,i*2+1] = -h11.conj()

        w, v = scipy.linalg.eig(heff[:row1*2,:row1*2])
        w, v, idx = pick(w, v, nroots, locals())
        if len(w) == 0:
            raise RuntimeError('Not enough eigenvalues')

        e = w[:nroots]
        v = v[:,:nroots]
        conv = np.zeros(nroots, dtype=bool)
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x0 = _gen_x0(v, xs)
        ax0 = _gen_ax0(v, ax)

        dx_norm = np.zeros(nroots)
        xt = [None] * nroots
        for k, ek in enumerate(e):
            xt[k] = ax0[k] - ek * x0[k]
            dx_norm[k] = np.linalg.norm(xt[k])
            conv[k] = dx_norm[k] < tol_residual
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        max_dx_norm = max(dx_norm)
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break

        # remove subspace linear dependency
        norm_min = 1
        for k, ek in enumerate(e):
            if not conv[k]:
                xk = precond(xt[k], e[0], x0[k])
                norm_xk = np.linalg.norm(xk)
                for xi in xs:
                    xk -= xi * dot(xi.conj(), xk)
                    c = _conj_dot(xi, xk)
                    xk[:half_size] -= xi[half_size:].conj() * c
                    xk[half_size:] -= xi[:half_size].conj() * c
                norm = np.linalg.norm(xk)
                if (norm/norm_xk)**2 > lindep and norm/norm_xk > tol_residual:
                    norm_min = min(norm_min, norm)
                    xk /= norm
                    if norm < tol_residual:
                        # To reduce numerical errors in basis orthogonalization
                        for xi in xs:
                            xk -= xi * dot(xi.conj(), xk)
                            c = _conj_dot(xi, xk)
                            xk[:half_size] -= xi[half_size:].conj() * c
                            xk[half_size:] -= xi[:half_size].conj() * c
                        xk /= np.linalg.norm(xk)
                    xt[k] = xk
                else:
                    xt[k] = None
                    log.debug1('Drop eigenvector %d, norm=%4.3g', k, dx_norm[k])
            else:
                xt[k] = None

        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        xt = [x for x in xt if x is not None]
        if len(xt) == 0:
            log.debug(f'Linear dependency in trial subspace. |r| for each state {dx_norm}')
            conv = dx_norm < tol_residual
            break

        fresh_start = space+nroots > max_space

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    return np.asarray(conv), e, x0

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
    assert xi.ndim == 1
    n = xi.size // 2
    return xi[n:].dot(xj[:n]) + xi[:n].dot(xj[n:])

def _symmetric_orth(xt, lindep=1e-6):
    xt = np.asarray(xt)
    n, m = xt.shape
    if n == 0:
        raise RuntimeError('Linear dependency in trial bases')
    m = m // 2
    # The conjugated basis np.hstack([b2, b1]).conj()
    b1 = xt[:,:m]
    b2 = xt[:,m:]
    s11 = xt.conj().dot(xt.T)
    s21 = b2.dot(b1.T)
    s21 += b1.dot(b2.T)
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
    x_orth[:,:m] += c_orth[n:].T.dot(b2.conj())
    x_orth[:,m:] += c_orth[n:].T.dot(b1.conj())
    return x_orth
