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
from pyscf.lib.linalg_helper import (
    FOLLOW_STATE, _sort_elast, _outprod_to_subspace, _normalize_xt_, _qr)

def lr_eig(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
           lindep=1e-12, nroots=1, pick=None, verbose=logger.WARN,
           follow_state=FOLLOW_STATE, tol_residual=None, metric=None):
    '''
    Solver for linear response eigenvalues
    [ A    B] [X] = w [X]
    [-B* -A*] [Y]     [Y]

    subject to normalization X^2 - Y^2 = 1

    Reference:
      Olsen, Jensen, and Jorgenson, J Comput Phys, 74, 265,
      DOI: 10.1016/0021-9991(88)90081-2
    '''

    assert metric is None
    assert callable(pick)
    assert callable(precond)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        tol_residual = tol**.5
    log.debug1('tol %g  tol_residual %g', tol, tol_residual)

    if callable(x0):
        x0 = x0()
    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = x0[None,:]
    x0 = np.asarray(x0)

    max_space = max_space + (nroots-1) * 6
    dtype = None
    heff = None
    seff = None
    e = None
    v = None
    conv = np.zeros(nroots, dtype=bool)
    emin = None
    level_shift = 0

    dot = np.dot
    half_size = x0[0].size // 2
    fresh_start = True

    for icyc in range(max_cycle):
        if fresh_start:
            xs = []
            ax = []
            row1 = 0
            xt = x0
        xt = _qr(xt, np.dot, lindep)[0]
        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        row0, row1 = row1, row1+len(xt)
        space = row1

        if heff is None:
            dtype = np.result_type(*axt, *xt)
            heff = np.empty((max_space*2,max_space*2), dtype=dtype)
            seff = np.empty((max_space*2,max_space*2), dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        for i in range(row1):
            xi1, xi2 = xs[i][:half_size], xs[i][half_size:]
            ui1, ui2 = ax[i][:half_size], ax[i][half_size:]
            for jp, j in enumerate(range(row0, row1)):
                xj1, xj2 = xt[jp][:half_size], xt[jp][half_size:]
                uj1, uj2 = axt[jp][:half_size], axt[jp][half_size:]
                s11 = dot(xi1.conj(), xj1) - dot(xi2.conj(), xj2)
                s21 = dot(xi2, xj1) - dot(xi1, xj2)
                seff[i*2  ,j*2  ] = s11
                seff[i*2+1,j*2  ] = s21
                seff[i*2  ,j*2+1] = -s21.conj()
                seff[i*2+1,j*2+1] = -s11.conj()

                h11 = dot(xi1.conj(), uj1) - dot(xi2.conj(), uj2)
                h21 = dot(xi2, uj1) - dot(xi1, uj2)
                heff[i*2  ,j*2  ] = h11
                heff[i*2+1,j*2  ] = h21
                heff[i*2  ,j*2+1] = h21.conj()
                heff[i*2+1,j*2+1] = h11.conj()

                if i < row0:
                    s11 = dot(xj1.conj(), xi1) - dot(xj2.conj(), xi2)
                    s21 = dot(xj2, xi1) - dot(xj1, xi2)
                    seff[j*2  ,i*2  ] = s11
                    seff[j*2+1,i*2  ] = s21
                    seff[j*2  ,i*2+1] = -s21.conj()
                    seff[j*2+1,i*2+1] = -s11.conj()

                    h11 = dot(xj1.conj(), ui1) - dot(xj2.conj(), ui2)
                    h21 = dot(xj2, ui1) - dot(xj1, ui2)
                    heff[j*2  ,i*2  ] = h11
                    heff[j*2+1,i*2  ] = h21
                    heff[j*2  ,i*2+1] = h21.conj()
                    heff[j*2+1,i*2+1] = h11.conj()

        xt = axt = None
        w, v = scipy.linalg.eig(heff[:row1*2,:row1*2], seff[:row1*2,:row1*2])
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
            if not conv[k]:
                xt[k] = ax0[k] - ek * x0[k]
                dx_norm[k] = np.linalg.norm(xt[k])
                conv[k] = abs(de[k]) < tol and dx_norm[k] < tol_residual
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
                xk = precond(xt[k], e[0]-level_shift, x0[k])
                for xi in xs:
                    xk -= xi * dot(xi.conj(), xk)
                norm = np.linalg.norm(xk)
                if norm**2 > lindep:
                    norm_min = min(norm_min, norm)
                    xk /= norm
                    if norm < tol_residual:
                        # To reduce errors in basis orthogonalization
                        for xi in xs:
                            xk -= xi * dot(xi.conj(), xk)
                    norm = np.linalg.norm(xk)
                    xk /= norm
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
            conv = conv & (dx_norm < tol_residual)
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
