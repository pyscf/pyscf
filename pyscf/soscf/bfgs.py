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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#


import numpy as np
from pyscf.lib import logger

def _vdot_real(a, b):
    return np.vdot(a.conj().ravel(), b.ravel()).real

def _lbfgs_two_loop(g, s_list, y_list, rho_list, h0_scale=1.0):
    q = g.copy()
    alpha = []
    for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
        a = rho * _vdot_real(s, q)
        alpha.append(a)
        q -= a * y
    r = h0_scale * q
    for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alpha)):
        b = rho * _vdot_real(y, r)
        r += s * (a - b)
    return r

def rotate_orb_cc(localizer, u0, conv_tol_grad=None, verbose=None,
                  maximize=False, m=10, c1=1e-4, max_ls=15, ls_shrink=0.5,
                  max_stepsize=None):
    r'''Optimize orbital rotations with a quasi-Newton scheme (L-BFGS) on a
    unitary manifold.

    This driver updates a unitary rotation matrix ``u`` (typically acting on
    MOs/LMOs) to minimize (or maximize) a localization objective provided by
    ``localizer``. The unitary is parameterized as ``u = exp(K) @ u0`` where
    ``K`` is an anti-Hermitian generator built from the (trivialized) gradient.
    A backtracking line search is used to ensure sufficient decrease/increase.

    Args:
    ----------
    localizer : object
        Localization / objective wrapper. It must provide at least
        1) ``localizer.cost_function(u)``
        2) ``localizer.get_grad(u)``
        3) ``extract_rotation(dr)``
        4) ``update_rotation(u, u_step)``
    u0 : ndarray
        Initial unitary rotation matrix. Shape ``(norb, norb)`` (or compatible
        block shape used by the localizer).

    Kwargs:
    ----------
    conv_tol_grad : float
        Convergence threshold for the gradient norm. Default is
        `localizer.conv_tol_grad``. If the latter is None, 1e-4 is used.
    verbose : int
        Verbosity level. Defaul is ``localizer.verbose``.
    maximize : bool
        If True, maximize the objective instead of minimizing it. Default is
        False (minimize).
    m : int
        L-BFGS history size (number of stored ``(s, y)`` pairs). Default is 10.
    c1 : float
        Armijo parameter for sufficient decrease (or increase if ``maximize`` is
        True) in the backtracking line search. Default is 1e-4.
    max_ls : int
        Maximum number of backtracking line-search steps per iteration.
        Default is 15.
    ls_shrink : float
        Step-size reduction factor in backtracking line search (0 < ls_shrink < 1).
        Default is 0.5.
    max_stepsize : float
        Optional cap on the step length (in the generator norm) to prevent
        overly large rotations. If provided, the trial step is scaled to satisfy
        the cap. Default is ``localizer.max_stepsize``. If the latter is None,
        0.05 is used.
    '''
    log = logger.new_logger(localizer, verbose=verbose)
    if conv_tol_grad is None:
        conv_tol_grad = getattr(localizer, "conv_tol_grad", 1e-4)
    if max_stepsize is None:
        max_stepsize = float(getattr(localizer, "max_stepsize", 0.05))

    class Statistic:
        def __init__(self):
            self.tot_kf = 0     # gradient builds
            # self.tot_e = 0    # objective evals
            # self.tot_ls = 0   # line-search tries
            self.tot_hop = 0

    if maximize:
        def f(u):
            # stat.tot_e += 1
            return -localizer.cost_function(u)  # minimize
    else:
        def f(u):
            # stat.tot_e += 1
            return localizer.cost_function(u)  # minimize

    # L-BFGS memory in packed coordinate space
    s_list, y_list, rho_list = [], [], []
    g_prev = None
    dr_prev = None

    while True:
        stat = Statistic()

        g = localizer.get_grad(u0)
        stat.tot_kf += 1
        gnorm = float(np.linalg.norm(g))

        if gnorm < conv_tol_grad:
            u_step = localizer.identity_rotation()
            u0 = (yield u_step, g, stat)
            g_prev, dr_prev = g, None
            continue

        # update memory from last accepted step
        if g_prev is not None and dr_prev is not None:
            s = dr_prev
            y = g - g_prev
            sty = _vdot_real(s, y)
            if sty > 1e-12:
                rho = 1.0 / sty
                s_list.append(s); y_list.append(y); rho_list.append(rho)
                if len(s_list) > m:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            else:
                s_list.clear(); y_list.clear(); rho_list.clear()

        # initial scaling
        h0_scale = 1.0
        if s_list:
            y_last = y_list[-1]
            s_last = s_list[-1]
            yy = _vdot_real(y_last, y_last)
            if yy > 1e-18:
                h0_scale = _vdot_real(s_last, y_last) / yy

        # direction
        if not s_list:
            p = -g
        else:
            p = -_lbfgs_two_loop(g, s_list, y_list, rho_list, h0_scale=h0_scale)

        # step cap (in packed coords)
        pmax = float(np.max(np.abs(p))) if p.size else 0.0
        if pmax > max_stepsize:
            p *= (max_stepsize / pmax)

        # Armijo backtracking
        f0 = f(u0)
        gtp = _vdot_real(g, p)
        if gtp >= 0:
            p = -g
            pmax = float(np.max(np.abs(p))) if p.size else 0.0
            if pmax > max_stepsize:
                p *= (max_stepsize / pmax)
            gtp = _vdot_real(g, p)

        alpha = 1.0
        u_step = None
        for _ in range(max_ls):
            # stat.tot_ls += 1
            dr = alpha * p
            step = localizer.extract_rotation(dr)
            u_try = localizer.update_rotation(u0, step)
            f1 = f(u_try)
            if f1 <= f0 + c1 * alpha * gtp:
                u_step = step
                break
            alpha *= ls_shrink

        if u_step is None:
            dr = alpha * p
            u_step = localizer.extract_rotation(dr)

        g_prev = g
        dr_prev = dr

        u0 = (yield u_step, g, stat)
