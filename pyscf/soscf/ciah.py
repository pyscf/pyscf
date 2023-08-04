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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__


def expmat(a):
    return scipy.linalg.expm(a)

class CIAHOptimizer(lib.StreamObject):

    conv_tol_grad = getattr(__config__, 'soscf_ciah_CIAHOptimizer_conv_tol_grad', 1e-4)
    max_stepsize = getattr(__config__, 'soscf_ciah_CIAHOptimizer_max_stepsize', .05)
    max_iters = getattr(__config__, 'soscf_ciah_CIAHOptimizer_max_iters', 10)
    kf_interval = getattr(__config__, 'soscf_ciah_CIAHOptimizer_kf_interval', 5)
    kf_trust_region = getattr(__config__, 'soscf_ciah_CIAHOptimizer_kf_trust_region', 5)
    ah_start_tol = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_start_tol', 5.)
    ah_start_cycle = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_start_cycle', 1)
    ah_level_shift = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_level_shift', 0)
    ah_conv_tol = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_conv_tol', 1e-12)
    ah_lindep = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_lindep', 1e-14)
    ah_max_cycle = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_max_cycle', 30)
    ah_trust_region = getattr(__config__, 'soscf_ciah_CIAHOptimizer_ah_trust_region', 3.)

    def __init__(self):
        self._keys = set(('conv_tol_grad', 'max_stepsize', 'max_iters',
                          'kf_interval', 'kf_trust_region', 'ah_start_tol',
                          'ah_start_cycle', 'ah_level_shift', 'ah_conv_tol',
                          'ah_lindep', 'ah_max_cycle', 'ah_trust_region'))

    def gen_g_hop(self, u):
        pass

    def pack_uniq_var(self, mat):
        nmo = mat.shape[0]
        idx = numpy.tril_indices(nmo, -1)
        return mat[idx]

    def unpack_uniq_var(self, v):
        nmo = int(numpy.sqrt(v.size*2)) + 1
        idx = numpy.tril_indices(nmo, -1)
        mat = numpy.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.conj().T

    def extract_rotation(self, dr, u0=1):
        dr = self.unpack_uniq_var(dr)
        return numpy.dot(u0, expmat(dr))

    def get_grad(self, u):
        pass

    def cost_function(self, u):
        pass


def rotate_orb_cc(iah, u0, conv_tol_grad=None, verbose=logger.NOTE):
    t2m = (logger.process_clock(), logger.perf_counter())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if conv_tol_grad is None:
        conv_tol_grad = iah.conv_tol_grad

    g_orb, h_op, h_diag = iah.gen_g_hop(u0)
    g_kf = g_orb
    norm_gkf = norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|= %4.3g (keyframe)', norm_gorb)
    t3m = log.timer('gen h_op', *t2m)

    if h_diag is None:
        def precond(x, e):
            return x
    else:
        def precond(x, e):
            hdiagd = h_diag-(e-iah.ah_level_shift)
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            x = x/hdiagd
            return x

    def scale_down_step(dxi, hdxi, norm_gorb):
        dxmax = abs(dxi).max()
        if dxmax > iah.max_stepsize:
            scale = iah.max_stepsize / dxmax
            log.debug1('Scale rotation by %g', scale)
            dxi *= scale
            hdxi *= scale
        return dxi, hdxi

    class Statistic:
        def __init__(self):
            self.imic = 0
            self.tot_hop = 0
            self.tot_kf = 0

    kf_trust_region = iah.kf_trust_region
    g_op = lambda: g_orb
    x0_guess = g_orb
    while True:
        stat = Statistic()
        dr = 0
        ikf = 0
        ukf = 1

        for ah_conv, ihop, w, dxi, hdxi, residual, seig \
                in davidson_cc(h_op, g_op, precond, x0_guess,
                               tol=iah.ah_conv_tol, max_cycle=iah.ah_max_cycle,
                               lindep=iah.ah_lindep, verbose=log):
            stat.tot_hop = ihop
            norm_residual = numpy.linalg.norm(residual)
            if (ah_conv or ihop == iah.ah_max_cycle or # make sure to use the last step
                ((norm_residual < iah.ah_start_tol) and (ihop >= iah.ah_start_cycle)) or
                (seig < iah.ah_lindep)):
                stat.imic += 1
                dxmax = abs(dxi).max()
                dxi, hdxi = scale_down_step(dxi, hdxi, norm_gorb)

                dr = dr + dxi
                g_orb = g_orb + hdxi
                norm_dr = numpy.linalg.norm(dr)
                norm_gorb = numpy.linalg.norm(g_orb)
                log.debug('    imic %d(%d)  |g|= %4.3g  |dxi|= %4.3g  '
                          'max(|x|)= %4.3g  |dr|= %4.3g  eig= %4.3g  seig= %4.3g',
                          stat.imic, ihop, norm_gorb, numpy.linalg.norm(dxi),
                          dxmax, norm_dr, w, seig)

                max_cycle = max(iah.max_iters,
                                iah.max_iters-int(numpy.log(norm_gkf+1e-9)*2))
                log.debug1('Set max_cycle %d', max_cycle)
                ikf += 1
                if stat.imic > 3 and norm_gorb > norm_gkf*iah.ah_trust_region:
                    g_orb = g_orb - hdxi
                    dr -= dxi
                    norm_gorb = numpy.linalg.norm(g_orb)
                    log.debug('|g| >> keyframe, Restore previouse step')
                    break

                elif (stat.imic >= max_cycle or norm_gorb < conv_tol_grad*.2):
                    break

                elif (ikf > 2 and # avoid frequent keyframe
                      (ikf >= max(iah.kf_interval, iah.kf_interval-numpy.log(norm_dr+1e-9)) or
                       # Insert keyframe if the keyframe and the esitimated g_orb are too different
                       norm_gorb < norm_gkf/kf_trust_region)):
                    ikf = 0
                    ukf = iah.extract_rotation(dr, ukf)
                    dr[:] = 0
                    g_kf1 = iah.get_grad(u0.dot(ukf))
                    stat.tot_kf += 1
                    norm_gkf1 = numpy.linalg.norm(g_kf1)
                    norm_dg = numpy.linalg.norm(g_kf1-g_orb)
                    log.debug('Adjust keyframe g_orb to |g|= %4.3g  '
                              '|g-correction|= %4.3g', norm_gkf1, norm_dg)

                    if (norm_dg < norm_gorb*iah.ah_trust_region  # kf not too diff
                        #or norm_gkf1 < norm_gkf  # grad is decaying
                        # close to solution
                        or norm_gkf1 < conv_tol_grad*iah.ah_trust_region):
                        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), iah.kf_trust_region), 10)
                        log.debug1('Set kf_trust_region = %g', kf_trust_region)
                        g_orb = g_kf = g_kf1
                        norm_gorb = norm_gkf = norm_gkf1
                    else:
                        g_orb = g_orb - hdxi
                        dr -= dxi
                        norm_gorb = numpy.linalg.norm(g_orb)
                        log.debug('Out of trust region. Restore previouse step')
                        break

        u = iah.extract_rotation(dr, ukf)
        log.debug('    tot inner=%d  |g|= %4.3g  |u-1|= %4.3g',
                  stat.imic, norm_gorb, numpy.linalg.norm(numpy.tril(u,-1)))
        h_op = h_diag = None
        t3m = log.timer('aug_hess in %d inner iters' % stat.imic, *t3m)
        u0 = (yield u, g_kf, stat)

        g_kf, h_op, h_diag = iah.gen_g_hop(u0)
        norm_gkf = numpy.linalg.norm(g_kf)
        norm_dg = numpy.linalg.norm(g_kf-g_orb)
        log.debug('    |g|= %4.3g (keyframe), |g-correction|= %4.3g',
                  norm_gkf, norm_dg)
        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), iah.kf_trust_region), 10)
        log.debug1('Set  kf_trust_region = %g', kf_trust_region)
        g_orb = g_kf
        norm_gorb = norm_gkf
        x0_guess = dxi

def davidson_cc(h_op, g_op, precond, x0, tol=1e-10, xs=[], ax=[],
                max_cycle=30, lindep=1e-14, dot=numpy.dot, verbose=logger.WARN):

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    toloose = numpy.sqrt(tol)
    # the first trial vector is (1,0,0,...), which is not included in xs
    xs = list(xs)
    ax = list(ax)
    nx = len(xs)

    problem_size = x0.size
    max_cycle = min(max_cycle, problem_size)
    heff = numpy.zeros((max_cycle+nx+1,max_cycle+nx+1), dtype=x0.dtype)
    ovlp = numpy.eye(max_cycle+nx+1, dtype=x0.dtype)
    if nx == 0:
        xs.append(x0)
        ax.append(h_op(x0))
    else:
        for i in range(1, nx+1):
            for j in range(1, i+1):
                heff[i,j] = dot(xs[i-1].conj(), ax[j-1])
                ovlp[i,j] = dot(xs[i-1].conj(), xs[j-1])
            heff[1:i,i] = heff[i,1:i].conj()
            ovlp[1:i,i] = ovlp[i,1:i].conj()

    w_t = 0
    for istep in range(max_cycle):
        g = g_op()
        nx = len(xs)
        for i in range(nx):
            heff[i+1,0] = dot(xs[i].conj(), g)
            heff[nx,i+1] = dot(xs[nx-1].conj(), ax[i])
            ovlp[nx,i+1] = dot(xs[nx-1].conj(), xs[i])
        heff[0,:nx+1] = heff[:nx+1,0].conj()
        heff[1:nx,nx] = heff[nx,1:nx].conj()
        ovlp[1:nx,nx] = ovlp[nx,1:nx].conj()
        nvec = nx + 1
        #s0 = scipy.linalg.eigh(ovlp[:nvec,:nvec])[0][0]
        #if s0 < lindep:
        #    yield True, istep, w_t, xtrial, hx, dx, s0
        #    break
        wlast = w_t
        xtrial, w_t, v_t, index, seig = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs,
                              lindep, log)
        s0 = seig[0]
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - w_t * v_t[0]*xtrial
        norm_dx = numpy.linalg.norm(dx)
        log.debug1('... AH step %d  index= %d  |dx|= %.5g  eig= %.5g  v[0]= %.5g  lindep= %.5g',
                   istep+1, index, norm_dx, w_t, v_t[0].real, s0)
        hx *= 1/v_t[0] # == h_op(xtrial)
        if ((abs(w_t-wlast) < tol and norm_dx < toloose) or
            s0 < lindep or
            istep+1 == problem_size):
            # Avoid adding more trial vectors if hessian converged
            yield True, istep+1, w_t, xtrial, hx, dx, s0
            if s0 < lindep or norm_dx < lindep:# or numpy.linalg.norm(xtrial) < lindep:
                # stop the iteration because eigenvectors would be barely updated
                break
        else:
            yield False, istep+1, w_t, xtrial, hx, dx, s0
            x0 = precond(dx, w_t)
            xs.append(x0)
            ax.append(h_op(x0))


def _regular_step(heff, ovlp, xs, lindep, log):
    w, v, seig = lib.safe_eigh(heff, ovlp, lindep)
    if log.verbose >= logger.DEBUG3:
        numpy.set_printoptions(3, linewidth=1000)
        log.debug3('v[0] %s', v[0])
        log.debug3('AH eigs %s', w)
        numpy.set_printoptions(8, linewidth=75)

    #if e[0] < -.1:
    #    sel = 0
    #else:
    # There exists systems that the first eigenvalue of AH is -inf.
    # Dynamically choosing the eigenvectors may be better.
    idx = numpy.where(abs(v[0]) > 0.1)[0]
    sel = idx[0]
    log.debug1('CIAH eigen-sel %s', sel)
    w_t = w[sel]

    if w_t < 1e-4:
        try:
            e, c = scipy.linalg.eigh(heff[1:,1:], ovlp[1:,1:])
        except scipy.linalg.LinAlgError:
            e, c = lib.safe_eigh(heff[1:,1:], ovlp[1:,1:], lindep)[:2]
        if numpy.any(e < -1e-5):
            log.debug('Negative hessians found %s', e[e<0])

    xtrial = _dgemv(v[1:,sel]/v[0,sel], xs)
    return xtrial, w_t, v[:,sel], sel, seig

def _dgemv(v, m):
    vm = v[0] * m[0]
    for i,vi in enumerate(v[1:]):
        vm += vi * m[i+1]
    return vm


