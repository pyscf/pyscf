#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import tempfile
from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger

raise RuntimeError('TODO: test davidson')

def davidson(a, x0, precond=None, tol=1e-14, max_cycle=50, maxspace=12,
             nroots=1, hermi=1, lindep=1e-16, max_memory=2000,
             dot=numpy.dot, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

#    # if trial vectors are held in memory, store as many as possible
#    maxspace = max(int((max_memory-1e3)*1e6/x0.nbytes/2), maxspace)

    if hermi:
        toloose = max(tol, lindep)
        gen = davidson_cc(a, x0, precond, tol, max_cycle, maxspace,
                          nroots, hermi, lindep, max_memory, _real_lowest, dot)
        for istep, rdic in enumerate(gen):
            log.debug('davidson %d, subspace %d, |resid|=%g, e=%s, seig=%g',
                      istep, len(rdic['xs'][-1]), rdic['rnorm'],
                      str(rdic['e']), rdic['lindep'])
            if numpy.max(rdic['rnorm']) < toloose:
                break
        log.debug('final step %d', istep+1)
        return rdic['e'], rdic['v']
    else:
        gen = davidson_cc(a, x0, precond, tol, max_cycle, maxspace,
                          nroots, hermi, lindep, max_memory, _real_lowest, dot)
        for istep, rdic in enumerate(gen):
            log.debug('davidson %d, subspace %d, |resid(L,R)|=(%g,%g), e=%s, seig=%g',
                      istep, len(rdic['xs'][-1]),
                      rdic['rnorm'][0], rdic['rnorm'][1],
                      str(rdic['e']), rdic['lindep'])
        log.debug('final step %d', istep+1)
        return rdic['e'], rdic['v'][0], rdic['v'][1]

eigh = davidson
dsyev = davidson


# Sort real part of eigenvalues and pick the lowest ones
def _real_lowest(nroots, e, *args):
    return numpy.argsort(e.real)[:nroots]


def davidson_cc(a, x0, precond=None, tol=1e-14, max_cycle=50, maxspace=12,
                nroots=1, hermi=1, lindep=1e-16, max_memory=2000,
                eig_pick=_real_lowest, dot=numpy.dot):
    if hermi:
        return davidson_he_cc(a, x0, precond, tol, max_cycle, maxspace,
                              nroots, lindep, max_memory, eig_pick, dot)
    else:
        return davidson_ge_cc(a, x0, precond, tol, max_cycle, maxspace,
                              nroots, lindep, max_memory, eig_pick, dot)

#FIXME: unstable for nroots > 1?
def davidson_ge_cc(a, x0, precond=None, tol=1e-14, max_cycle=50, maxspace=12,
                   nroots=1, lindep=1e-16, max_memory=2000,
                   eig_pick=_real_lowest, dot=numpy.dot):
    assert(max_cycle >= maxspace >= nroots)

    max_cycle = max_cycle + nroots - 1
    heff = numpy.zeros((max_cycle+1,max_cycle+1), dtype=numpy.complex)
    ovlp = numpy.zeros((max_cycle+1,max_cycle+1), dtype=numpy.complex)
    e = numpy.zeros(nroots, dtype=numpy.complex)

    opl, opr = a
    if isinstance(x0, (tuple, list)):
        xls = [x0[0]]
        xrs = [x0[1]]
        ax  = []
    else:
        xls = [x0]
        xrs = [x0]
        ax  = []

    for istep in range(max_cycle):
        ax.append(opr(xrs[istep]))
        for j in range(istep):
            heff[istep,j] = dot(xls[istep].conj(), ax[j])
            heff[j,istep] = dot(xls[j].conj(), ax[istep])
            ovlp[istep,j] = dot(xls[istep].conj(), xrs[j])
            ovlp[j,istep] = dot(xls[j].conj(), xrs[istep])
        heff[istep,istep] = dot(xls[istep].conj(), ax[istep])
        ovlp[istep,istep] = dot(xls[istep].conj(), xrs[istep])

        nroots_now = min(nroots,istep+1)
        hnow = heff[:istep+1,:istep+1]
        snow = ovlp[:istep+1,:istep+1]
        w, vl, vr = scipy.linalg.eig(hnow, snow, left=True)
        eigidx = eig_pick(nroots_now, w, vl, vr, hnow, snow)
        w  = w[eigidx]
        vl = vl[:,eigidx]
        vr = vr[:,eigidx]
        seig = scipy.linalg.eig(ovlp[:istep+1,:istep+1])[0]

        xli = []
        xri = []
        axi = []
        for i in range(nroots_now):
            xli.append(xls[istep] * vl[istep,i])
            xri.append(xrs[istep] * vr[istep,i])
            axi.append(ax[istep]  * vr[istep,i])
        for k in reversed(range(istep)):
            for i in range(nroots_now):
                xli[i] += vl[k,i] * xls[k]
                xri[i] += vr[k,i] * xrs[k]
                axi[i] += vr[k,i] * ax[k]

        # scipy.linalg.eig does not return normalized dot(vl,vr)
        innerprod = []
        for i in range(nroots_now):
            innerprod.append(dot(xli[i].conj(), xri[i]))
            xli[i] *= 1/numpy.sqrt(innerprod[i])
            xri[i] *= 1/numpy.sqrt(innerprod[i])
            axi[i] *= 1/numpy.sqrt(innerprod[i])

        ril = []
        rir = []
        rrl = []
        rrr = []
        for i in range(nroots_now):
            rir.append(axi[i] - w[i] * xri[i])
            rrr.append(numpy.linalg.norm(rir[i]))
        if opl == opr:
            ril = rir
            rrl = rrr
        else:
            for i in range(nroots_now):
                ril.append(opl(xli[i]) - w[i].conj() * xli[i])
                rrl.append(numpy.linalg.norm(ril[i]))

        if ((abs(w[:nroots_now]-e[:nroots_now]).max() < tol) or
            (numpy.max(rrl) < tol and numpy.max(rrr) < tol) or
            (seig[0] < lindep)):
            break
        else:
            e[:nroots_now] = w[:nroots_now]

        if nroots == 1:
            yield {'e': w[0], 'v': (xli[0], xri[0]),
                   'rnorm': (rrl[0], rrr[0]), 'lindep':  seig[0],
                   'xs' : (xls, xrs), 'ax' : ax}
        elif istep+1 >= nroots:
            yield {'e': w[:nroots_now], 'v': (xli, xri),
                   'rnorm': (rrl, rrr), 'lindep':  seig[0],
                   'xs' : (xls, xrs), 'ax' : ax}

        #if len(xs) >= maxspace:
        #    xls.pop(0)
        #    xrs.pop(0)
        #    ax.pop(0)
        xls.append(precond(ril[0], w[0], xli[0], xri[0]))
        xrs.append(precond(rir[0], w[0], xli[0], xri[0]))

    if nroots == 1:
        yield {'e': w[0], 'v': (xli[0], xri[0]),
               'rnorm': (rrl[0], rrr[0]), 'lindep':  seig[0],
               'xs' : (xls, xrs), 'ax' : ax}
    else:
        yield {'e': w[:nroots_now], 'v': (xli, xri),
               'rnorm': (rrl, rrr), 'lindep':  seig[0],
               'xs' : (xls, xrs), 'ax' : ax}

# JCC, 22, 1574
def davidson_he_cc(a, x0, precond=None, tol=1e-14, max_cycle=50, maxspace=12,
                   nroots=1, lindep=1e-16, max_memory=2000,
                   eig_pick=_real_lowest, dot=numpy.dot):
    assert(max_cycle >= maxspace >= nroots)

    max_cycle = max_cycle + nroots - 1
    heff = numpy.zeros((max_cycle+1,max_cycle+1), dtype=x0.dtype)
    ovlp = numpy.zeros((max_cycle+1,max_cycle+1), dtype=x0.dtype)
    e = numpy.zeros(nroots)
    xs = [x0]
    ax = []
    subspace = 0
    for istep in range(max_cycle+nroots):
        ax.append(a(xs[istep]))
        for j in range(istep+1):
            heff[istep,j] = heff[j,istep] = dot(xs[istep].conj(), ax[j])
            ovlp[istep,j] = ovlp[j,istep] = dot(xs[istep].conj(), xs[j])
        nroots_now = min(nroots,istep+1)
        hnow = heff[:istep+1,:istep+1]
        snow = ovlp[:istep+1,:istep+1]
        w, v = scipy.linalg.eigh(hnow, snow)
        eigidx = eig_pick(nroots_now, w, v, hnow, snow)
        w = w[eigidx]
        v = v[:,eigidx]
        seig = scipy.linalg.eigh(ovlp[:istep+1,:istep+1])[0]

        xi  = []
        axi = []
        for i in range(nroots_now):
            xi .append(xs[istep] * v[istep,i])
            axi.append(ax[istep] * v[istep,i])
        for k in reversed(range(istep)):
            for i in range(nroots_now):
                xi [i] += v[k,i] * xs[k]
                axi[i] += v[k,i] * ax[k]

        ri = []
        rr = []
        for i in range(nroots_now):
            ri.append(axi[i] - w[i] * xi[i])
            rr.append(numpy.linalg.norm(ri[i]))

        if ((abs(w[:nroots_now]-e[:nroots_now]).max() < tol) or
            (numpy.max(rr) < tol) or (seig[0] < lindep)):
            break
        else:
            e[:nroots_now] = w[:nroots_now]

        if nroots == 1:
            yield {'e': w[0], 'v': xi[0],
                   'rnorm': rr[0], 'lindep':  seig[0], 'xs' : xs, 'ax' : ax}
        elif istep+1 >= nroots:
            yield {'e': w[:nroots_now], 'v': xi,
                   'rnorm': rr, 'lindep':  seig[0], 'xs' : xs, 'ax' : ax}

        #if len(xs) >= maxspace:
        #    xs.pop(0)
        #    ax.pop(0)
        xs.append(precond(ri[0], w[0], xi[0]))

    if nroots == 1:
        yield {'e': w[0], 'v': xi[0],
               'rnorm': rr[0], 'lindep':  seig[0], 'xs' : xs, 'ax' : ax}
    else:
        yield {'e': w[:nroots_now], 'v': xi,
               'rnorm': rr, 'lindep':  seig[0], 'xs' : xs, 'ax' : ax}


if __name__ == '__main__':
####################################
# Hermitian
    numpy.random.seed(12)
    n = 300
    a = numpy.random.random((n,n))
    a = a + a.T + numpy.diag(numpy.random.random(n))*10

    e, u = scipy.linalg.eigh(a)
    print(e[0], u[0,0])

    def aop(x):
        return numpy.dot(a, x)

    def precond(r, e, xs):
        return r / (a.diagonal() - e)

    x0 = u[:,0] + a[0]*1e-2
    x0 *= 1/numpy.linalg.norm(x0)
    e0,x0 = davidson(aop, x0, precond, max_cycle=50, maxspace=6, max_memory=.0001)
    print(e0 - e[0])

####################################
# Non-hermitian
    numpy.random.seed(12)
    n = 300
    a = numpy.random.random((n,n))
    a = a + numpy.diag(numpy.random.random(n))*10

    e,ul,ur = scipy.linalg.eig(a,left=True)
    print(e[0], ul[0,0], ur[0,0])

    aop =(lambda x: numpy.dot(a.T, x), lambda x: numpy.dot(a, x))

    def precond(r, e, *args):
        return r / (a.diagonal() - e)

    x0 = ur[:,0] + a[0]*1e-2
    x0 *= 1/numpy.linalg.norm(x0)
# Do not use general davidson algorithm, it's unstable due to the uncertainty
# of complex eigenvalue ordering
#    e0, xl, xr = davidson(aop, x0, precond, max_cycle=40, maxspace=6,
#                          max_memory=.0001, hermi=0)

# Stick on the states as close as possible to the initial guess
    last_eigs = [None]
    def eig_pick(nroots, e, vl, vr, h, s):
        if len(e) > nroots:
            eigidx = []
            for ei in last_eigs[0]:
                eigidx.append(numpy.argmin(abs(ei-e)))
            eigidx = numpy.array(eigidx)
            last_eigs[0] = e[eigidx]
            return eigidx
        else:
            last_eigs[0] = e
            return range(nroots)
    for rdic in davidson_cc(aop, x0, precond, tol=1e-10,
                            nroots=1, hermi=False, eig_pick=eig_pick):
        print rdic['e']
    e0 = rdic['e']
    vl, vr = rdic['v']
    print(e0 - e[0])
    lr = numpy.dot(numpy.array(vl).conj(), numpy.array(vr).T)
    print(numpy.allclose(numpy.eye(1), lr))
