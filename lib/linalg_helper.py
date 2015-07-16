#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf.lib import logger


def safe_eigh(h, s, lindep=1e-15):
    seig, t = scipy.linalg.eigh(s)
    if seig[0] < lindep:
        idx = seig >= lindep
        t = t[:,idx] * (1/numpy.sqrt(seig[idx]))
        heff = reduce(numpy.dot, (t.T, h, t))
        w, v = scipy.linalg.eigh(heff)
        v = numpy.dot(t, v)
    else:
        w, v = scipy.linalg.eigh(h, s)
    return w, v, seig

# default max_memory 2000 MB
def davidson(a, x0, precond, tol=1e-14, max_cycle=50, max_space=12,
             lindep=1e-16, max_memory=2000, dot=numpy.dot, callback=None,
             nroots=1, lessio=False, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        xt = [x0]
        axt = [a(x0)]
        max_cycle = min(max_cycle,x0.size)
    else:
        xt = [xi for xi in x0]
        axt = [a(xi) for xi in x0]
        max_cycle = min(max_cycle,x0[0].size)

    max_space = max_space + nroots * 2
    if max_memory*1e6/xt[0].nbytes/2 > max_space+nroots*2:
        xs = []
        ax = []
        _incore = True
    else:
        xs = _Xlist()
        ax = _Xlist()
        _incore = False

    toloose = numpy.sqrt(tol) * 1e-2
    head = 0
    rnow = len(xt)

    for i in range(rnow):
        xs.append(xt[i])
        ax.append(axt[i])

    heff = numpy.empty((max_space,max_space), dtype=x0[0].dtype)
    ovlp = numpy.empty((max_space,max_space), dtype=x0[0].dtype)
    e = 0
    for icyc in range(max_cycle):
        space = len(xs)
        for i in range(space):
            if head <= i < head+rnow:
                for k in range(i-head+1):
                    heff[head+k,i] = dot(xt[k].conj(), axt[i-head])
                    ovlp[head+k,i] = dot(xt[k].conj(), xt [i-head])
                    heff[i,head+k] = heff[head+k,i].conj()
                    ovlp[i,head+k] = ovlp[head+k,i].conj()
            else:
                for k in range(rnow):
                    heff[head+k,i] = dot(xt[k].conj(), ax[i])
                    ovlp[head+k,i] = dot(xt[k].conj(), xs[i])
                    heff[i,head+k] = heff[head+k,i].conj()
                    ovlp[i,head+k] = ovlp[head+k,i].conj()

        w, v, seig = safe_eigh(heff[:space,:space], ovlp[:space,:space])
        try:
            de = w[:nroots] - e
        except ValueError:
            de = w[:nroots]
        e = w[:nroots]

        x0 = []
        ax0 = []
        if lessio and not _incore:
            for k, ek in enumerate(e):
                x0 .append(xs[space-1] * v[space-1,k])
            for i in reversed(range(space-1)):
                xsi = xs[i]
                for k, ek in enumerate(e):
                    x0 [k] += v[i,k] * xsi
            ax0 = [a(xi) for xi in x0]
        else:
            for k, ek in enumerate(e):
                x0 .append(xs[space-1] * v[space-1,k])
                ax0.append(ax[space-1] * v[space-1,k])
            for i in reversed(range(space-1)):
                xsi = xs[i]
                axi = ax[i]
                for k, ek in enumerate(e):
                    x0 [k] += v[i,k] * xsi
                    ax0[k] += v[i,k] * axi

        head += rnow
        dx = []
        dx_norm = []
        xt = []
        axt = []
        for k, ek in enumerate(e):
            dxtmp = ax0[k] - ek * x0[k]
            dxtmp_norm = numpy.linalg.norm(dxtmp)
            if dxtmp_norm > toloose or abs(de[k]) > tol:
                dx.append(dxtmp)
                dx_norm.append(dxtmp_norm)

                #xt.append(precond(dxtmp, ek, x0[k])) # not accurate enough?
                xt.append(precond(dxtmp, e[0], x0[k]))
                xt[-1] *= 1/numpy.linalg.norm(xt[-1])
                axt.append(a(xt[-1]))
        rnow = len(xt)
# Cannot require both dx_norm and de converged, because we want to stick on
# the states associated with the initial guess.  Numerical instability can
# break the symmetry restriction and move to lower state, eg BeH2 CAS(2,2)
        if rnow > 0:
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  seig= %4.3g',
                      icyc, space, max(dx_norm), e, seig[0])
        if (rnow == 0 or seig[0] < lindep or
            (max(dx_norm) < toloose or max(abs(de)) < tol)):
            break

        if head+rnow > max_space:
            head = nroots
            for k in range(nroots):
                xs[k] = x0[k]
                ax[k] = ax0[k]
            tmp = numpy.dot(heff[nroots:space,:space],v[:,:nroots])
            heff[nroots:space,:nroots] = tmp
            heff[:nroots,nroots:space] = tmp.T.conj()
            heff[:nroots,:nroots] = numpy.diag(e)
            tmp = numpy.dot(ovlp[nroots:space,:space],v[:,:nroots])
            ovlp[nroots:space,:nroots] = tmp
            ovlp[:nroots,nroots:space] = tmp.T.conj()
            ovlp[:nroots,:nroots] = numpy.eye(nroots)

        for k in range(rnow):
            if head + k >= space:
                xs.append(xt[k])
                ax.append(axt[k])
            else:
                xs[head+k] = xt[k]
                ax[head+k] = axt[k]

        if callable(callback):
            callback(locals())

    if nroots == 1:
        return e[0], x0[0]
    else:
        return e, x0

eigh = davidson
dsyev = davidson


# Krylov subspace method
# ref: J. A. Pople, R. Krishnan, H. B. Schlegel, and J. S. Binkley,
#      Int. J.  Quantum. Chem.  Symp. 13, 225 (1979).
# solve (1+aop) x = b
def krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=numpy.dot, \
           lindep=1e-16, callback=None, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if x0 is not None:
        b = b - (x0 + aop(x0))

    xs = [b]
    ax = [aop(xs[0])]
    innerprod = [dot(xs[0].conj(), xs[0])]

    h = numpy.empty((max_cycle,max_cycle), dtype=b.dtype)
    for cycle in range(max_cycle):
        x1 = ax[-1].copy()
# Schmidt orthogonalization
        for i in range(cycle+1):
            s12 = h[i,cycle] = dot(xs[i].conj(), ax[-1])        # (*)
            x1 -= (s12/innerprod[i]) * xs[i]
        h[cycle,cycle] += innerprod[cycle]                      # (*)
        innerprod.append(dot(x1.conj(), x1).real)
        log.debug('krylov cycle %d  r = %g', cycle, numpy.sqrt(innerprod[-1]))
        if innerprod[-1] < lindep:
            break
        xs.append(x1)
        ax.append(aop(x1))

        if callable(callback):
            callback(cycle, xs, ax)

    log.debug('final cycle = %d', cycle)

    nd = cycle + 1
# h = numpy.dot(xs[:nd], ax[:nd].T) + numpy.diag(innerprod[:nd])
# to reduce IO, move upper triangle (and diagonal) part to (*)
    for i in range(nd):
        for j in range(i):
            h[i,j] = dot(xs[i].conj(), ax[j])
    g = numpy.zeros(nd, dtype=b.dtype)
    g[0] = innerprod[0]
    c = numpy.linalg.solve(h[:nd,:nd], g)
    x = xs[0] * c[0]
    for i in range(1, len(c)):
        x += c[i] * xs[i]

    if x0 is not None:
        x += x0
    return x

# Davidson-like linear eq solver.  It does not work well.
def dsolve(a, b, precond, tol=1e-14, max_cycle=30, dot=numpy.dot, \
           lindep=1e-16, verbose=0):

    toloose = numpy.sqrt(tol)

    xs = [precond(b)]
    ax = [a(xs[-1])]

    aeff = numpy.zeros((max_cycle,max_cycle), dtype=b.dtype)
    beff = numpy.zeros((max_cycle), dtype=b.dtype)
    for istep in range(max_cycle):
        beff[istep] = dot(xs[istep], b)
        for i in range(istep+1):
            aeff[istep,i] = dot(xs[istep], ax[i])
            aeff[i,istep] = dot(xs[i], ax[istep])

        v = scipy.linalg.solve(aeff[:istep+1,:istep+1], beff[:istep+1])
        xtrial = dot(v, xs)
        dx = b - dot(v, ax)
        rr = numpy.linalg.norm(dx)
        if verbose:
            print('davidson', istep, rr)
        if rr < toloose:
            break
        xs.append(precond(dx))
        ax.append(a(xs[-1]))

    if verbose:
        print(istep)

    return xtrial

class _Xlist(list):
    def __init__(self):
        self._fd = tempfile.NamedTemporaryFile()
        self.scr_h5 = h5py.File(self._fd.name, 'w')
        self.index = []
    def __del__(self):
        self.scr_h5.close()

    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[key].value

    def append(self, x):
        key = str(len(self.index) + 1)
        if key in self.index:
            for i in range(len(self.index)+1):
                if str(i) not in self.index:
                    key = str(i)
                    break
        self.index.append(key)
        self.scr_h5[key] = x
        self.scr_h5.flush()

    def __setitem__(self, n, x):
        key = self.index[n]
        self.scr_h5[key][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        self.index.pop(index)


if __name__ == '__main__':
    numpy.random.seed(12)
    n = 1000
    #a = numpy.random.random((n,n))
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a))
    a = a + a.T + numpy.diag(numpy.random.random(n))*10

    e,u = scipy.linalg.eigh(a)
    #a = numpy.dot(u[:,:15]*e[:15], u[:,:15].T)
    print(e[0], u[0,0])

    def aop(x):
        return numpy.dot(a, x)

    def precond(r, e0, x0):
        idx = numpy.argwhere(abs(x0)>.1).ravel()
        #idx = numpy.arange(20)
        m = idx.size
        if m > 2:
            h0 = a[idx][:,idx] - numpy.eye(m)*e0
            h0x0 = x0 / (a.diagonal() - e0)
            h0x0[idx] = numpy.linalg.solve(h0, h0x0[idx])
            h0r = r / (a.diagonal() - e0)
            h0r[idx] = numpy.linalg.solve(h0, r[idx])
            e1 = numpy.dot(x0, h0r) / numpy.dot(x0, h0x0)
            x1 = (r - e1*x0) / (a.diagonal() - e0)
            x1[idx] = numpy.linalg.solve(h0, (r-e1*x0)[idx])
            return x1
        else:
            return r / (a.diagonal() - e0)

    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),
          a[2]/numpy.linalg.norm(a[2]),
          a[3]/numpy.linalg.norm(a[3])]
    e0,x0 = dsyev(aop, x0, precond, max_cycle=30, max_space=12,
                  max_memory=.0001, verbose=5, nroots=4)
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])

##########
    a = a + numpy.diag(numpy.random.random(n)+1.1)* 10
    b = numpy.random.random(n)
    def aop(x):
        return numpy.dot(a,x)
    def precond(x, *args):
        return x / a.diagonal()
    x = numpy.linalg.solve(a, b)
    x1 = dsolve(aop, b, precond, max_cycle=50)
    print(abs(x - x1).sum())
    a_diag = a.diagonal()
    log = logger.Logger(sys.stdout, 5)
    aop = lambda x: numpy.dot(a-numpy.diag(a_diag), x)/a_diag
    x1 = krylov(aop, b/a_diag, max_cycle=50, verbose=log)
    print(abs(x - x1).sum())
    x1 = krylov(aop, b/a_diag, None, max_cycle=10, verbose=log)
    x1 = krylov(aop, b/a_diag, x1, max_cycle=30, verbose=log)
    print(abs(x - x1).sum())
