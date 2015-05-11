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

# default max_memory 2000 MB

def davidson(a, x0, precond, tol=1e-14, max_cycle=50, max_space=12, lindep=1e-16,
             max_memory=2000, eig_pick=None, dot=numpy.dot, callback=None,
             verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)
    toloose = numpy.sqrt(tol)
    # if trial vectors are held in memory, store as many as possible
    max_space = max(int((max_memory-1e3)*1e6/x0.nbytes/2), max_space)

    xs = _TrialXs(x0.nbytes, max_space, max_memory)
    ax = _TrialXs(x0.nbytes, max_space, max_memory)
    if eig_pick is None:
        eig_pick = lambda w, v: 0
    #e0_hist = []
    #def eig_pick(w, v):
    #    idx = 0
    #    if len(e0_hist) > 3:
    #        idx = numpy.argmin([abs(e0_hist[-1]-ei) for ei in w])
    #    e0_hist.append(w[idx])
    #    return idx

    heff = numpy.zeros((max_cycle+1,max_cycle+1), dtype=x0.dtype)
    ovlp = numpy.zeros((max_cycle+1,max_cycle+1), dtype=x0.dtype)
    e = 0
    for istep in range(min(max_cycle,x0.size)):
        subspace = len(xs)
        if subspace == 0:
            ax0 = dx = None
            xt = x0
        else:
            ax0 = None
            xt = precond(dx, e, x0)
            dx = None
        axt = a(xt)
        for i in range(subspace):
            heff[subspace,i] = heff[i,subspace] = dot(xt.conj(), ax[i])
            ovlp[subspace,i] = ovlp[i,subspace] = dot(xt.conj(), xs[i])
        heff[subspace,subspace] = dot(xt.conj(), axt)
        ovlp[subspace,subspace] = dot(xt.conj(), xt)

        w, v = scipy.linalg.eigh(heff[:subspace+1,:subspace+1], \
                                 ovlp[:subspace+1,:subspace+1])
        index = eig_pick(w, v)
        de = w[index] - e
        e = w[index]

        x0  = xt  * v[subspace,index]
        ax0 = axt * v[subspace,index]
        for i in reversed(range(subspace)):
            x0  += v[i,index] * xs[i]
            ax0 += v[i,index] * ax[i]

        seig = scipy.linalg.eigh(ovlp[:subspace+1,:subspace+1])[0]
        dx = ax0 - e * x0
        rr = numpy.linalg.norm(dx)
        log.debug('davidson %d %d, rr=%g, e=%.12g, seig=%g',
                  istep, subspace, rr, e, seig[0])

        if rr/numpy.sqrt(rr.size) < tol or abs(de) < tol or seig[0] < lindep:
            break

# refresh and restart
# floating size of subspace, prevent the new intital guess going too bad
        if subspace < max_space \
           or (subspace<max_space+2 and seig[0] > toloose):
            xs.append(xt)
            ax.append(axt)
        else:
# After several updates of the trial vectors, the trial vectors are highly
# linear dependent which seems reducing the accuracy. Removing all trial
# vectors and restarting iteration with better initial guess gives better
# accuracy, though more iterations are required.
            xs = _TrialXs(x0.nbytes, max_space, max_memory)
            ax = _TrialXs(x0.nbytes, max_space, max_memory)
            e = 0

        if callable(callback):
            callback(istep, xs, ax)

    log.debug('final step %d', istep)

    return e, x0

eigh = davidson
dsyev = davidson


class _TrialXs(list):
    def __init__(self, xbytes, max_space, max_memory):
        if xbytes*max_space*2 > max_memory*1e6:
            _fd = tempfile.NamedTemporaryFile()
            self.scr_h5 = h5py.File(_fd.name, 'w')
        else:
            self.scr_h5 = None
    def __del__(self):
        if self.scr_h5 is not None:
            self.scr_h5.close()

    def __getitem__(self, n):
        if self.scr_h5 is None:
            return list.__getitem__(self, n)
        else:
            return self.scr_h5[str(n)]

    def append(self, x):
        if self.scr_h5 is None:
            list.append(self, x)
        else:
            n = len(self.scr_h5)
            self.scr_h5[str(n)] = x

    def __setitem__(self, n, x):
        if self.scr_h5 is None:
            list.__setitem__(self, n, x)
        else:
            self.scr_h5[str(n)] = x

    def __len__(self):
        if self.scr_h5 is None:
            return list.__len__(self)
        else:
            return len(self.scr_h5)

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
        log.debug('krylov cycle %d, r = %g', cycle, numpy.sqrt(innerprod[-1]))
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

    x0 = a[0]/numpy.linalg.norm(a[0])
    #x0 = (u[:,0]+.01)/numpy.linalg.norm(u[:,0]+.01)
    #print(dsyev(aop, x0, precond, eig_pick=lambda w,y: numpy.argmax(abs(w)<1e-4))[0])
    #print(dsyev(aop, x0, precond)[0] - -42.8144765196)
    e0,x0 = dsyev(aop, x0, precond, max_cycle=30, max_space=6, max_memory=.0001)
    print(e0 - e[0])

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
