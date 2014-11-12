#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import numpy
import scipy.linalg
import h5py

# default max_memory 2000 MB

def eigh(a, x0, precond, tol=1e-14, maxiter=50, maxspace=12, lindep=1e-16,
          max_memory=2000, eig_pick=None, dot=numpy.dot, verbose=0):
    return dsyev(a, x0, precond, tol, maxiter, maxspace, lindep, max_memory,
                 eig_pick, dot, verbose)

def dsyev(a, x0, precond, tol=1e-14, maxiter=50, maxspace=12, lindep=1e-16,
          max_memory=2000, eig_pick=None, dot=numpy.dot, verbose=0):

    toloose = numpy.sqrt(tol)
    # if trial vectors are held in memory, store as many as possible
    maxspace = max(int((max_memory-1e3)*1e6/x0.nbytes/2), maxspace)

    xs = _TrialXs(x0.nbytes, maxspace, max_memory)
    ax = _TrialXs(x0.nbytes, maxspace, max_memory)
    if eig_pick is None:
        eig_pick = lambda w, v: 0
    #e0_hist = []
    #def eig_pick(w, v):
    #    idx = 0
    #    if len(e0_hist) > 3:
    #        idx = numpy.argmin([abs(e0_hist[-1]-ei) for ei in w])
    #    e0_hist.append(w[idx])
    #    return idx

    heff = numpy.zeros((maxiter+1,maxiter+1))
    ovlp = numpy.zeros((maxiter+1,maxiter+1))
    v_prev = numpy.eye(1)
    e = 0
    for istep in range(min(maxiter,x0.size)):
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
            heff[subspace,i] = heff[i,subspace] = dot(xt, ax[i])
            ovlp[subspace,i] = ovlp[i,subspace] = dot(xt, xs[i])
        heff[subspace,subspace] = dot(xt, axt)
        ovlp[subspace,subspace] = dot(xt, xt)

        seig = scipy.linalg.eigh(ovlp[:subspace+1,:subspace+1])[0]
        if seig[0] < lindep:
            break
        w, v = scipy.linalg.eigh(heff[:subspace+1,:subspace+1], \
                                 ovlp[:subspace+1,:subspace+1])
        index = eig_pick(w, v)
        de = w[index] - e
        e = w[index]

## Seldomly, precond may produce unacceptable basis, which leads eigvalue collapse.
## clear basis and start the iteration again
#        nprev = len(v_prev)
#        sim = reduce(numpy.dot,(v_prev, ovlp[:nprev,:subspace+1],v[:,index]))
#        if abs(sim) < .8:
#            xs = _TrialXs(x0.nbytes, maxspace, max_memory)
#            ax = _TrialXs(x0.nbytes, maxspace, max_memory)
#            e = 0
#            continue

        x0  = xt  * v[subspace,index]
        ax0 = axt * v[subspace,index]
        for i in reversed(range(subspace)):
            x0  += v[i,index] * xs[i]
            ax0 += v[i,index] * ax[i]

        dx = ax0 - e * x0
        rr = numpy.linalg.norm(dx)
        if verbose:
            print('davidson', istep, subspace, rr, e, seig[0])

# floating size of subspace, prevent the new intital guess going too bad
        if subspace < maxspace \
           or (subspace<maxspace+2 and seig[0] > toloose):
            xs.append(xt)
            ax.append(axt)
        else:
# After several updates of the trial vectors, the trial vectors are highly
# linear dependent which seems reducing the accuracy. Removing all trial
# vectors and restarting iteration with better initial guess gives better
# accuracy, though more iterations are required.
            xs = _TrialXs(x0.nbytes, maxspace, max_memory)
            ax = _TrialXs(x0.nbytes, maxspace, max_memory)
            e = 0
        v_prev = v[:,index]


        if rr < toloose or abs(de) < tol or seig[0] < lindep:
            break

    if verbose:
        print('final step', istep)

    return e, x0


class _TrialXs(list):
    def __init__(self, xbytes, maxspace, max_memory):
        if xbytes*maxspace*2 > max_memory*1e6:
            _fd = tempfile.NamedTemporaryFile()
            self.scr_h5 = h5py.File(_fd.name, 'w')
        else:
            self.scr_h5 = None

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


if __name__ == '__main__':
    numpy.random.seed(12)
    n = 100
    #a = numpy.random.random((n,n))
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a))
    a = a + a.T

    e,u = numpy.linalg.eigh(a)
    #a = numpy.dot(u[:,:15]*e[:15], u[:,:15].T)
    print(e[0], u[0,0])

    def aop(x):
        return numpy.dot(a, x)

    def precond(x, e0, x0):
        return x / (a.diagonal() - e0)

    x0 = a[0]/numpy.linalg.norm(a[0])
    #x0 = (u[:,0]+.01)/numpy.linalg.norm(u[:,0]+.01)
    #print(dsyev(aop, x0, precond, eig_pick=lambda w,y: numpy.argmax(abs(w)<1e-4))[0])
    #print(dsyev(aop, x0, precond)[0] - -42.8144765196)
    e0,x0 = dsyev(aop, x0, precond, maxiter=30, maxspace=6, max_memory=.0001, verbose=9)
    print(e0 - e[0])
