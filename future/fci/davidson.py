#!/usr/bin/env python

import numpy
import scipy.linalg

def dsyev(a, x0, precond, tol=1e-8, maxiter=20, eig_pick=None, dot=numpy.dot,
          verbose=0):
    lindep = tol**2
    #x0 = x0/numpy.linalg.norm(x0)
    xs = []
    ax = []
    if eig_pick is None:
        eig_pick = lambda w, v: 0

    heff = numpy.zeros((maxiter,maxiter))
    ovlp = numpy.zeros((maxiter,maxiter))
    e = 0
    for istep in range(min(maxiter,x0.size)):
        ax0 = a(x0)
        xs.append(x0)
        ax.append(ax0)
        for i in range(istep):
            heff[istep,i] = heff[i,istep] = dot(xs[istep], ax[i])
            ovlp[istep,i] = ovlp[i,istep] = dot(xs[istep], xs[i])
        heff[istep,istep] = dot(x0, ax0)
        ovlp[istep,istep] = dot(x0, x0)
        s0 = scipy.linalg.eigh(ovlp[:istep+1,:istep+1])[0][0]
        if s0 < lindep:
            break

        w, v = scipy.linalg.eigh(heff[:istep+1,:istep+1], \
                                 ovlp[:istep+1,:istep+1])
        index = eig_pick(w, v)
        e = w[index]

        xtrial = v[istep,index] * xs[istep]
        x1     = v[istep,index] * ax[istep]
        for i in reversed(range(istep)):
            xtrial += v[i,index] * xs[i]
            x1     += v[i,index] * ax[i]

        dx = x1
        dx += (-e) * xtrial
        rr = numpy.linalg.norm(dx)
        if verbose:
            print 'davidson', istep, rr, e
        if rr < tol:
            break
        x0 = precond(dx, e)

    if verbose:
        print 'final step', istep

    return e, xtrial

if __name__ == '__main__':
    n = 100
    #a = numpy.random.random((n,n))
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a))
    a = a + a.T

    e,u = numpy.linalg.eigh(a)
    #a = numpy.dot(u[:,:15]*e[:15], u[:,:15].T)
    print e[0]

    def aop(x):
        return numpy.dot(a, x)

    def precond(x, e0):
        return x / (a.diagonal() - e0)

    x0 = a[0]/numpy.linalg.norm(a[0])
    #x0 = (u[:,0]+.01)/numpy.linalg.norm(u[:,0]+.01)
    #print dsyev(aop, x0, precond, eig_pick=lambda w,y: numpy.argmax(abs(w)<1e-4))[0]
    print dsyev(aop, x0, precond)[0] - -42.8144765196
