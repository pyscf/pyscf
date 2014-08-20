#!/usr/bin/env python

import numpy
import scipy.linalg

def pick_large_mode(e, v):
    # pick the first mode, so that davidson can stay on one mode
    #index = numpy.argmax(abs(v[0]) > .01)
    for index in range(v.shape[1]):
        if abs(v[0,index]) > .1:
            break
#FIXME should I pick the most siginificant mode of hessian? instable
    #for index in reversed(range(v.shape[1])):
    #    if abs(v[0,index]) > .1:
    #        break
#FIXME: should I ignore the smallest mode as it may increase the energy?
    #index = numpy.argmax(abs(v[0,1:]) > .01) + 1
    if abs(v[0,index]) < .1:
        return -1
    else:
        return index

# IJQC, 109, 2178
# use davidson algorithm to solve augmented hessian  Ac = ce
# c is the trial vector = (1, xtrial)
def davidson(h_op, g, precond, x0, log, tol=1e-7, max_cycle=10, max_stepsize=.6,
             lindep=1e-14, pick_mode=pick_large_mode, dot=numpy.dot):
    # the first trial vector is (1,0,0,...), which is not included in xs
    x0 = x0/numpy.linalg.norm(x0)
    xs = []
    ax = []

    lambda0 = 1
    heff = numpy.zeros((max_cycle+1,max_cycle+1))
    ovlp = numpy.eye(max_cycle+1)
    for istep in range(min(max_cycle,x0.size)):
        xs.append(x0)
        ax.append(h_op(x0))
        heff[istep+1,0] = heff[0,istep+1] = dot(xs[istep], g)
        for i in range(istep+1):
            heff[istep+1,i+1] = heff[i+1,istep+1] = dot(xs[istep], ax[i])
            ovlp[istep+1,i+1] = ovlp[i+1,istep+1] = dot(xs[istep], xs[i])
        nvec = len(xs) + 1
#FIXME: the lambda method used by ORZ does not help convergence
#        xtrial, w_t, v_t, index, lambda0 = \
#                _opt_step_as_orz_lambda(heff[:nvec,:nvec],ovlp[:nvec,:nvec], \
#                                        xs, pick_mode, lambda0, max_stepsize)
        xtrial, w_t, v_t, index = \
                _regular_step(heff[:nvec,:nvec], ovlp[:nvec,:nvec], xs, pick_mode)
        if index == -1:
            # use previous trial vector
            xtrial = _regular_step(heff[:nvec-1,:nvec-1], ovlp[:nvec-1,:nvec-1],
                                   xs[:-1], pick_mode)[0]
            break
        hx = _dgemv(v_t[1:], ax)
        # note g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - xtrial * (w_t*v_t[0])
        norm_dx = numpy.linalg.norm(dx)
# note that linear dependence of trial-vectors is very common, it always
# causes numerical problems in CASSCF
        s0 = numpy.linalg.eigh(ovlp[:nvec,:nvec])[0][0]
        if norm_dx < tol or s0 < lindep:
            break
        log.debug1('AH step %d, index=%d, |dx|=%.5g, lambda=%.5g, eig=%.5g, v[0]=%.5g, lindep=%.5g', \
                   istep, index, norm_dx, lambda0, w_t, v_t[0], s0)
#FIXME: The hessian can be very small sometime, the precond may have problems
        x0 = precond(dx, w_t)

    norm_x = numpy.linalg.norm(xtrial)
    if norm_x > max_stepsize:
        xtrial *= (max_stepsize/norm_x)
    log.debug('aug_hess: total step %d, |g| = %4.3g, |x| = %4.3g, max(|x|) = %4.3g, eig = %4.3g',
              istep+1, numpy.linalg.norm(g), norm_x, numpy.max(abs(xtrial)), w_t)
    return w_t, xtrial

# As did in orz, optimize the stepsize by searching the best lambda
def _opt_step_as_orz_lambda(heff, ovlp, xs, pick_mode, lambda0,
                            max_stepsize=.2, lambda_tries=6):
    lambdas = [lambda0]
    steps = []
    htest = heff.copy()
    xtrial = 0
    for itry in range(lambda_tries):
        w, v = scipy.linalg.eigh(htest, ovlp)
        # should it be < 1.1, like orz did?
        # unless the ovlp is quite singular, v > 1.1 is not likely to happen
        index = pick_mode(w, v)
        if index == -1:
            return xtrial, w_t, v[:,0], index, lambdas[-1]

        w_t = w[index]
        xtrial = _dgemv(v[1:,index]/v[0,index], xs)
        stepsize = numpy.linalg.norm(xtrial)
        steps.append(stepsize*(1./lambdas[-1]))
        if (stepsize - max_stepsize)/max_stepsize < .001:
            return xtrial, w_t, v[:,index], index, lambdas[-1]
        elif itry < 2:
            lambdas.append(stepsize/max_stepsize)
            htest[1:,1:] *= 1/lambdas[-1]
        else:
            # extrapolating lambda, find lambda that can produce max_stepsize
            lambdas.append(stepsize/max_stepsize)
            k = (lambdas[-2]-lambdas[-1]) / (steps[-2]-steps[-1])
            lambdax = (max_stepsize - steps[-1])*k + lambdas[-1]
            htest[1:,1:] *= 1/lambdax
    return xtrial, w_t, v[:,index], index, lambdax

def _regular_step(heff, ovlp, xs, pick_mode):
    w, v = scipy.linalg.eigh(heff, ovlp)
    index = pick_mode(w, v)
    if index == -1:
        return 0, 0, v[:,0], index
    else:
        w_t = w[index]
        xtrial = _dgemv(v[1:,index]/v[0,index], xs)
        return xtrial, w_t, v[:,index], index

def _dgemv(v, m):
    vm = v[0] * m[0]
    for i,vi in enumerate(v[1:]):
        vm += vi * m[i+1]
    return vm


if __name__ == '__main__':
    numpy.random.seed(15)
    heff = numpy.random.random((5,5))
    heff = heff + heff.T
    seff = numpy.eye(5)
    xs = numpy.random.random((4,20))
    seff[1:,1:] = numpy.dot(xs, xs.T)
    print _opt_step_as_orz_lambda(heff, seff, xs, pick_large_mode, 1, max_stepsize=.9)
