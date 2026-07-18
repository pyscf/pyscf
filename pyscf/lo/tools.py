import numpy


def findiff_grad(func, x, delta=1e-4):
    ''' Finite-difference gradient
    '''
    x = numpy.asarray(x)
    n = x.size
    g = numpy.zeros_like(x)
    for i in range(n):
        dx = numpy.zeros_like(x)
        dx[i] = delta*0.5
        g[i] = (func(x+dx) - func(x-dx)) / delta
    return g

def findiff_hess(fgrad, x, delta=1e-4):
    ''' Finite-difference Hessian from gradient
    '''
    x = numpy.asarray(x)
    n = x.size
    h = numpy.zeros((n,n), dtype=x.dtype)
    for i in range(n):
        dxi = numpy.zeros_like(x)
        dxi[i] = delta*0.5
        h[i] = (fgrad(x+dxi) - fgrad(x-dxi)) / delta
    h = (h + h.T) * 0.5
    return h
