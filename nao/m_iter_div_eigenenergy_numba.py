from __future__ import division
import numba as nb
import numpy as np

@nb.jit
def div_eigenenergy_numba(ksn2e, ksn2f, nfermi, vstart, comega, nm2v):
    """
        multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
            frac{1.0}{w + (Em - En)})
        using numba
    """

    for n,[en,fn] in enumerate(zip(ksn2e[0,0,:nfermi], ksn2f[0,0,:nfermi])):
        for j,[em,fm] in enumerate(zip(ksn2e[0,0,n+1:], ksn2f[0,0,n+1:])):
            m = j+n+1-vstart
            nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /\
                    (comega + (em - en)) )
