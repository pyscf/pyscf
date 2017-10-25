from __future__ import division
import numba as nb
import numpy as np

@nb.jit(nopython=True)
def div_eigenenergy_numba(ksn2e, ksn2f, nfermi, vstart, comega, nm2v_re, nm2v_im, norbs):
    """
        multiply the temporary matrix by (fn - fm) (frac{1.0}{w - (Em-En) -1} -
            frac{1.0}{w + (Em - En)})
        using numba
    """

    for n in range(nfermi):
        en = ksn2e[n]
        fn = ksn2f[n]
        for j in range(n+1, norbs, 1):
            em = ksn2e[j]
            fm = ksn2f[j]
            m = j - vstart

            nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            nm2v = nm2v * (fn-fm) * ( 1.0 / (comega - (em - en)) - 1.0 /\
                    (comega + (em - en)) )

            nm2v_re[n, m] = nm2v.real
            nm2v_im[n, m] = nm2v.imag

@nb.jit(nopython=True)
def mat_mul_numba(a, b):
    return a*b
