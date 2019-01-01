from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_tools import find_nearrest_index

import numba as nb
from pyscf.nao.m_numba_utils import csphar_numba

@nb.jit(nopython=True)
def c2r_lm(conj_c2r, jmx, clm, clmm, m):
    """
        clm: sph harmonic l and m
        clmm: sph harmonic l and -m
        convert from real to complex spherical harmonic
        for an unique value of l and m
    """
    rlm = 0.0
    if m == 0:
        rlm = conj_c2r[jmx, jmx]*clm
    else:
        rlm = conj_c2r[m+jmx, m+jmx]*clm +\
                conj_c2r[m+jmx, -m+jmx]*clmm

    return rlm.real

@nb.jit(nopython=True)
def get_index_lm(l, m):
    """
        return the index of an array ordered as 
        [l=0 m=0, l=1 m=-1, l=1 m=0, l=1 m=1, ....]
    """
    return (l+1)**2 -1 -l + m


#@nb.jit(nopython=True, parallel=True)
def get_tem_potential_numba(time, R0, vnorm, vdir, center, rcut, inte1, 
        rr, dr, fr_val, conj_c2r, l, m, jmx, ind_lm, ind_lmm, V_time):
    """
        Numba version of the computation of the external potential in time
        for tem calculations
    """
    for it in nb.prange(time.shape[0]):
        R_sub = R0 + vnorm*vdir*(time[it] - time[0]) - center
        norm = np.sqrt(np.dot(R_sub, R_sub))

        if norm > rcut:
            I1 = inte1/(norm**(l+1))
            I2 = 0.0
        else:
            rsub_max = (np.abs(rr - norm)).argmin() # find_nearrest_index(rr, norm)

            I1 = np.sum(fr_val[0:rsub_max+1]*
                    rr[0:rsub_max+1]**(l+2)*rr[0:rsub_max+1])
            I2 = np.sum(fr_val[rsub_max+1:]*
                    rr[rsub_max+1:]/(rr[rsub_max+1:]**(l-1)))


            I1 = I1*dr/(norm**(l+1))
            I2 = I2*(norm**l)*dr
        clm_tem = csphar_numba(R_sub, l)
        clm = (4*np.pi/(2*l+1))*clm_tem[ind_lm]*(I1 + I2)
        clmm = (4*np.pi/(2*l+1))*clm_tem[ind_lmm]*(I1 + I2)
        rlm = c2r_lm(conj_c2r, jmx, clm, clmm, m)
        V_time[it] = rlm + 0.0j
