import numpy as np
import warnings

import numba as nb

@nb.njit(parallel=True)
def get_bessel_xjl_numba(kk, dist, j, nr):
    '''
    Calculate spherical bessel functions in all k space
    Args:
    kk : 1D array (float): k grid
    dist : (float) distance between pairs??
    j : (integer) angular momentum
    nr: (integer) k grid dimension
    Result:
    xj[1:2*j+1, 1:nr] : 2D array (float)
    '''

    bessel_pp = np.zeros((j*2+1, nr), dtype=np.float64)

    lc = 2*j
    for ip in nb.prange(nr):
        # Computes a table of j_l(x) for fixed xx, Eq. (39)
        p = kk[ip]
        xx = p*dist
        if (lc<-1): raise ValueError("lc < -1")
      
        xj = np.zeros((lc+1), dtype=np.float64)
        if abs(xx)<1.0e-10:
            xj[0] = 1.0
            bessel_pp[:, ip] = xj*p
            continue

        sin_xx_div_xx = np.sin(xx)/xx
        if xx < 0.75*lc :
            aam,aa,bbm,bb,sa,qqm = 1.0, (2*lc+1)/xx, 0.0, 1.0, -1.0, 1e10
            for k in range(1,51):
                sb = (2*(lc+k)+1)/xx
                aap,bbp = sb*aa+sa*aam,sb*bb+sa*bbm
                aam,bbm = aa,bb
                aa,bb   = aap,bbp
                qq      = aa/bb
                if abs(qq-qqm)<1.0e-15 : break
                qqm = qq

            xj[lc] = 1.0
            if lc > 0 : 
                xj[lc-1] = qq
                if lc > 1 :
                    for l in range(lc-1,0,-1):
                        xj[l-1] = (2*l+1)*xj[l]/xx-xj[l+1]
            cc = sin_xx_div_xx/xj[0]
            for l in range(lc+1): xj[l] = cc*xj[l]
        else :
            xj[0] = sin_xx_div_xx
            if lc > 0: 
                xj[1] = xj[0]/xx-np.cos(xx)/xx
                if lc > 1:
                    for l in range(1,lc): 
                        xj[l+1] = (2*l+1)*xj[l]/xx-xj[l-1]
        bessel_pp[:, ip] = xj*p
    return bessel_pp
