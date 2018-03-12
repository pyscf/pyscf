# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import numpy as np
import warnings

import numba as nb

#@nb.njit(parallel=True)
@nb.jit(nopython=True)
def get_bessel_xjl_numba(kk, dist, j, nr):
    '''
    Calculate spherical bessel functions j_l(k*r) for a given r and on a grid of momentas k given by the array kk
    Args:
    kk : 1D array (float): k grid
    r : (float) radial coordinate
    l : (integer) angular momentum
    nr: (integer) k grid dimension
    Result:
    xj[1:2*j+1, 1:nr] : 2D array (float)
    '''

    bessel_pp = np.zeros((j*2+1, nr), dtype=np.float64)

    lc = 2*j
    for ip, p in enumerate(kk):
        # Computes a table of j_l(x) for fixed xx, Eq. (39)
        # p = kk[ip]
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

@nb.jit(nopython=True)
def calc_oo2co(bessel_pp, dg_jt, ao1_sp2info_sp1, ao1_sp2info_sp2,
        ao1_psi_log_mom_sp1, ao1_psi_log_mom_sp2,
        njm, gaunt_iptr, gaunt_data, ylm,
        j, jmx, tr_c2r, conj_c2r, l2S, cS, rS, cmat, oo2co):

    for v2 in range(ao1_sp2info_sp2.shape[0]):
        mu2 = ao1_sp2info_sp2[v2, 0]
        l2 = ao1_sp2info_sp2[v2, 1]
        s2 = ao1_sp2info_sp2[v2, 2]
        f2 = ao1_sp2info_sp2[v2, 3]
        for v1 in range(ao1_sp2info_sp1.shape[0]):
            mu1 = ao1_sp2info_sp1[v1, 0]
            l1 = ao1_sp2info_sp1[v1, 1]
            s1 = ao1_sp2info_sp1[v1, 2]
            f1 = ao1_sp2info_sp1[v1, 3]

            f1f2_mom = ao1_psi_log_mom_sp2[mu2,:] * ao1_psi_log_mom_sp1[mu1,:]
            for il2S in range(l2S.shape[0]):
                l2S[il2S] = 0.0

            for l3 in range( abs(l1-l2), l1+l2+1):
                l2S[l3] = (f1f2_mom[:]*bessel_pp[l3,:]).sum() + f1f2_mom[0]*bessel_pp[l3,0]/dg_jt*0.995

            #cS.fill(0.0)
            for icS1 in range(cS.shape[0]):
                for icS2 in range(cS.shape[1]):
                    cS[icS1, icS2] = 0.0

            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    #gc,m3 = self.get_gaunt(l1,-m1,l2,m2), m2-m1
                    m3 = m2-m1
                    i1 = l1*(l1+1)-m1
                    i2 = l2*(l2+1)+m2
                    ind = i1*njm+i2
                    s,f = gaunt_iptr[ind], gaunt_iptr[ind+1]
                    gc = gaunt_data[s:f]
                    for l3ind,l3 in enumerate(range(abs(l1-l2),l1+l2+1)):
                        if abs(m3) > l3 : continue
                        cS[m1+j,m2+j] = cS[m1+j,m2+j] + l2S[l3]*ylm[ l3*(l3+1)+m3] *\
                            gc[l3ind] * (-1.0)**((3*l1+l2+l3)//2+m2)
            c2r_numba(j, tr_c2r, conj_c2r, l1, l2, jmx, cS, rS, cmat)
            oo2co[s1:f1,s2:f2] = rS[-l1+j:l1+j+1,-l2+j:l2+j+1]

@nb.jit(nopython=True)
def c2r_numba(j, tr_c2r, conj_c2r, j1,j2, jm,cmat,rmat,mat):
    #assert(type(mat[0,0])==np.complex128)
    #mat.fill(0.0)
    #rmat.fill(0.0)
    for mm1 in range(-j1,j1+1):
        for mm2 in range(-j2,j2+1):
            if mm2 == 0 :
                mat[mm1+jm,mm2+jm] = cmat[mm1+jm,mm2+jm]*tr_c2r[mm2+j,mm2+j]
            else :
                mat[mm1+jm,mm2+jm] = \
                    (cmat[mm1+jm,mm2+jm]*tr_c2r[mm2+j,mm2+j] + \
                    cmat[mm1+jm,-mm2+jm]*tr_c2r[-mm2+j,mm2+j])
                #if j1==2 and j2==1:
                #  print( mm1,mm2, mat[mm1+jm,mm2+jm] )

    for mm2 in range(-j2,j2+1):
        for mm1 in range(-j1,j1+1):
            if mm1 == 0 :
                rmat[mm1+jm, mm2+jm] = \
                        (conj_c2r[mm1+j,mm1+j]*mat[mm1+jm,mm2+jm]).real
            else :
                rmat[mm1+jm, mm2+jm] = \
                    (conj_c2r[mm1+j,mm1+j] * mat[mm1+jm,mm2+jm] + \
                     conj_c2r[mm1+j,-mm1+j] * mat[-mm1+jm,mm2+jm]).real
