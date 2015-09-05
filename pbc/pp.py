import itertools
import math
import numpy as np
import scipy.linalg
import scipy.special


pi=math.pi
sqrt=math.sqrt
exp=math.exp

'''PP module.
   For GTH PPs, see Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
   Element  Znuc    Zion(here: Z)
   rloc     C1      C2      C3      C4
   rs       h1s     h2s
   rp       h1p
'''

class PP:
    def __init__(self, **kwargs):
        self.typ = None

        # EXAMPLE, B w/ LDA:
        Zion = 3
        rloc = 0.4324996
        C = [-5.6004798, 0.8062843]
        rl = [0.3738823, 0.0]
        hs = [6.2352212, 0.0]
        h1p = 0.0

        self.Zion = Zion
        # GTH-specific:
        self.rloc = rloc
        self.C = C
        # C[2],C[3] should be zero if necessary
        self.rl = rl   # a vector of length 2: [rs, rp]
        self.hs = hs   # a vector of length 2: [h1s, h2s]
        self.h1p = h1p # a scalar
        # hs and hp should be zeros as necessary 
        # (e.g. both zero for H,He,Li,Be)
        self.h = np.array( [ [hs[0], hs[1]], [h1p, 12345] ] ).transpose()

    def gth_vloc_r(r):
        '''
        local part of the GTH pseudopotential

        r: scalar or 1D np.array

        Returns 
             scalar or 1D np.array
        '''
        r_red = r/self.rloc
        return ( -self.Zion/r * math.erf(r_red/sqrt(2))
            + exp(-0.5*r_red**2)*(self.C[0] + self.C[1]*r_red**2
                + self.C[2]*r_red**4 + self.C[3]*r_red**6) )

    def gth_vnonloc_r(rvecs):
        '''
        all contributing (separable) projectors needed for the
        nonlocal part of the GTH pseudopotential

        rvecs: np.array [nrpts, 3]

        Returns
            np.array [nproj=6], np.array [nproj=6, nrpts]
        '''
        hs = []
        projs = []
        for [i,l,m] in [ [0,0,0], [1,0,0], [0,1,-1], [0,1,0], [0,1,1] ]:
            proj_vec = []
            for rvec in rvecs:
                r, theta, phi = self.cart2polar(rvec)
                proj_vec.append(self.proj_il(i,l,r)*self.Ylm(l,m,theta,phi))
            hs.append(self.h[i,l])
            projs.append(np.array(proj_vec))
        return np.array(hs), np.array(projs)

    def proj_il_r(i,l,r):
        rl = self.rl[l]
        l = l + i*2 # gives either l or l+2 for i=0,1
        return sqrt(2) * ( r**l * exp(-0.5*(r/rl)**2)
                  /(rl**(l+3/2.)*sqrt(math.gamma(l+3/2.))) )

    def Ylm(l,m,theta,phi):
        # Note: l and m are reversed in sph_harm
        return scipy.special.sph_harm(m,l,theta,phi)

    def cart2polar(rvec):
        x,y,z = rvec
        r = scipy.linalg.norm(rvec)
        theta = math.atan2(z,sqrt(x**2+y**2))
        phi = math.atan2(y,x)
        return r, theta, phi

