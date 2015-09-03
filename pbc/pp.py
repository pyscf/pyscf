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
        Z = 3
        rloc = 0.4324996
        C = [-5.6004798, 0.8062843]
        rl = [0.3738823, 0.0]
        hs = [6.2352212, 0.0]
        h1p = 0.0

        self.Z = Z
        # GTH-specific:
        self.rloc = rloc
        self.C = C
        # C[2],C[3] should be zero if necessary
        self.rl = rl   # a vector of length 2: [rs, rp]
        self.hs = hs   # a vector of length 2: [h1s, h2s]
        self.h1p = h1p # a scalar
        # hs and hp should be zeros as necessary 
        # (e.g. both zero for H,He,Li,Be)


    def v_gth_loc(r):
        # Beware division by zero
        r_red = r/(self.rloc+1e-10)
        return ( -self.Z/r * np.erf(r_red/sqrt(2))
            + exp(-0.5*r_red**2)*(self.C[0] + self.C[1]*r_red**2
                + self.C[2]*r_red**4 + self.C[3]*r_red**6) )

    def v_gth_nonloc(rvec,rpvec):
        # rvec and rpvec are vectors of length 3
        # currently assumed to be referenced to the atom center
        r, theta, phi = self.cart2polar(rvec)
        rp, thetap, phip = self.cart2polar(rpvec)
        sum_i = 0.
        sum_m = 0.
        for i in range(2):
            sum_i += ( self.Ylm(0,0,theta,phi)*self.proj_il(i,0,r)*hs[i]
                        *self.proj_il(i,0,rp)*self.Ylm(0,0,thetap,phip).conjugate() )
        for m in [-1,0,1]:
            sum_m += ( self.Ylm(1,m,theta,phi)*self.proj_il(i,1,r)*h1p
                        *self.proj_il(i,1,rp)*self.Ylm(1,m,thetap,phip).conjugate() )
        return sum_i + sum_m
        
    def proj_il(i,l,r):
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

