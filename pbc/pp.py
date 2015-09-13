import itertools
import math
import numpy as np
import scipy.linalg
import scipy.special

import pbc

pi=math.pi
sqrt=math.sqrt
exp=math.exp

'''PP module.
   For GTH PPs, see Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
'''

def get_vlocG(cell, gs):
    '''
    Local PP kernel in G space (Vloc(G) for G!=0, 0 for G=0)

    Returns
        np.array([natm, ngs])
    '''
    Gvnorm=np.linalg.norm(pbc.get_Gv(cell, gs),axis=0)
    #vlocG=4*pi/np.sum(np.conj(Gv)*Gv,axis=0)
    vlocG = get_gth_vlocG(cell, Gvnorm)
    vlocG[:,0] = 0.
    return vlocG

def get_gth_vlocG(cell, G):
    '''
    Local part of the GTH pseudopotential

    G: np.array([ngs]) 

    Returns 
         np.array([natm,ngs])
    '''
    vlocG = np.zeros((cell.natm,len(G))) 
    for ia in range(cell.natm):
        pp = cell._pseudo[ cell.atom_symbol(ia) ] 
        rloc, nexp, cexp = pp[1:3+1]

        G_red = G*rloc
        cfacs = np.array(
                [1*G_red**0, 
                 3 - G_red**2, 
                 15 - 10*G_red**2 + G_red**4, 
                 105 - 105*G_red**2 + 21*G_red**4 - G_red**6])

        with np.errstate(divide='ignore'):
            # Note the signs -- potential here is positive
            vlocG[ia,:] = ( 4*pi * np.exp(-0.5*G_red**2)/G**2
                           - (2*pi)**(3/2.)*rloc**3*np.exp(-0.5*G_red**2)*(
                                np.dot(cexp, cfacs[:nexp])) )
    return vlocG

def gtth_vnonloc_G(Gvecs):
    '''
    Returns
        list of length len(nprojs)
    '''
    G_red = G*rloc

    projs = []
    for l in range(nproj_types):
        rl = r[l]
        npl = nproj[l]
        for i in range(npl):
            for j in range(i,npl):
                hij = hproj[i,j]
                pYs = []
                for Gvec in Gvecs:
                    G, theta, phi = cart2polar(Gvec)
                    pYs.append( proj_il_G(i,j,G,rl)*Ylm(l,m,theta,phi) )
                projs.append(hij, pYs)
        

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

