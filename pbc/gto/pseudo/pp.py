import numpy as np
import scipy.linalg
import scipy.special
from pyscf import lib

'''PP module.
    
For GTH/HGH PPs, see: 
    Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
    Hartwigsen, Goedecker, and Hutter, PRB 58, 3641 (1998)
'''

def get_alphas(cell):
    '''alpha parameters from the non-divergent Hartree+Vloc G=0 term.

    See ewald.pdf

    Returns:
        alphas : (natm,) ndarray
    '''
    return get_alphas_gth(cell)

def get_alphas_gth(cell):
    '''alpha parameters for the local GTH pseudopotential.'''

    alphas = np.zeros(cell.natm) 
    for ia in range(cell.natm):
        Zia = cell.atom_charge(ia)
        pp = cell._pseudo[ cell.atom_symbol(ia) ] 
        rloc, nexp, cexp = pp[1:3+1]

        cfacs = [1., 3., 15., 105.]
        alphas[ia] = ( 2*np.pi*Zia*rloc**2
                     + (2*np.pi)**(3/2.)*rloc**3*np.dot(cexp,cfacs[:nexp]) )
    return alphas

def get_vlocG(cell):
    '''Local PP kernel in G space: Vloc(G) for G!=0, 0 for G=0.

    Returns:
        (natm, ngs) ndarray
    '''
    Gvnorm = np.linalg.norm(cell.Gv,axis=0)
    vlocG = get_gth_vlocG(cell, Gvnorm)
    vlocG[:,0] = 0.
    return vlocG

def get_gth_vlocG(cell, G):
    '''Local part of the GTH pseudopotential.

    See MH (4.79).

    Args:
        G : (ngs,) ndarray

    Returns: 
         (natm, ngs) ndarray
    '''
    vlocG = np.zeros((cell.natm,len(G))) 
    for ia in range(cell.natm):
        Zia = cell.atom_charge(ia)
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
            vlocG[ia,:] = ( 4*np.pi * Zia * np.exp(-0.5*G_red**2)/G**2
                           - (2*np.pi)**(3/2.)*rloc**3*np.exp(-0.5*G_red**2)*(
                                np.dot(cexp, cfacs[:nexp])) )
    return vlocG

def get_projG(cell):
    '''PP weight and projector for the nonlocal PP in G space.

    Returns:
        hs : list( list( np.array( , ) ) )
         - hs[atm][l][i,j]
        projs : list( list( list( list( np.array(ngs) ) ) ) )
         - projs[atm][l][m][i][ngs]
    '''
    return get_gth_projG(cell, cell.Gv) 

def get_gth_projG(cell, Gvs):
    '''
    MH Eq.(4.80)
    '''
    Gs,thetas,phis = cart2polar(Gvs)
        
    hs = []
    projs = []
    for ia in range(cell.natm):
        pp = cell._pseudo[ cell.atom_symbol(ia) ] 
        nproj_types = pp[4]
        h_ia = []
        proj_ia = []
        for l,proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            h_ia.append( (-1)**l * np.array(hl) )
            proj_ia_l = []
            for m in range(-l,l+1):
                projG_ang = Ylm(l,m,thetas,phis).conj()
                proj_ia_lm = []
                for i in range(nl):
                    projG_radial = projG_li(Gs,l,i,rl)
                    proj_ia_lm.append( projG_radial*projG_ang )

                proj_ia_l.append(proj_ia_lm)
            proj_ia.append(proj_ia_l)
        hs.append(h_ia)
        projs.append(proj_ia)

    return hs, projs

def projG_li(G, l, i, rl): 
    G = np.array(G)
    G_red = G*rl

    # MH Eq. (4.81)
    return _qli(G_red,l,i)*np.pi**(5/4.)*G**l*np.sqrt(rl**(2*l+3))/np.exp(0.5*G_red**2)

def _qli(x,l,i):
    # MH Eqs. (4.82)-(4.93) :: beware typos!
    sqrt = np.sqrt
    if l==0 and i==0:
        return 4*sqrt(2.)
    elif l==0 and i==1:
        return 8*sqrt(2/15.)*(3-x**2) # MH & GTH (right)
        #return sqrt(8*2/15.)*(3-x**2) # HGH (wrong)
    elif l==0 and i==2:
        #return 16/3.*sqrt(2/105.)*(15-20*x**2+4*x**4) # MH (wrong)
        return 16/3.*sqrt(2/105.)*(15-10*x**2+x**4) # HGH (right)
    elif l==1 and i==0:
        return 8*sqrt(1/3.)
    elif l==1 and i==1:
        return 16*sqrt(1/105.)*(5-x**2)
    elif l==1 and i==2:
        #return 32/3.*sqrt(1/1155.)*(35-28*x**2+4*x**4) # MH (wrong)
        return 32/3.*sqrt(1/1155.)*(35-14*x**2+x**4) # HGH (right)
    elif l==2 and i==0:
        return 8*sqrt(2/15.)
    elif l==2 and i==1:
        return 16/3.*sqrt(2/105.)*(7-x**2)
    elif l==2 and i==2:
        #return 32/3.*sqrt(2/15015.)*(63-36*x**2+4*x**4) # MH (wrong I think)
        return 32/3.*sqrt(2/15015.)*(63-18*x**2+x**4) # TCB
    else:
        print "*** WARNING *** l =", l, ", i =", i, "not yet implemented for NL PP!"
        return 0.

def Ylm(l,m,theta,phi):
    '''
    Spherical harmonics; returns a complex number
    '''
    return scipy.special.sph_harm(m,l,theta,phi)

def cart2polar(rvec):
    # The columns of rvec are the 3-component vectors
    # i.e. rvec is 3 x N
    x,y,z = rvec
    r = lib.norm(rvec,axis=0)
    theta = np.arctan2(z,np.sqrt(x**2+y**2))
    phi = np.arctan2(y,x)
    return r, theta, phi

