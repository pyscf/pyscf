import pyscf.dft
from pyscf.lib import logger

import math
import numpy as np
import scipy
import scipy.linalg
import scipy.special

pi=math.pi
exp=np.exp
sqrt=np.sqrt
erfc=scipy.special.erfc

def get_Gv(cell, gs):
    '''
    cube of G vectors (Eq. (3.8), MH),

    Indices(along each direction go as
    [0...gs, -gs...-1]
    to follow FFT convention

    Returns 
        np.array([3, ngs], np.float64)
    '''
    invhT=scipy.linalg.inv(cell.h.T)

    # maybe there's a better numpy way?
    gxrange=range(gs[0]+1)+range(-gs[0],0)
    gyrange=range(gs[1]+1)+range(-gs[1],0)
    gzrange=range(gs[2]+1)+range(-gs[2],0)
    #:gxyz=np.array([gxyz for gxyz in 
    #:               itertools.product(gxrange, gyrange, gzrange)])
    gxyz = _span3(gxrange, gyrange, gzrange)

    Gv=2*pi*np.dot(invhT,gxyz)
    return Gv

def get_SI(cell, Gv):
    '''
    structure factor (Eq. (3.34), MH)

    Returns 
        np.array([natm, ngs], np.complex128)
    '''
    ngs=Gv.shape[1]
    SI=np.empty([cell.natm, ngs], np.complex128)

    for ia in range(cell.natm):
        SI[ia,:]=np.exp(-1j*np.dot(Gv.T, cell.atom_coord(ia)))

    return SI

def get_coulG(cell, gs):
    '''
    Coulomb kernel in G space (4*pi/G^2 for G!=0, 0 for G=0)
    '''
    Gv=get_Gv(cell, gs)
    coulG=np.zeros(Gv.shape[1]) 
    coulG[1:]=4*pi/np.einsum('ij,ij->j',np.conj(Gv[:,1:]),Gv[:,1:])
    return coulG

def _gen_qv(ngs):
    '''
    integer cube of indices, 0...ngs-1 along each direction
    ngs: [ngsx, ngsy, ngsz]

    Returns 
         3 * (ngsx*ngsy*ngsz) matrix
         [0, 0, ... ngsx-1]
         [0, 0, ... ngsy-1]
         [0, 1, ... ngsz-1]
    '''
    #return np.array(list(np.ndindex(tuple(ngs)))).T
    return _span3(np.arange(ngs[0]), np.arange(ngs[1]), np.arange(ngs[2]))

def _span3(*xs):
    c = np.empty([3]+[len(x) for x in xs])
    c[0,:,:,:] = np.asarray(xs[0]).reshape(-1,1,1)
    c[1,:,:,:] = np.asarray(xs[1]).reshape(1,-1,1)
    c[2,:,:,:] = np.asarray(xs[2]).reshape(1,1,-1)
    return c.reshape(3,-1)

def setup_uniform_grids(cell, gs):
    '''
    Real-space AO uniform grid, following Eq. (3.19) (MH)
    '''
    ngs=2*gs+1
    qv=_gen_qv(ngs)
    invN=np.diag(1./np.array(ngs))
    R=np.dot(np.dot(cell.h, invN), qv)
    coords=R.T.copy() # make C-contiguous with copy() for pyscf
    return coords

def get_aoR(cell, coords):
    return _eval_ao(cell, coords)

def _eval_ao(cell, coords):
    nimgs = cell.nimgs
    aoR=pyscf.dft.numint.eval_ao(cell, coords)
    Ts = [[i,j,k] for i in range(-nimgs,nimgs+1)
                  for j in range(-nimgs,nimgs+1)
                  for k in range(-nimgs,nimgs+1) 
                  if i**2+j**2+k**2 <= nimgs**2 and i**2+j**2+k**2 != 0]

    for T in Ts:
        aoR+=pyscf.dft.numint.eval_ao(cell, coords+np.dot(cell.h,T))
    return aoR

def get_rhoR(cell, aoR, dm):
    rhoR=pyscf.dft.numint.eval_rho(cell, aoR, dm)
    return rhoR

def fft(f, gs):
    '''
    3D FFT R to G space.
    
    f is evaluated on a real-space grid, assumed flattened
    to a 1d array corresponding to the index order of gen_q, where q=(u, v, w)
    u = range(0, ngs[0]), v = range(0, ngs[1]), w = range(0, ngs[2]).
    
    (re: Eq. (3.25), we assume Ns := ngs = 2*gs+1)

    After FT, (u, v, w) -> (j, k, l).
    (jkl) is in the index order of Gv

    Returns
        FFT 1D array in same index order as Gv (natural order of numpy.fft)

    Note: FFT norm factor is 1., as in MH and in numpy.fft
    '''
    ngs=2*gs+1
    f3d=np.reshape(f, ngs)
    g3d=np.fft.fftn(f3d)
    return np.ravel(g3d)

def ifft(g, gs):
    '''
    Note: invFFT norm factor is 1./N - same as numpy but **different** 
    from MH (they use 1.)
    '''
    ngs=2*gs+1
    g3d=np.reshape(g, ngs)
    f3d=np.fft.ifftn(g3d)
    return np.ravel(f3d)

def ewald(cell, gs, ew_eta, ew_cut, verbose=logger.DEBUG):
    '''
    Real and G-space Ewald sum 

    Formulation of Martin, App. F2.

    Returns
        float
    '''
    log=logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    chargs = [cell.atom_charge(i) for i in range(len(cell._atm))]
    coords = [cell.atom_coord(i) for i in range(len(cell._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange=range(-ew_cut[0],ew_cut[0]+1)
    ewyrange=range(-ew_cut[1],ew_cut[1]+1)
    ewzrange=range(-ew_cut[2],ew_cut[2]+1)

    #:ewxyz=np.array([xyz for xyz in itertools.product(ewxrange,ewyrange,ewzrange)])
    ewxyz=_span3(ewxrange,ewyrange,ewzrange)

    #:for ic, (ix, iy, iz) in enumerate(ewxyz):
    #:    #:L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]
    #:    L = np.einsum('ij,j->i', cell.h, ewxyz[ic])

    #:    # prime in summation to avoid self-interaction in unit cell
    #:    if (ix == 0 and iy == 0 and iz == 0):
    #:        for ia in range(cell.natm):
    #:            qi = chargs[ia]
    #:            ri = coords[ia]
    #:            for ja in range(ia):
    #:                qj = chargs[ja]
    #:                rj = coords[ja]
    #:                r = np.linalg.norm(ri-rj)
    #:                ewovrl += qi * qj / r * erfc(ew_eta * r)
    #:    else:
    #:        #:for ia in range(cell.natm):
    #:        #:    qi = chargs[ia]
    #:        #:    ri = coords[ia]
    #:        #:    for ja in range(cell.natm):
    #:        #:        qj=chargs[ja]
    #:        #:        rj=coords[ja]
    #:        #:        r=np.linalg.norm(ri-rj+L)
    #:        #:        ewovrl += qi * qj / r * erfc(ew_eta * r)
    #:        r1 = rij + L
    #:        r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
    #:        ewovrl += (qij/r * erfc(ew_eta * r)).sum()
    nx = len(ewxrange)
    ny = len(ewyrange)
    nz = len(ewzrange)
    Lall = np.einsum('ij,jk->ki', cell.h, ewxyz).reshape(3,nx,ny,nz)
    #exclude the point where Lall == 0
    Lall[:,ew_cut[0],ew_cut[1],ew_cut[2]] = 1e200
    Lall = Lall.reshape(-1,3)
    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(ia):
            qj = chargs[ja]
            rj = coords[ja]
            r = np.linalg.norm(ri-rj)
            ewovrl += qi * qj / r * erfc(ew_eta * r)
    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(cell.natm):
            qj = chargs[ja]
            rj = coords[ja]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            ewovrl += (qi * qj / r * erfc(ew_eta * r)).sum()

    ewovrl *= 0.5

    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / sqrt(pi)
    ewself += -1./2. * np.sum(chargs)**2 * pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    Gv=get_Gv(cell, gs)
    SI=get_SI(cell, Gv)

    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #
    # 2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |S_I(G)|^2 \exp[-|G|^2/4\eta^2]
    coulG=get_coulG(cell, gs)
    absG2=np.einsum('ij,ij->j',np.conj(Gv),Gv)
    SIG2=np.abs(SI)**2
    expG2=np.exp(-absG2/(4*ew_eta**2))
    JexpG2=2*coulG*expG2
    ewgI=np.dot(SIG2,JexpG2)
    ewg=np.sum(ewgI)
    ewg/=cell.vol

    log.debug('Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

def get_ao_pairs_G(cell, gs):
    '''
    forward and inverse FFT of AO pairs -> (ij|G), (G|ij)

    Returns
        [ndarray, ndarray] : ndarray [ngs, nao*(nao+1)/2]
    '''
    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, coords) # shape(coords, nao)
    nao=aoR.shape[1]
    npair=nao*(nao+1)/2
    ao_pairs_G=np.zeros([coords.shape[0], npair], np.complex128)
    ao_pairs_invG=np.zeros([coords.shape[0], npair], np.complex128)
    ij=0
    for i in range(nao):
        for j in range(i+1):
            ao_ij_R=np.einsum('r,r->r', aoR[:,i], aoR[:,j])
            ao_pairs_G[:, ij]=fft(ao_ij_R, gs)         
            ao_pairs_invG[:, ij]=ifft(ao_ij_R, gs)
            ij+=1
    return ao_pairs_G, ao_pairs_invG
    
def get_mo_pairs_G(cell, gs, mo_coeffs):
    '''
    forward and inverse FFT of MO pairs -> (ij|G), (G|ij)
    
    Not correctly implemented: simplifications for real (ij), or for complex MOs!
    Returns

        [ndarray, ndarray] : ndarray [ngs, nmo[0]*nmo[1]]
    '''
    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, coords) # shape(coords, nao)
    nmoi=mo_coeffs[0].shape[1]
    nmoj=mo_coeffs[1].shape[1]

    # this also doesn't check for the (common) case
    # where mo_coeffs[0] == mo_coeffs[1]
    moiR=np.einsum('ri,ia->ra',aoR, mo_coeffs[0])
    mojR=np.einsum('ri,ia->ra',aoR, mo_coeffs[1])

    # this would need a conj on moiR if we have complex fns
    mo_pairs_R=np.einsum('ri,rj->rij',moiR,mojR)
    mo_pairs_G=np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)
    mo_pairs_invG=np.zeros([coords.shape[0],nmoi*nmoj], np.complex128)

    for i in xrange(nmoi):
        for j in xrange(nmoj):
            mo_pairs_G[:,i*nmoj+j]=fft(mo_pairs_R[:,i,j], gs)
            mo_pairs_invG[:,i*nmoj+j]=ifft(mo_pairs_R[:,i,j], gs)
    return mo_pairs_G, mo_pairs_invG

def assemble_eri(cell, gs, orb_pair_G1, orb_pair_invG2, verbose=logger.DEBUG):
    '''
    Assemble 4-index ERI

    \sum_G (ij|G)(G|kl) 

    Returns
        [nmo1*nmo2, nmo3*nmo4] ndarray
    '''
    log=logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    log.debug('Performing periodic ERI assembly of (%i, %i) ij pairs', 
              orb_pair_G1.shape[1], orb_pair_invG2.shape[1])
    coulG=get_coulG(cell, gs)
    ngs=orb_pair_invG2.shape[0]
    Jorb_pair_invG2=np.einsum('g,gn->gn',coulG,orb_pair_invG2)*(cell.vol/ngs)
    eri=np.einsum('gm,gn->mn',orb_pair_G1, Jorb_pair_invG2)
    return eri

def get_ao_eri(cell, gs):
    '''
    Convenience function to return AO integrals
    '''

    ao_pairs_G, ao_pairs_invG=get_ao_pairs_G(cell, gs)
    return assemble_eri(cell, gs, ao_pairs_G, ao_pairs_invG)
        
def get_mo_eri(cell, gs, mo_coeffs12, mo_coeffs34):
    '''
    Convenience function to return MO integrals
    '''
    # don't really need FFT and iFFT for both sets
    mo_pairs12_G, mo_pairs12_invG=get_mo_pairs_G(cell, gs, mo_coeffs12)
    mo_pairs34_G, mo_pairs34_invG=get_mo_pairs_G(cell, gs, mo_coeffs34)
    return assemble_eri(cell, gs, mo_pairs12_G, mo_pairs34_invG)

class UniformGrids(object):
    '''
    Uniform Grid
    '''
    def __init__(self, cell, gs):
        self.cell = cell
        self.gs = gs
        self.coords = None
        self.weights = None
        
    def setup_grids_(self, cell=None, gs=None):
        if cell==None: cell=self.cell
        if gs==None: gs=self.gs

        self.coords=setup_uniform_grids(self.cell, self.gs)
        self.weights=np.ones(self.coords.shape[0]) 
        self.weights*=1.*cell.vol/self.weights.shape[0]

        return self.coords, self.weights

    def dump_flags(self):
        logger.info(self, 'uniform grid')

    def kernel(self, cell=None):
        self.dump_flags()
        return self.setup_grids()
