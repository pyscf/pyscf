import pyscf.dft
from pyscf.lib import logger

import math
import numpy as np
import scipy
import scipy.linalg
import scipy.special
import scipy.optimize

pi=math.pi
exp=np.exp
sqrt=np.sqrt
erfc=scipy.special.erfc
cos=math.cos
sin=math.sin

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
    #coulG=np.zeros(Gv.shape[1]) 
    #coulG[1:]=4*pi/np.einsum('ij,ij->j',np.conj(Gv[:,1:]),Gv[:,1:])
    with np.errstate(divide='ignore'):
        coulG=4*pi/np.sum(np.conj(Gv)*Gv,axis=0)
    coulG[0] = 0.

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

def get_aoR(cell, coords, kpt=None, isgga=False, relativity=0, bastart=0,
            bascount=None, non0tab=None, verbose=None):
    '''
    Values of AO crystal orbitals on a set of coords.

    With the optional kpt arguments, returns the pair of 
    sin, cos crystal orbitals at that kpt. 

    (kpt==0 denotes Gamma pt, returns values of only one set)

    Returns
        ndarray [ncoords, nao] where nao=cell.nbas at Gamma point
                               
                                        =2*cell.nbas (any other k point)
    '''  
    if kpt is None:
        kpt=np.zeros([3,1])

    # if nimgs is None:
    nimgs = cell.nimgs

    Ts = [[i,j,k] for i in range(-nimgs[0],nimgs[0]+1)
                  for j in range(-nimgs[1],nimgs[1]+1)
                  for k in range(-nimgs[2],nimgs[2]+1)
                  if i**2+j**2+k**2 <= 1./3*np.dot(nimgs,nimgs)]
    
    nao=cell.nbas


    if isgga:
        aoR=np.zeros([4,coords.shape[0], nao], np.complex128)
    else:
        aoR=np.zeros([coords.shape[0], nao], np.complex128)

    for T in Ts:
        L = np.dot(cell.h, T)

        aoR+=(exp(1j*np.dot(kpt.T,L)) *
              pyscf.dft.numint.eval_ao(cell, coords-L,
                                       isgga, relativity, 
                                       bastart, bascount, 
                                       non0tab, verbose))
    return aoR

def get_rhoR(mol, ao, dm, non0tab=None, 
             isgga=False, verbose=None):
    '''
    See pyscf.dft.numint.eval_rho, generalized for complex aoR.

    Returns 
        *Real* density (and optionally derivatives) on grid
    '''
    import numpy
    from pyscf.dft.numint import _dot_ao_dm, eval_rho, BLKSIZE

    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    # For below code, also see pyscf.dft.numint.eval_rho
    #if ao[0].dtype==numpy.complex128: # complex orbitals
    if True:
        dm_re=numpy.ascontiguousarray(dm.real)
        dm_im=numpy.ascontiguousarray(dm.imag)

        if isgga:
            rho = numpy.empty((4,ngrids))
            ao_re=numpy.ascontiguousarray(ao[0].real)
            ao_im=numpy.ascontiguousarray(ao[0].imag)
            ao_re=numpy.ascontiguousarray(ao[0].real)
            ao_im=numpy.ascontiguousarray(ao[0].imag)

            # DM * ket: e.g. ir denotes dm_im | ao_re >
            c0_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)
            c0_ri = _dot_ao_dm(mol, ao_im, dm_re, nao, ngrids, non0tab)
            c0_ir = _dot_ao_dm(mol, ao_re, dm_im, nao, ngrids, non0tab)
            c0_ii = _dot_ao_dm(mol, ao_im, dm_im, nao, ngrids, non0tab)

            # bra * DM
            rho[0] = (numpy.einsum('pi,pi->p', ao_im, c0_ri) +
                      numpy.einsum('pi,pi->p', ao_re, c0_rr) +
                      numpy.einsum('pi,pi->p', ao_im, c0_ir) -
                      numpy.einsum('pi,pi->p', ao_re, c0_ii))

            for i in range(1, 4):
                ao_re=numpy.ascontiguousarray(ao[i].real)
                ao_im=numpy.ascontiguousarray(ao[i].imag)

                c1_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)
                c1_ri = _dot_ao_dm(mol, ao_im, dm_re, nao, ngrids, non0tab)
                c1_ir = _dot_ao_dm(mol, ao_re, dm_im, nao, ngrids, non0tab)
                c1_ii = _dot_ao_dm(mol, ao_im, dm_im, nao, ngrids, non0tab)

                rho[i] = (numpy.einsum('pi,pi->p', ao_im, c1_ri) +
                          numpy.einsum('pi,pi->p', ao_re, c1_rr) +
                          numpy.einsum('pi,pi->p', ao_im, c1_ir) -
                          numpy.einsum('pi,pi->p', ao_re, c1_ii)) * 2 # *2 for +c.c.       
        else:
            ao_re=numpy.ascontiguousarray(ao.real)
            ao_im=numpy.ascontiguousarray(ao.imag)
            # DM * ket: e.g. ir denotes dm_im | ao_re >
            
            c0_rr = _dot_ao_dm(mol, ao_re, dm_re, nao, ngrids, non0tab)
            c0_ri = _dot_ao_dm(mol, ao_im, dm_re, nao, ngrids, non0tab)
            c0_ir = _dot_ao_dm(mol, ao_re, dm_im, nao, ngrids, non0tab)
            c0_ii = _dot_ao_dm(mol, ao_im, dm_im, nao, ngrids, non0tab)
            # bra * DM
            rho = (numpy.einsum('pi,pi->p', ao_im, c0_ri) +
                   numpy.einsum('pi,pi->p', ao_re, c0_rr) +
                   numpy.einsum('pi,pi->p', ao_im, c0_ir) -
                   numpy.einsum('pi,pi->p', ao_re, c0_ii))
                                    
    else:
        rho=eval_rho(mol, ao, dm, non0tab,
                     isgga, verbose)
    
    return rho


def eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
             isgga=False, verbose=None):
    '''Calculate XC potential matrix.
    
    See also pyscf.dft.numint.eval_mat
    '''
    from pyscf.dft.numint import BLKSIZE, _dot_ao_ao
    import numpy

    if isgga:
        ngrids, nao = ao[0].shape
    else:
        ngrids, nao = ao.shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.int8)

    if ao[0].dtype==numpy.complex128:
        if isgga:
            assert(vsigma is not None and rho.ndim==2)
            #wv = weight * vsigma * 2
            #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
            #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
            wv = numpy.empty_like(rho)
            wv[0]  = weight * vrho * .5
            wv[1:] = rho[1:] * (weight * vsigma * 2)
            aow = numpy.einsum('npi,np->pi', ao, wv)

            ao_re=numpy.ascontiguousarray(ao[0].real)
            ao_im=numpy.ascontiguousarray(ao[0].imag)

            aow_re = numpy.ascontiguousarray(aow.real)
            aow_im = numpy.ascontiguousarray(aow.imag)

        else:
            # *.5 because return mat + mat.T
            #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
            ao_re=numpy.ascontiguousarray(ao.real)
            ao_im=numpy.ascontiguousarray(ao.imag)

            aow_re = ao_re * (.5*weight*vrho).reshape(-1,1)
            aow_im = ao_im * (.5*weight*vrho).reshape(-1,1)
            #mat = pyscf.lib.dot(ao.T, aow)

        mat_re = _dot_ao_ao(mol, ao_re, aow_re, nao, ngrids, non0tab)
        mat_re += _dot_ao_ao(mol, ao_im, aow_im, nao, ngrids, non0tab)
        mat_im = _dot_ao_ao(mol, ao_re, aow_im, nao, ngrids, non0tab)
        mat_im -= _dot_ao_ao(mol, ao_im, aow_re, nao, ngrids, non0tab)

        mat=mat_re+1j*mat_im

        # print "MATRIX", mat.dtype
        return mat + mat.T.conj()
        
    else:
        return pyscf.dft.numint.eval_mat(mol, ao, 
                                         weight, rho, vrho, 
                                         vsigma=None, non0tab=None,
                                         isgga=False, verbose=None)


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

def get_nimgs_cutoff(cell, precision):
    '''
    Returns number of AO lattice images
    to include to achieve a given precision

    Returns
        nimgs: ndarray
    '''
    pass

def ewald_params(cell, gs, precision):
    '''
    Based on largest G vector and desired relative precision,
    choose a reasonable value of ewald eta parameter

    G sum Relative error given by (keeping only exponential factors)

        precision ~ e^{(-Gmax^2)/(4 \eta^2)} 

    This determines alpha. Then, real-space cutoff is 
    determined by (exp. factors only)

        precision ~ erfc(eta*rcut) / rcut ~ e^{(-eta**2 rcut*2)}

    Returns

        ew_eta, ew_cut
    '''
    invhT=scipy.linalg.inv(cell.h.T)

    Gmax=2*pi*np.dot(invhT, gs)
    Gmax=abs(np.min(Gmax))
    log_precision=math.log(precision)
    ew_eta=math.sqrt(-Gmax**2/(4*log_precision))

    rcut=sqrt(-log_precision)/ew_eta
    rlengths=np.sqrt(np.diag(np.dot(cell.h, cell.h.T)))
    #print "rlengths", rcut, rlengths
    ew_cut=np.rint(np.reshape(rcut/rlengths, rlengths.shape[0])).astype(int)

    return ew_eta, ew_cut
    
    

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
    ewxyz=_span3(ewxrange,ewyrange,ewzrange)

    # SLOW=True
    # if SLOW==True:
    #     ewxyz=ewxyz.T
    #     for ic, (ix, iy, iz) in enumerate(ewxyz):
    #         L = np.einsum('ij,j->i', cell.h, ewxyz[ic])

    #         # prime in summation to avoid self-interaction in unit cell
    #         if (ix == 0 and iy == 0 and iz == 0):
    #             print "L is", L
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 #for ja in range(ia):
    #                 for ja in range(cell.natm):
    #                     if ja != ia:
    #                         qj = chargs[ja]
    #                         rj = coords[ja]
    #                         r = np.linalg.norm(ri-rj)
    #                         ewovrl += qi * qj / r * erfc(ew_eta * r)
    #         else:
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 for ja in range(cell.natm):
    #                     qj=chargs[ja]
    #                     rj=coords[ja]
    #                     r=np.linalg.norm(ri-rj+L)
    #                     ewovrl += qi * qj / r * erfc(ew_eta * r)

    # # else:
    nx = len(ewxrange)
    ny = len(ewyrange)
    nz = len(ewzrange)
    Lall = np.einsum('ij,jk->ik', cell.h, ewxyz).reshape(3,nx,ny,nz)
    #exclude the point where Lall == 0
    Lall[:,ew_cut[0],ew_cut[1],ew_cut[2]] = 1e200
    Lall=Lall.reshape(3,nx*ny*nz)
    Lall = Lall.T

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(ia):
            qj = chargs[ja]
            rj = coords[ja]
            r = np.linalg.norm(ri-rj)
            ewovrl += 2 * qi * qj / r * erfc(ew_eta * r)

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
    ZSI=np.einsum("i,ij->j", chargs, SI)

    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #
    # 1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    #
    # where
    #
    #     ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    #
    # See also Eq. (32) of ewald.pdf at 
    #     http://www.fisica.uniud.it/~giannozz/public/ewald.pdf
    coulG=get_coulG(cell, gs)
    absG2=np.einsum('ij,ij->j',np.conj(Gv),Gv)

    ZSIG2=np.abs(ZSI)**2
    expG2=np.exp(-absG2/(4*ew_eta**2))
    JexpG2=coulG*expG2
    ewgI=np.dot(ZSIG2,JexpG2)
    ewg=.5*np.sum(ewgI)
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
        self.stdout = cell.stdout
        self.verbose = cell.verbose

    def setup_grids_(self, cell=None, gs=None):
        if cell==None: cell=self.cell
        if gs==None: gs=self.gs

        self.coords=setup_uniform_grids(self.cell, self.gs)
        self.weights=np.ones(self.coords.shape[0]) 
        self.weights*=1.*cell.vol/self.weights.shape[0]

        return self.coords, self.weights

    def dump_flags(self):
        logger.info(self, 'Uniform grid')

    def kernel(self, cell=None):
        self.dump_flags()
        return self.setup_grids()

class _NumInt(pyscf.dft.numint._NumInt):
    '''
    Generalization for a single k-pt shift, and
    periodic images
    '''
    def __init__(self, kpt=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpt=kpt

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        return get_aoR(mol, coords, self.kpt, isgga, relativity, bastart, bascount,
                       non0tab, verbose)

    def eval_rho(self, mol, ao, dm, non0tab=None, 
             isgga=False, verbose=None):
        return get_rhoR(mol, ao, dm, non0tab=None, 
             isgga=False, verbose=None)

    def eval_rho2(self, mol, ao, dm, non0tab=None, isgga=False,
                  verbose=None):
        raise NotImplementedError

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''
        Use slow function in numint, which only calls eval_rho, eval_mat.
        Faster function uses eval_rho2 which is not yet implemented.
        '''
        return pyscf.dft.numint.nr_rks_vxc(self, mol, grids, x_id, c_id, dms, 
                                           spin=0, relativity=0, hermi=1,
                                           max_memory=2000, verbose=None)

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        raise NotImplementedError

    def eval_mat(self, mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                 isgga=False, verbose=None):
        # use local function for complex eval_mat
        return eval_mat(mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                        isgga=False, verbose=None)


def test_ewald():

    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR
    import cell as cl

    B=BOHR
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None


    Lunit=5
    Ly=Lz=Lunit
    Lx=Lunit

    h=np.diag([Lx,Ly,Lz])
    
    mol.atom.extend([['He', (2*B,0.5*Ly*B,0.5*Lz*B)],
                     ['He', (3*B,0.5*Ly*B,0.5*Lz*B)]])

    # these are some exponents which are 
    # not hard to integrate
    mol.basis = { 'He': [[0, (1.0, 1.0)]] }
    mol.build()

    cell=cl.Cell()
    cell.__dict__=mol.__dict__ # hacky way to make a cell
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = [1,1,1]
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()

    ew_cut=(20,20,20)
    #ew_cut=(2,2,2)

    gs=np.array([20,20,20])
    ew_eta0, ew_cut0=ewald_params(cell, gs, 1.e-3)
    print ew_eta0, ew_cut0

    for ew_eta in [0.05, 0.1, 0.2, 1]:
        print ewald(cell, gs, ew_eta, ew_cut) # -0.468640671931

    for precision in [1.e-3, 1.e-5, 1.e-7, 1.e-9]:
        ew_eta0, ew_cut0=ewald_params(cell, gs, precision)
        print "precision", precision, ew_eta0, ew_cut0
        print ewald(cell, gs, ew_eta0, ew_cut0) 
        # Ewald values
        # 1.e-3: -0.469112631739
        # 1.e-5: -0.468642153932
        # 1.e-7: -0.468640678042
        # 1.e-9: -0.46864067196
