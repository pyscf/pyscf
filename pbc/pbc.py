import itertools
import math
import numpy as np
import numpy.fft
import scipy.linalg
import pyscf.gto.mole
import pyscf.dft.numint

from pyscf.lib import logger

pi=math.pi
sqrt=math.sqrt
exp=math.exp

'''PBC module. Notation follows Marx and Hutter (MH), "Ab Initio Molecular Dynamics"
'''
class Cell:
    def __init__(self, **kwargs):
        self.mol = None
        self.h = None
        self.vol = 0.

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
    gxyz=np.array([gxyz for gxyz in 
                   itertools.product(gxrange, gyrange, gzrange)])

    Gv=2*pi*np.dot(invhT,gxyz.T)
    return Gv

def get_SI(cell, Gv):
    '''
    structure factor (Eq. (3.34), MH)

    Returns 
        np.array([natm, ngs], np.complex128)
    '''
    ngs=Gv.shape[1]
    mol=cell.mol
    SI=np.empty([mol.natm, ngs], np.complex128)
    mol=cell.mol

    for ia in range(mol.natm):
        SI[ia,:]=np.exp(-1j*np.dot(Gv.T, mol.atom_coord(ia)))

    return SI

def ewald_rs(cell, gs, ew_eta, ew_cut, verbose=logger.DEBUG):
    '''
    Real-space Ewald sum 

    Formulation of Martin, App. F2.

    Returns
        float
    '''
    mol=cell.mol

    log=logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)

    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange=range(-ew_cut[0],ew_cut[0]+1)
    ewyrange=range(-ew_cut[1],ew_cut[1]+1)
    ewzrange=range(-ew_cut[2],ew_cut[2]+1)

    ewxyz=np.array([xyz for xyz in itertools.product(ewxrange,ewyrange,ewzrange)])

    for (ix, iy, iz) in ewxyz:
        L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]

        # prime in summation to avoid self-interaction in unit cell
        for ia in range(mol.natm):
            qi = chargs[ia]
            ri = coords[ia]

            if (ix == 0 and iy == 0 and iz == 0):
                for ja in range(ia):
                    qj = chargs[ja]
                    rj = coords[ja]
                    r = np.linalg.norm(ri-rj)
                    ewovrl += qi * qj / r * math.erfc(ew_eta * r)
            else:
                for ja in range(mol.natm):
                    qj=chargs[ja]
                    rj=coords[ja]

                    r=np.linalg.norm(ri-rj+L)
                    ewovrl += qi * qj / r * math.erfc(ew_eta * r)
        
    ewovrl *= 0.5

    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / sqrt(pi)
    ewself += -1./2. * np.sum(chargs)**2 * pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    ewg=0.

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

    log.debug('ewald components= %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg

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
    return np.array(list(np.ndindex(tuple(ngs)))).T

def setup_uniform_grids(cell, gs):
    '''
    Real-space AO uniform grid, following Eq. (3.19) (MH)
    '''
    mol=cell.mol
    ngs=2*gs+1
    qv=_gen_qv(ngs)
    invN=np.diag(1./np.array(ngs))
    R=np.dot(np.dot(cell.h, invN), qv)
    coords=R.T.copy() # make C-contiguous with copy() for pyscf
    return coords

def get_aoR(cell, coords):
    aoR=pyscf.dft.numint.eval_ao(cell.mol, coords)
    return aoR

def get_rhoR(cell, aoR, dm):
    rhoR=pyscf.dft.numint.eval_rho(cell.mol, aoR, dm)
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

def get_nuc(cell, gs):
    '''
    Bare nuc-el AO matrix (G=0 component removed)
    
    Returns
        v_nuc (nao x nao) matrix
    '''
    mol=cell.mol

    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    nuc_coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    coords=setup_uniform_grids(cell,gs)
    aoR=get_aoR(cell, coords)

    vneR=np.zeros(aoR.shape[0])

    for a in range(mol.natm):
        qa = chargs[a]
        ra = nuc_coords[a]
        riav = [np.linalg.norm(ra-ri) for ri in coords]
        #vneR-=np.array([qa/ria*math.erf(ria/ewrc)
        #                for ria in riav]) # Eq. (3.40) MH
        vneR-=np.array([qa/ria for ria in riav])

    # Set G=0 component to 0.
    vneG=fft(vneR, gs)
    vneG[0]=0.
    vneR=ifft(vneG,gs)

    nao=aoR.shape[1]
    vne=np.zeros([nao, nao])
    for i in range(nao):
        for j in range(nao):
            vne[i,j]=np.vdot(aoR[:,i],vneR*aoR[:,j])

    ngs=aoR.shape[0]
    vne *= (cell.vol/ngs)
    return vne

def get_t(cell, gs):
    '''
    Kinetic energy AO matrix
    '''
    Gv=get_Gv(cell, gs)
    G2=np.array([np.inner(Gv[:,i], Gv[:,i]) for i in xrange(Gv.shape[1])])

    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, coords)
    aoG=np.empty(aoR.shape, np.complex128)
    TaoG=np.empty(aoR.shape, np.complex128)

    nao=aoR.shape[1]
    for i in range(nao):
        aoG[:,i]=fft(aoR[:,i], gs)
        TaoG[:,i]=0.5*G2*aoG[:,i]
                
    t=np.empty([nao,nao])
    for i in range(nao):
        for j in range(nao):
            t[i,j]=np.vdot(aoG[:,i],TaoG[:,j])

    ngs=aoR.shape[0]
    t *= (cell.vol/ngs**2)

    return t

def get_ovlp(cell, gs):
    '''
    Overlap AO matrix
    '''
    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, coords)
    nao=aoR.shape[1]

    s=np.empty([nao,nao])
    for i in range(nao):
        for j in range(nao):
            s[i,j]=np.vdot(aoR[:,i],aoR[:,j])

    ngs=aoR.shape[0]
    s *= cell.vol/ngs
    return s
    
def get_coulG(cell, gs):
    '''
    Coulomb kernel in G space (4*pi/G^2 for G!=0, 0 for G=0)
    '''
    Gv=get_Gv(cell, gs)
    coulG=np.zeros(Gv.shape[1]) 
    coulG[1:]=4*pi/np.einsum('ij,ij->j',np.conj(Gv[:,1:]),Gv[:,1:])
    return coulG

def get_j(cell, dm, gs):
    '''
    Coulomb AO matrix 
    '''
    coulG=get_coulG(cell, gs)

    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, coords)

    rhoR=get_rhoR(cell, aoR, dm)
    rhoG=fft(rhoR, gs)

    vG=coulG*rhoG
    vR=ifft(vG, gs)

    nao=aoR.shape[1]
    ngs=aoR.shape[0]
    vj=np.zeros([nao,nao])
    for i in range(nao):
        for j in range(nao):
            vj[i,j]=cell.vol/ngs*np.dot(aoR[:,i],vR*aoR[:,j])
           
    return vj

def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=40
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    #mol.basis = { 'He': 'STO-3G'}
    mol.basis = { 'He': [[0,
                          (1.0, 1.0)]] }
#        (0.2089, 0.307737),]] }
    
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=Cell()
    cell.mol=mol
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)

    #gs=np.array([10,10,10]) # number of G-points in grid. Real-space dim=2*gs+1
    gs=np.array([60,60,60]) # number of G-points in grid. Real-space dim=2*gs+1
    Gv=get_Gv(cell, gs)

    dm=m.make_rdm1()

    print "Kinetic"
    tao=get_t(cell, gs)
    tao2 = mol.intor_symmetric('cint1e_kin_sph') 

    print "Kinetic energies"
    print np.dot(np.ravel(tao), np.ravel(dm))
    print np.dot(np.ravel(tao2), np.ravel(dm))
    
    print "Overlap"
    sao=get_ovlp(cell,gs)
    print np.dot(np.ravel(sao), np.ravel(dm))
    print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm))

    print "Coulomb (G!=0)"
    jao=get_j(cell,dm,gs)
    print np.dot(np.ravel(dm),np.ravel(jao))
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))

    print "Nuc-el (G!=0)"
    neao=get_nuc(cell,gs)
    vne=mol.intor_symmetric('cint1e_nuc_sph') 
    print np.dot(np.ravel(dm), np.ravel(neao))
    print np.dot(np.ravel(dm), np.ravel(vne))

    print "Normalization"
    coords=setup_uniform_grids(cell, gs)
    aoR=get_aoR(cell, gs)
    rhoR=get_rhoR(cell, aoR, dm)
    print cell.vol/len(rhoR)*np.sum(rhoR) # should be 2.0
    
    print "(Hartree + vne) * DM"
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    print np.einsum("ij,ij",dm,neao+jao)

    ewcut=(40,40,40)
    ew_eta=0.05
    for ew_eta in [0.1, 0.5, 1.]:
        ew=ewald_rs(cell, gs, ew_eta, ewcut)
        print "Ewald (eta, energy)", ew_eta, ew # should be same for all eta

    print "Ewald divergent terms summation", ew

    print "Total coulomb (analytic)", .5*np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    print "Total coulomb (fft coul + ewald)", np.einsum("ij,ij",dm,neao+.5*jao)+ew
