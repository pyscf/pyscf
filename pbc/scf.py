import itertools
import math
import numpy as np
import numpy.fft
import scipy.linalg
import pyscf.gto.mole
import pyscf.dft.numint
import pyscf.scf
import pyscf.scf.hf
import cell as cl

from pyscf.lib import logger

pi=math.pi
sqrt=math.sqrt
exp=math.exp

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

def get_hcore(mf):
    '''H core. Modeled after get_veff_ in rks.py'''
    hcore=get_nuc(mf.cell, mf.gs) 
    hcore+=get_t(mf.cell, mf.gs)
    return hcore

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

def ewald(cell, gs, ew_eta, ew_cut, verbose=logger.DEBUG):
    '''
    Real and G-space Ewald sum 

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

class RHF(pyscf.scf.hf.SCF):
    '''
    RHF adapted for PBC
    '''
    def __init__(self, cell, gs, ew_eta, ew_cut):
        self.cell=cell
        pyscf.scf.hf.SCF.__init__(self, cell.mol)
        self.grids=UniformGrids(cell, gs)
        self.gs=gs
        self.ew_eta=ew_eta
        self.ew_cut=ew_cut

    def get_hcore(self, cell=None):
        if cell is None: cell=self.cell
        return get_hcore(self)

    def get_ovlp(self, cell=None):
        if cell is None: cell=self.cell
        return get_ovlp(self.cell, self.gs)

    def get_j(self, cell=None, dm=None, hermi=1):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        return get_j(self.cell, dm, self.gs)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
    
    def ewald_nuc(self):
        return ewald(self.cell, self.gs, self.ew_eta, self.ew_cut)
        
class RKS(RHF):
    '''
    RKS adapted for PBC. This is a literal duplication of the
    molecular RKS class with some "mol" variables replaced by cell.
    '''
    def __init__(self, cell, gs, ew_eta, ew_cut):
        RHF.__init__(self, cell, gs, ew_eta, ew_cut)
        self._numint = None

    def dump_flags(self):
        pass

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        return pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = ks.get_hcore()
        return pyscf.dft.rks.energy_elec(self, dm, h1e)
    

def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=40
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    #mol.basis = { 'He': [[0, (1.0, 1.0)]] }
    mol.basis = { 'He': 'cc-pVTZ' }
    mol.build()

    # benchmark first with molecular DFT calc
    m=pyscf.dft.rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print "Molecular DFT energy"
    print (m.scf())

    # this is the PBC DFT calc!!
    cell=cl.Cell()
    cell.mol=mol
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)

    # gs=np.array([40,40,40])
    # ew_eta=0.05
    # ew_cut=(40,40,40)
    # mf=RKS(cell, gs, ew_eta, ew_cut)
    # mf.xc = 'LDA,VWN_RPA'
    # print (mf.scf()) # -2.33905569579385

    gs=np.array([60,60,60])
    ew_eta=0.05
    ew_cut=(40,40,40)
    mf=RKS(cell, gs, ew_eta, ew_cut)
    mf.xc = 'LDA,VWN_RPA'
    print (mf.scf()) # 
