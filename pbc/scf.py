import sys
import itertools
import math
import numpy as np
import numpy.fft
import scipy.linalg
import scipy.special
import pyscf.gto.mole
import pyscf.dft.numint
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import cell as cl
import pbc
import pp

from pyscf.lib import logger

pi=math.pi
sqrt=math.sqrt
exp=math.exp
erfc = scipy.special.erfc

def get_hcore(mf, cell):
    '''H core. Modeled after get_veff_ in rks.py'''
    hcore=get_nuc(cell, mf.gs) 
    hcore+=get_t(cell, mf.gs)
    return hcore

def get_nuc(cell, gs):
    '''
    Bare periodic nuc-el AO matrix (G=0 component removed).
    c.f. Martin Eq. (12.16)-(12.21)
    
    Returns
        v_nuc (nao x nao) matrix
    '''
    chargs=[cell.atom_charge(i) for i in range(len(cell._atm))]

    Gv=pbc.get_Gv(cell, gs)
    SI=pbc.get_SI(cell, Gv)
    coulG=pbc.get_coulG(cell, gs)
    
    coords=pbc.setup_uniform_grids(cell,gs)

    #vneG=np.zeros(coords.shape[0], np.complex128)
    #for ia, qa in enumerate(chargs):
    #    vneG+=-chargs[ia] * SI[ia,:] * coulG
    vneG = -np.dot(chargs,SI) * coulG

    vneR=pbc.ifft(vneG, gs)
    aoR=pbc.get_aoR(cell, coords)
        
    nao=aoR.shape[1]
    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR).real
    return vne

def get_pp(cell, gs):
    '''
    Nuc-el pseudopotential AO matrix
    
    Only local part right now, completely untested.
    '''
    chargs=[cell.atom_charge(i) for i in range(len(cell._atm))]

    Gv=pbc.get_Gv(cell, gs)
    SI=pbc.get_SI(cell, Gv)
    vlocG=pp.get_vlocG(cell, gs)
    
    coords=pbc.setup_uniform_grids(cell,gs)

    #vpplocG=np.zeros(coords.shape[0], np.complex128)
    #for ia, qa in enumerate(chargs):
    #    vpplocG+=-chargs[ia] * SI[ia,:] * vlocG
    qvlocG = chargs*vlocG
    vpplocG = -np.sum(SI * qvlocG, axis=0)

    vpplocR=pbc.ifft(vpplocG, gs)
    aoR=pbc.get_aoR(cell, coords)
        
    nao=aoR.shape[1]
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR).real
    return vpploc

def get_t(cell, gs, kpt=None):
    '''
    Kinetic energy AO matrix
    '''
    if kpt is None:
        kpt=np.zeros([3,1])
    
    Gv=pbc.get_Gv(cell, gs)

    print type(kpt)
    print type(Gv)
    print kpt

    Gv+=kpt
    #:G2=np.array([np.inner(Gv[:,i], Gv[:,i]) for i in xrange(Gv.shape[1])])
    G2=np.einsum('ji,ji->i', Gv, Gv)

    coords=pbc.setup_uniform_grids(cell, gs)
    aoR=pbc.get_aoR(cell, coords)
    aoG=np.empty(aoR.shape, np.complex128)
    TaoG=np.empty(aoR.shape, np.complex128)

    nao=aoR.shape[1]
    for i in range(nao):
        aoG[:,i]=pbc.fft(aoR[:,i], gs)
        TaoG[:,i]=0.5*G2*aoG[:,i]
                
    #:t=np.empty([nao,nao])
    #:for i in range(nao):
    #:    for j in range(nao):
    #:        t[i,j]=np.vdot(aoG[:,i],TaoG[:,j])
    t = np.dot(aoG.T.conj(), TaoG).real

    ngs=aoR.shape[0]
    t *= (cell.vol/ngs**2)
    return t

def get_ovlp(cell, gs):
    '''
    Overlap AO matrix
    '''
    coords=pbc.setup_uniform_grids(cell, gs)
    aoR=pbc.get_aoR(cell, coords)
    nao=aoR.shape[1]
    ngs=aoR.shape[0]

    s = np.dot(aoR.T.conj(), aoR).real
    s *= cell.vol/ngs

    #aoG=np.empty(aoR.shape, np.complex128)
    #
    #for i in range(nao):
    #    aoG[:,i]=pbc.fft(aoR[:,i], gs)
    #            
    #s = np.dot(aoG.T.conj(), aoG).real
    #s *= (cell.vol/ngs**2)

    return s
    
def get_j(cell, dm, gs):
    '''
    Coulomb AO matrix 
    '''
    coulG=pbc.get_coulG(cell, gs)

    coords=pbc.setup_uniform_grids(cell, gs)
    aoR=pbc.get_aoR(cell, coords)

    rhoR=pbc.get_rhoR(cell, aoR, dm)
    rhoG=pbc.fft(rhoR, gs)

    vG=coulG*rhoG
    vR=pbc.ifft(vG, gs)

    nao=aoR.shape[1]
    ngs=aoR.shape[0]
    #:vj=np.zeros([nao,nao])
    #:for i in range(nao):
    #:    for j in range(nao):
    #:        vj[i,j]=cell.vol/ngs*np.dot(aoR[:,i],vR*aoR[:,j])
    vj = cell.vol/ngs * np.dot(aoR.T.conj(), vR.reshape(-1,1)*aoR).real
           
    return vj


class RHF(pyscf.scf.hf.RHF):
    '''
    RHF adapted for PBC

    TODO: maybe should create SCF class derived from pyscf.scf.hf.SCF, then
          derive from that
    '''
    def __init__(self, cell, gs, ew_eta, ew_cut):
        self.cell=cell
        pyscf.scf.hf.RHF.__init__(self, cell)
        self.grids=pbc.UniformGrids(cell, gs)
        self.gs=gs
        self.ew_eta=ew_eta
        self.ew_cut=ew_cut
        self.mol_ex=False

    def get_hcore(self, cell=None):
        if cell is None: cell=self.cell
        return get_hcore(self, cell)

    def get_ovlp(self, cell=None):
        if cell is None: cell=self.cell
        return get_ovlp(cell, self.gs)

    def get_j(self, cell=None, dm=None, hermi=1):
        if cell is None: cell=self.cell
        if dm is None: dm = self.make_rdm1()
        return get_j(cell, dm, self.gs)

    def get_jk_(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG):
        '''
        *Incore* version of Coulomb and exchange build only.
        
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals

        c.f. scf.hf.RHF.get_jk_
        '''
        if cell is None:
            cell=self.cell
        
        log=logger.Logger
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(cell.stdout, verbose)

        log.debug('JK PBC build: incore only with PBC integrals')

        if self._eri is None:
            self._eri=np.real(pbc.get_ao_eri(cell, self.gs))

        vj, vk=pyscf.scf.hf.RHF.get_jk_(self, cell, dm, hermi) 
        
        if self.mol_ex: # use molecular exchange, but periodic J
            log.debug('K PBC build: using molecular integrals')
            mol_eri=pyscf.scf._vhf.int2e_sph(cell._atm, cell._bas, cell._env)
            mol_vj, vk=pyscf.scf.hf.dot_eri_dm(mol_eri, dm, hermi)

        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
    
    def ewald_nuc(self):
        return pbc.ewald(self.cell, self.gs, self.ew_eta, self.ew_cut)
        
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
        if h1e is None: h1e = get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)
    
class KRKS(RKS):
    '''
    Periodic RKS with K-points
    '''
    pass

def test_pp():
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = { 'He': 'STO-3G'}
    
    mol.build()
    
    cell=cl.Cell()
    cell.__dict__=mol.__dict__

    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 0
    cell.pseudo=None
    cell.output=None
    cell.verbose=7

    # Add a pseudopotential:
    cell.pseudo = 'gth-blyp'

    cell.build()

    print "Internal PP format"
    print cell._pseudo

def test_components():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = { 'He': 'STO-3G'}
    
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    #m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=cl.Cell()
    cell.__dict__=mol.__dict__

    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 1
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()
    
    #gs=np.array([10,10,10]) # number of G-points in grid. Real-space dim=2*gs+1
    gs=np.array([100,100,100]) # number of G-points in grid. Real-space dim=2*gs+1
    Gv=pbc.get_Gv(cell, gs)

    dm=m.make_rdm1()

    print "Kinetic"
    tao=get_t(cell, gs) 
    tao2 = mol.intor_symmetric('cint1e_kin_sph') 

    # These should match reasonably well (roughly with accuracy of normalization)
    print "Kinetic energies" 
    print np.dot(np.ravel(tao), np.ravel(dm))  # 2.82793077196
    print np.dot(np.ravel(tao2), np.ravel(dm)) # 2.82352636524
    
    print "Overlap"
    sao=get_ovlp(cell,gs)
    print np.dot(np.ravel(sao), np.ravel(dm)) # 1.99981725342
    print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm)) # 2.0

    # The next two entries should *not* match, since G=0 component is removed
    print "Coulomb (G!=0)"
    jao=get_j(cell,dm,gs)
    print np.dot(np.ravel(dm),np.ravel(jao))  # 4.03425518427
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm))) # 4.22285177049

    # The next two entries should *not* match, since G=0 component is removed
    print "Nuc-el (G!=0)"
    neao=get_nuc(cell,gs)
    vne=mol.intor_symmetric('cint1e_nuc_sph') 
    print np.dot(np.ravel(dm), np.ravel(neao)) # -6.50203360062
    print np.dot(np.ravel(dm), np.ravel(vne))  # -6.68702326551

    print "Normalization" 
    coords=pbc.setup_uniform_grids(cell, gs)
    aoR=pbc.get_aoR(cell, coords)
    rhoR=pbc.get_rhoR(cell, aoR, dm)
    print cell.vol/len(rhoR)*np.sum(rhoR) # 1.99981725342 (should be 2.0)
    
    print "(Hartree + vne) * DM"
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    print np.einsum("ij,ij",dm,neao+jao)

    ew_cut=(40,40,40)
    ew_eta=0.05
    for ew_eta in [0.1, 0.5, 1.]:
        ew=pbc.ewald(cell, gs, ew_eta, ew_cut)
        print "Ewald (eta, energy)", ew_eta, ew # should be same for all eta

    print "Ewald divergent terms summation", ew

    # These two should now match if the box is reasonably big to
    # remove images, and ngs is big.
    print "Total coulomb (analytic)", .5*np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne)) # -4.57559738004
    print "Total coulomb (fft coul + ewald)", np.einsum("ij,ij",dm,neao+.5*jao)+ew # -4.57948259115

    # Exc
    mf=RKS(cell, gs, ew_eta, ew_cut)
    mf.xc = 'LDA,VWN_RPA'

    pyscf.dft.rks.get_veff_(mf, cell, dm)
    print "Exc", mf._exc # -1.05967570089


def test_ks():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=np.eye(3.)*L
    
    # place atom in middle of big box
    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])

    # these are some exponents which are 
    # not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]
                         ]}
    mol.build()

    # benchmark first with molecular DFT calc
    m=pyscf.dft.rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print "Molecular DFT energy"
    print (m.scf()) # -2.64096172441

    # this is the PBC DFT calc!!
    cell=cl.Cell()
    cell.__dict__=mol.__dict__ # hacky way to make a cell
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 1
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()
    
    # points in grid (x,y,z)
    gs=np.array([80,80,80])

    # Ewald parameters
    ew_eta=0.05
    ew_cut=(40,40,40)
    mf=RKS(cell, gs, ew_eta, ew_cut)
    mf.xc = 'LDA,VWN_RPA'
    print (mf.scf()) # -2.64086844062, not bad!

def test_hf():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])

    mol.basis = { 'He': [[0,(0.8, 1.0)], 
                         [0,(1.0, 1.0)],
                         [0,(1.2, 1.0)]
                     ] }
    mol.build()

    # benchmark first with molecular HF calc
    m=pyscf.scf.hf.RHF(mol)
    print (m.scf()) # -2.63502450321874

    # this is the PBC HF calc!!
    cell=cl.Cell()
    cell.__dict__=mol.__dict__
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 1
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()
    
    gs=np.array([80,80,80])
    ew_eta=0.05
    ew_cut=(40,40,40)
    mf=RHF(cell, gs, ew_eta, ew_cut)

    print (mf.scf()) # -2.58766850182551: doesn't look good, but this is due
                     # to interaction of the exchange hole with its periodic
                     # image, which can only be removed with *very* large boxes.

    # Now try molecular type integrals for the exchange operator, 
    # and periodic integrals for Coulomb. This effectively
    # truncates the exchange operator. 
    mf.mol_ex=True 
    print (mf.scf()) # -2.63493445685: much better!

def test_moints():

    # not yet working
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    import pyscf.ao2mo
    import pyscf.ao2mo.incore
    

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])

    mol.basis = { 'He': "cc-pVDZ" }
    # mol.basis = { 'He': [[0,(0.8, 1.0)], 
    #                      [0,(1.0, 1.0)],
    #                      [0,(1.2, 1.0)]
    #                  ] }
    #mol.basis = { 'He': [[0,(0.8, 1.0)]] }
    mol.build()

    # this is the PBC HF calc!!
    cell=cl.Cell()
    cell.__dict__=mol.__dict__
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 1
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()

    gs=np.array([40,40,40])
    ew_eta=0.05
    ew_cut=(40,40,40)
    mf=RHF(cell, gs, ew_eta, ew_cut)
    #mf=pyscf.scf.RHF(mol)

    print (mf.scf()) 

    print "mo coeff shape", mf.mo_coeff.shape
    nmo=mf.mo_coeff.shape[1]
    print mf.mo_coeff

    eri_mo=pbc.get_mo_eri(cell, gs, [mf.mo_coeff, mf.mo_coeff], [mf.mo_coeff, mf.mo_coeff])
    
    eri_ao=pbc.get_ao_eri(cell, gs)
    eri_mo2=ao2mo.incore.general(np.real(eri_ao), (mf.mo_coeff,mf.mo_coeff,mf.mo_coeff,mf.mo_coeff), compact=False)
    print eri_mo.shape
    print eri_mo2.shape
    for i in range(nmo*nmo):
        for j in range(nmo*nmo):
            print i, j, np.real(eri_mo[i,j]), eri_mo2[i,j]


    print ("ERI dimension")
    print (eri_mo.shape), nmo
    Ecoul=0.
    Ecoul2=0.
    nocc=1

    print "diffs"
    for i in range(nocc):
        for j in range(nocc):
            Ecoul+=2*eri_mo[i*nmo+i,j*nmo+j]-eri_mo[i*nmo+j,i*nmo+j]
            Ecoul2+=2*eri_mo2[i*nmo+i,j*nmo+j]-eri_mo2[i*nmo+j,i*nmo+j]
    print Ecoul, Ecoul2

def test_dimer():

    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR

    B=BOHR
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    
    Lunit=10
    Ly=Lz=Lunit
    Lx=2*Lunit

    h=np.diag([Lx,Ly,Lz])
    
    # place atom in middle of big box
    #mol.atom.extend([['He', (0.5*Lunit*B,0.5*Ly*B,0.5*Lz*B)]])

    #mol.atom.extend([['He', (0.5*Lunit*B,0.5*Ly*B,0.5*Lz*B)],
    #                 ['He', (1.5*Lunit*B,0.5*Ly*B,0.5*Lz*B)]])
    mol.atom.extend([['He', (5*B,0.5*Ly*B,0.5*Lz*B)],
                     ['He', (6*B,0.5*Ly*B,0.5*Lz*B)]])

    # these are some exponents which are 
    # not hard to integrate
    mol.basis = { 'He': [[0, (1.0, 1.0)]] }
    mol.build()

    # benchmark first with molecular DFT calc
    m=pyscf.dft.rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print "overlap"
    print m.get_ovlp()
    print mol.intor_symmetric('cint1e_ovlp_sph')
    
    #print "Molecular DFT energy"
    print (m.scf()) # 

    # this is the PBC DFT calc!!
    cell=cl.Cell()
    cell.__dict__=mol.__dict__ # hacky way to make a cell
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.nimgs = 0
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()
    
    # points in grid (x,y,z)
    gs=np.array([100,100,100])



    # Ewald parameters
    ew_eta=0.05
    ew_cut=(100,100,100)
    
    # check ewald
    for ew_eta in [0.05, 0.1, 0.5, 1.]:
        ew=pbc.ewald(cell, gs, ew_eta, ew_cut)
        print "Ewald (eta, energy)", ew_eta, ew # should be same for all eta

    #ew_cut=(20,20,20)
    ovlp=get_ovlp(cell, gs)
    print "pbc ovlp"
    print ovlp
    mf=RKS(cell, gs, ew_eta, ew_cut)
    mf.xc='LDA,VWN_RPA'
    mf.scf()
    dm=mf.make_rdm1()
    print np.einsum('ij,ij',dm,ovlp)



if __name__ == '__main__':
    test_pp()
    test_components()
    test_ks()
    test_hf()
    test_moints()
    

