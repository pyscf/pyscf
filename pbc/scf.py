'''
SCF (Hartree-Fock and DFT) tools for periodic systems at a *single* k-point.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.

'''

import numpy as np
import scipy.linalg
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import cell as cl
import pbc
import pp

from pyscf.lib import logger

def get_hcore(mf, cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    if cell.pseudo:
        hcore = get_pp(cell, mf.gs, kpt) + get_jvloc_G0(cell, mf.gs, kpt)
    else:
        hcore = get_nuc(cell, mf.gs, kpt)
    hcore += get_t(cell, mf.gs, kpt)
    return hcore

def get_jvloc_G0(cell, gs, kpt=None):
    '''Get the (separately) divergent Hartree + Vloc G=0 contribution.'''

    return 1./cell.vol * np.sum(pp.get_alphas(cell)) * get_ovlp(cell, gs, kpt)

def get_nuc(cell, gs, kpt=None):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords, kpt)

    chargs = [cell.atom_charge(i) for i in range(cell.natm)]
    Gv = pbc.get_Gv(cell, gs)
    SI = pbc.get_SI(cell, Gv)
    coulG = pbc.get_coulG(cell, gs)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = pbc.ifft(vneG, gs)

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR).real
    return vne

def get_pp(cell, gs, kpt=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords, kpt)
    nao = aoR.shape[1]

    Gv = pbc.get_Gv(cell, gs)
    SI = pbc.get_SI(cell, Gv)
    vlocG = pp.get_vlocG(cell, gs)
    vpplocG = -np.sum(SI * vlocG, axis=0)
    
    # vpploc in real-space
    vpplocR = pbc.ifft(vpplocG, gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR).real

    # vppnonloc in reciprocal space
    aoG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aoG[:,i] = pbc.fft(aoR[:,i], gs)
    ngs = aoG.shape[0]

    vppnl = np.zeros((nao,nao))
    hs, projGs = pp.get_projG(cell, gs)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    SPG_lmi_aoG = np.einsum('g,gp->p', SPG_lmi.conj(), aoG)
                    for j in range(nl):
                        SPG_lmj= SI[ia,:] * projG_ia[l][m][j]
                        SPG_lmj_aoG = np.einsum('g,gp->p', SPG_lmj.conj(), aoG)
                        vppnl += -h[i,j]*np.einsum('p,q->pq', 
                                                   SPG_lmi_aoG.conj(), 
                                                   SPG_lmj_aoG).real
    vppnl *= (1./ngs**2)
    return vpploc + vppnl

def get_t(cell, gs, kpt=None):
    '''Get the kinetic energy AO matrix.
    
    Due to `kpt`, this is evaluated in real space using orbital gradients.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords, kpt, isgga=True)
    ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]).real +
             np.dot(aoR[2].T.conj(), aoR[2]).real +
             np.dot(aoR[3].T.conj(), aoR[3]).real)
    t *= (cell.vol/ngs)
    return t

def get_ovlp(cell, gs, kpt=None):
    '''Get the overlap AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords, kpt)
    ngs = aoR.shape[0]

    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR).real
    return s
    
def get_j(cell, dm, gs, kpt=None):
    '''Get the Coulomb (J) AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords, kpt)
    ngs, nao = aoR.shape

    coulG = pbc.get_coulG(cell, gs)

    rhoR = pbc.get_rhoR(cell, aoR, dm)
    rhoG = pbc.fft(rhoR, gs)

    vG = coulG*rhoG
    vR = pbc.ifft(vG, gs)

    vj = (cell.vol/ngs) * np.dot(aoR.T.conj(), vR.reshape(-1,1)*aoR).real
    return vj


class RHF(pyscf.scf.hf.RHF):
    '''RHF class adapted for PBCs.

    TODO: Maybe should create PBC SCF class derived from pyscf.scf.hf.SCF, then
          inherit from that.

    '''
    def __init__(self, cell, gs, ew_eta, ew_cut, kpt=None):
        self.cell = cell
        pyscf.scf.hf.RHF.__init__(self, cell)
        self.grids = pbc.UniformGrids(cell, gs)
        self.gs = gs
        self.ew_eta = ew_eta
        self.ew_cut = ew_cut
        self.mol_ex = False
        if kpt is None:
            kpt = np.array([0,0,0])

        self.kpt = np.array([0,0,0])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'Ewald eta = %g', self.ew_eta)
        logger.info(self, 'Ewald real-space cutoff = (%d, %d, %d)', 
                    self.ew_cut[0], self.ew_cut[1], self.ew_cut[2])
        logger.info(self, 'Grid size = (%d, %d, %d)', 
                    self.gs[0], self.gs[1], self.gs[2])
        logger.info(self, 'Use molecule exchange = %s', self.mol_ex)

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_hcore(self, cell, np.reshape(kpt, (3,1)))

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_ovlp(cell, self.gs, np.reshape(kpt, (3,1)))

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        return get_j(cell, dm, self.gs, np.reshape(kpt, (3,1)))

    def get_jk_(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG, kpt=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.

        '''
        if cell is None:
            cell = self.cell
        
        log = logger.Logger
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(cell.stdout, verbose)

        log.debug('JK PBC build: incore only with PBC integrals')

        if self._eri is None:
            log.debug('Building PBC AO integrals')
            self._eri = np.real(pbc.get_ao_eri(cell, self.gs))

        vj, vk = pyscf.scf.hf.RHF.get_jk_(self, cell, dm, hermi) 
        
        if self.mol_ex: # use molecular exchange, but periodic J
            log.debug('K PBC build: using molecular integrals')
            mol_eri = pyscf.scf._vhf.int2e_sph(cell._atm, cell._bas, cell._env)
            mol_vj, vk = pyscf.scf.hf.dot_eri_dm(mol_eri, dm, hermi)

        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
    
    def ewald_nuc(self):
        return pbc.ewald(self.cell, self.gs, self.ew_eta, self.ew_cut)
        
class RKS(RHF):
    '''RKS class adapted for PBCs. 
    
    This is a literal duplication of the molecular RKS class with some `mol`
    variables replaced by `cell`.

    '''
    def __init__(self, cell, gs, ew_eta, ew_cut, kpt=None):
        RHF.__init__(self, cell, gs, ew_eta, ew_cut, kpt)
        self.xc = 'LDA,VWN'
        self._ecoul = 0
        self._exc = 0
        self._numint = pbc._NumInt(self.kpt) # use periodic images of AO in 
                                             # numerical integration
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        return pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last, 
                                       hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = get_hcore(self, self.cell)
        return pyscf.dft.rks.energy_elec(self, dm, h1e)
    

def test_components(pseudo=None):
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    #mol.verbose = 7
    #mol.output = None

    L = 60
    #L = 3
    h = np.eye(3)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = { 'He': 'STO-3G'}
    
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    #m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell = cl.Cell()
    cell.__dict__ = mol.__dict__

    cell.h = h
    cell.vol = scipy.linalg.det(cell.h)
    cell.nimgs = [0,0,0]
    cell.pseudo = pseudo 
    cell.basis = 'gth-szv'
    cell.output = None
    cell.verbose = 7
    cell.build()
    print cell._basis
    
    #gs = np.array([10,10,10]) # number of G-points in grid. Real-space dim=2*gs+1
    gs = np.array([80,80,80]) # number of G-points in grid. Real-space dim=2*gs+1
    Gv = pbc.get_Gv(cell, gs)

    dm = m.make_rdm1()

    # These should match reasonably well (roughly with accuracy of normalization)
    print "Kinetic energy" 
    tao = get_t(cell, gs) 
    tao2 = mol.intor_symmetric('cint1e_kin_sph') 
    print np.dot(np.ravel(tao), np.ravel(dm))  # 2.82793077196
    print np.dot(np.ravel(tao2), np.ravel(dm)) # 2.82352636524
    
    print "Overlap"
    sao = get_ovlp(cell, gs)
    print np.dot(np.ravel(sao), np.ravel(dm)) # 1.99981725342
    print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm)) # 2.0

    # The next two entries should *not* match, since G=0 component is removed
    print "Coulomb (G!=0)"
    jao = get_j(cell, dm, gs)
    print np.dot(np.ravel(dm),np.ravel(jao))  # 4.03425518427
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm))) # 4.22285177049

    # The next two entries should *not* match, since G=0 component is removed
    print "Nuc-el (G!=0)"
    if cell.pseudo:
        vppao = get_pp(cell, gs)
        print np.dot(np.ravel(dm), np.ravel(vppao)) 
    else:
        neao = get_nuc(cell, gs)
        print np.dot(np.ravel(dm), np.ravel(neao))  # -6.50203360062
    vne = mol.intor_symmetric('cint1e_nuc_sph') 
    print np.dot(np.ravel(dm), np.ravel(vne))   # -6.68702326551

    print "Normalization" 
    coords = pbc.setup_uniform_grids(cell, gs)
    aoR = pbc.get_aoR(cell, coords)
    rhoR = pbc.get_rhoR(cell, aoR, dm)
    print cell.vol/len(rhoR)*np.sum(rhoR) # 1.99981725342 (should be 2.0)
    
    print "(Hartree + vne) * DM"
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))+np.dot(np.ravel(dm), np.ravel(vne))
    if cell.pseudo:
        print np.einsum("ij,ij",dm,vppao+jao+get_jvloc_G0(cell,gs))
    else:
        print np.einsum("ij,ij",dm,neao+jao)

    ew_cut = (40,40,40)
    ew_eta = 0.05
    for ew_eta in [0.1, 0.5, 1.]:
        ew = pbc.ewald(cell, gs, ew_eta, ew_cut)
        print "Ewald (eta, energy)", ew_eta, ew # should be same for all eta

    print "Ewald divergent terms summation", ew

    # These two should now match if the box is reasonably big to
    # remove images, and ngs is big.
    print "Total coulomb (analytic) ", 
    print (.5*np.dot(np.ravel(dm), np.ravel(m.get_j(dm))) 
            + np.dot(np.ravel(dm), np.ravel(vne)) ) # -4.57559738004
    if not cell.pseudo:
        print "Total coulomb (fft coul + ewald)", 
        print np.einsum("ij,ij", dm, neao + .5*jao) + ew # -4.57948259115

    # Exc
    mf = RKS(cell, gs, ew_eta, ew_cut)
    mf.xc = 'LDA,VWN_RPA'

    pyscf.dft.rks.get_veff_(mf, cell, dm)
    print "Exc", mf._exc # -1.05967570089


def test_ks(pseudo=None):
    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR

    B = BOHR
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L = 60
    h = np.eye(3)*L
    
    # place atom in middle of big box
    mol.atom.extend([['He', (B*L/2.,B*L/2.,B*L/2.)], ])

    # these are some exponents which are not hard to integrate
    mol.basis = { 'He': [[0, (0.8, 1.0)],
                         [0, (1.0, 1.0)],
                         [0, (1.2, 1.0)]] }
    mol.build()

    print "coordinates"
    print np.array([mol.atom_coord(i) for i in range(mol.natm)])

    # benchmark first with molecular DFT calc
    m = pyscf.dft.rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    print "Molecular DFT energy"
    print (m.scf()) # -2.64096172441

    # now do the PBC DFT calc
    cell = cl.Cell()
    cell.__dict__ = mol.__dict__ # hacky way to make a cell
    cell.h = h
    cell.vol = scipy.linalg.det(cell.h)
    cell.nimgs = [0,0,0]
    cell.pseudo = pseudo
    cell.output = None
    cell.verbose = 7
    cell.build()
    
    # points in grid (x,y,z)
    gs = np.array([80,80,80])

    # Ewald parameters
    ew_eta, ew_cut = pbc.ewald_params(cell, gs, 1.e-7)

    mf = RKS(cell, gs, ew_eta, ew_cut)
    mf.xc = 'LDA,VWN_RPA'
    mf.kpt = np.reshape(np.array([1,1,1]), (3,1))
    print (mf.scf()) 
    # gs    mf.scf()
    # 80    -2.63907898485
    # 90    -2.64065784113
    # 100   -2.64086844062

def test_hf(pseudo=None):
    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR

    B = BOHR
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L = 60
    h = np.eye(3)*L

    mol.atom.extend([['He', (B*L/2.,B*L/2.,B*L/2.)], ])

    mol.basis = { 'He': [[0,(0.8, 1.0)], 
                         [0,(1.0, 1.0)],
                         [0,(1.2, 1.0)]] }
    mol.build()

    # benchmark first with molecular HF calc
    m = pyscf.scf.hf.RHF(mol)
    print "Molecular HF energy"
    print (m.scf()) # -2.63502450321874

    # now do the PBC HF calc
    cell = cl.Cell()
    cell.__dict__ = mol.__dict__
    cell.h = h
    cell.vol = scipy.linalg.det(cell.h)
    cell.nimgs = [0,0,0]
    cell.pseudo = pseudo 
    cell.output = None
    cell.verbose = 7
    cell.build()
    
    gs = np.array([80,80,80])
    # Ewald parameters
    #ew_eta = 0.05
    #ew_cut = (40,40,40)
    ew_eta, ew_cut = pbc.ewald_params(cell, gs, 1.e-7)
    mf = RHF(cell, gs, ew_eta, ew_cut)

    print (mf.scf()) # -2.58766850182551: doesn't look good, but this is due
                     # to interaction of the exchange hole with its periodic
                     # image, which can only be removed with *very* large boxes.

    # Now try molecular type integrals for the exchange operator, 
    # and periodic integrals for Coulomb. This effectively
    # truncates the exchange operator. 
    mf.mol_ex = True 
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
    h=np.eye(3)*L

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
    cell.nimgs = [1,1,1]
    cell.pseudo=None
    cell.output=None
    cell.verbose=7
    cell.build()

    gs=np.array([40,40,40])
    # Ewald parameters
    #ew_eta=0.05
    #ew_cut=(40,40,40)
    ew_eta, ew_cut = pbc.ewald_params(cell, gs, 1.e-7)
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


if __name__ == '__main__':
    test_components('gth-lda')
    #test_hf('gth-lda')
    #test_ks('gth-lda')
    #test_moints()
    

