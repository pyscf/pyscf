'''
Hartree-Fock for periodic systems at a *single* k-point.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.
'''

import numpy as np
import scipy.linalg
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import pyscf.pbc.dft
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo

from pyscf.lib import logger

def get_hcore(cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    if cell.pseudo:
        hcore = get_pp(cell, kpt) + get_jvloc_G0(cell, kpt)
    else:
        hcore = get_nuc(cell, kpt)
    hcore += get_t(cell, kpt)
    return hcore

def get_jvloc_G0(cell, kpt=None):
    '''Get the (separately) divergent Hartree + Vloc G=0 contribution.'''

    return 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * get_ovlp(cell, kpt)

def get_nuc(cell, kpt=None):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.get_ao(cell, coords, kpt)

    chargs = [cell.atom_charge(i) for i in range(cell.natm)]
    Gv = tools.get_Gv(cell)
    SI = tools.get_SI(cell, Gv)
    coulG = tools.get_coulG(cell)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = tools.ifft(vneG, cell.gs)

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR).real
    return vne

def get_pp(cell, kpt=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.get_ao(cell, coords, kpt)
    nao = aoR.shape[1]

    Gv = tools.get_Gv(cell)
    SI = tools.get_SI(cell, Gv)
    vlocG = pseudo.get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)
    
    # vpploc in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR).real

    # vppnonloc in reciprocal space
    aoG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aoG[:,i] = tools.fft(aoR[:,i], cell.gs)
    ngs = aoG.shape[0]

    vppnl = np.zeros((nao,nao))
    hs, projGs = pseudo.get_projG(cell)
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

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.
    
    Due to `kpt`, this is evaluated in real space using orbital gradients.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.get_ao(cell, coords, kpt, isgga=True)
    ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]).real +
             np.dot(aoR[2].T.conj(), aoR[2]).real +
             np.dot(aoR[3].T.conj(), aoR[3]).real)
    t *= (cell.vol/ngs)
    return t

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.get_ao(cell, coords, kpt)
    ngs = aoR.shape[0]

    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR).real
    return s
    
def get_j(cell, dm, kpt=None):
    '''Get the Coulomb (J) AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.get_ao(cell, coords, kpt)
    ngs, nao = aoR.shape

    coulG = tools.get_coulG(cell)

    rhoR = pyscf.pbc.dft.numint.get_rho(cell, aoR, dm)
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)

    vj = (cell.vol/ngs) * np.dot(aoR.T.conj(), vR.reshape(-1,1)*aoR).real
    return vj


class RHF(pyscf.scf.hf.RHF):
    '''RHF class adapted for PBCs.

    TODO: Maybe should create PBC SCF class derived from pyscf.scf.hf.SCF, then
          inherit from that.

    '''
    def __init__(self, cell, kpt=None):
        self.cell = cell
        pyscf.scf.hf.RHF.__init__(self, cell)
        self.grids = pyscf.pbc.dft.gen_grid.UniformGrids(cell)
        #TODO(TCB): Does RHF class need its own ewald params?
        self.ew_eta = cell.ew_eta
        self.ew_cut = cell.ew_cut
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
                    self.cell.gs[0], self.cell.gs[1], self.cell.gs[2])
        logger.info(self, 'Use molecule exchange = %s', self.mol_ex)

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_hcore(cell, np.reshape(kpt, (3,1)))

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_ovlp(cell, np.reshape(kpt, (3,1)))

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        return get_j(cell, dm, np.reshape(kpt, (3,1)))

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
            self._eri = np.real(tools.get_ao_eri(cell))

        vj, vk = pyscf.scf.hf.RHF.get_jk_(self, cell, dm, hermi) 
        
        if self.mol_ex: # use molecular exchange, but periodic J
            log.debug('K PBC build: using molecular integrals')
            mol_eri = pyscf.scf._vhf.int2e_sph(cell._atm, cell._bas, cell._env)
            mol_vj, vk = pyscf.scf.hf.dot_eri_dm(mol_eri, dm, hermi)

        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        return self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
    
    def ewald_nuc(self):
        return tools.ewald(self.cell, self.ew_eta, self.ew_cut)
        

