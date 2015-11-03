'''
Hartree-Fock for periodic systems at a *single* k-point.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.
'''

import copy
import numpy as np
import scipy.linalg
import pyscf.scf
import pyscf.scf.hf
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.dft.numint
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.pbc import tools
from pyscf.pbc import ao2mo
from pyscf.pbc.gto import pseudo
from pyscf.lib import logger
import scfint

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
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    chargs = [cell.atom_charge(i) for i in range(cell.natm)]
    SI = cell.get_SI()
    coulG = tools.get_coulG(cell)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = tools.ifft(vneG, cell.gs)

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR)
    return vne

def get_pp(cell, kpt=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    nao = aoR.shape[1]

    SI = cell.get_SI()
    vlocG = pseudo.get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)
    
    # vpploc in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc in reciprocal space
    aokplusG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aokplusG[:,i] = tools.fft(aoR[:,i]*np.exp(-1j*np.dot(coords,kpt)[:,0]), 
                                  cell.gs)
    ngs = aokplusG.shape[0]

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    hs, projGs = pseudo.get_projG(cell, kpt)
    for ia, [h_ia,projG_ia] in enumerate(zip(hs,projGs)):
        for l, h in enumerate(h_ia):
            nl = h.shape[0]
            for m in range(-l,l+1):
                for i in range(nl):
                    SPG_lmi = SI[ia,:] * projG_ia[l][m][i]
                    SPG_lmi_aoG = np.einsum('g,gp->p', SPG_lmi.conj(), aokplusG)
                    for j in range(nl):
                        SPG_lmj = SI[ia,:] * projG_ia[l][m][j]
                        SPG_lmj_aoG = np.einsum('g,gp->p', SPG_lmj.conj(), aokplusG)
                        # Note: There is no (-1)^l here.
                        vppnl += h[i,j]*np.einsum('p,q->pq', 
                                                   SPG_lmi_aoG.conj(), 
                                                   SPG_lmj_aoG)
    vppnl *= (1./ngs**2)

    #return vpploc
    return vpploc + vppnl

# def get_t2(cell, kpt=None):
#     '''Get the kinetic energy AO matrix.
    
#     Due to `kpt`, this is evaluated in real space using orbital gradients.

#     '''
#     if kpt is None:
#         kpt = np.zeros([3,1])
    
#     coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
#     # aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=True)
#     # ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4
#     aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=False)
#     ngs = aoR.shape[0]  # because we requested isgga, aoR.shape[0] = 4

#     Gv=cell.Gv
#     G2=np.einsum('ji,ji->i', Gv+kpt, Gv+kpt)

#     aoG=np.empty(aoR.shape, np.complex128)
#     TaoG=np.empty(aoR.shape, np.complex128)
#     nao = cell.nao_nr()
#     for i in range(nao):
#         aoG[:,i]=pyscf.pbc.tools.fft(aoR[:,i], cell.gs)
#         TaoG[:,i]=0.5*G2*aoG[:,i]

#     t = np.dot(aoG.T.conj(), TaoG)
#     t *= (cell.vol/ngs**2)


#     # t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
#     #          np.dot(aoR[2].T.conj(), aoR[2]) +
#     #          np.dot(aoR[3].T.conj(), aoR[3]))
#     # t *= (cell.vol/ngs)
    
#     return t

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.
    
    Due to `kpt`, this is evaluated in real space using orbital gradients.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, isgga=True)
    ngs = aoR.shape[1]  # because we requested isgga, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
             np.dot(aoR[2].T.conj(), aoR[2]) +
             np.dot(aoR[3].T.conj(), aoR[3]))
    t *= (cell.vol/ngs)
    
    return t

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])
    
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    # print "aoR"
    # print aoR
    # for i in range(aoR.shape[1]):
    #     print "AO", i, list(kpt.flat)
    #     print aoR[:,i]
    ngs = aoR.shape[0]

    #s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR).real
    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR)
    return s
    
def get_j(cell, dm, kpt=None):
    '''Get the Coulomb (J) AO matrix.

    '''
    if kpt is None:
        kpt = np.zeros([3,1])

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    ngs, nao = aoR.shape

    coulG = tools.get_coulG(cell)

    rhoR = pyscf.pbc.dft.numint.eval_rho(cell, aoR, dm)
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)

    vj = (cell.vol/ngs) * np.dot(aoR.T.conj(), vR.reshape(-1,1)*aoR)
    # print "dtype", aoR.dtype, vj.dtype
    # print "HACK HACK J"
    # return np.zeros_like(vj)
    return vj

def ewald(cell, ew_eta, ew_cut, verbose=logger.DEBUG):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Args:
        cell : instance of :class:`Cell`

        ew_eta, ew_cut : float
            The Ewald 'eta' and 'cut' parameters.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        ewald_params
        
    '''
    log = logger.Logger
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cell.stdout, verbose)

    chargs = [cell.atom_charge(i) for i in range(len(cell._atm))]
    coords = [cell.atom_coord(i) for i in range(len(cell._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange = range(-ew_cut[0],ew_cut[0]+1)
    ewyrange = range(-ew_cut[1],ew_cut[1]+1)
    ewzrange = range(-ew_cut[2],ew_cut[2]+1)
    ewxyz = cartesian_prod((ewxrange,ewyrange,ewzrange)).T

    # SLOW = True
    # if SLOW == True:
    #     ewxyz = ewxyz.T
    #     for ic, (ix, iy, iz) in enumerate(ewxyz):
    #         L = np.einsum('ij,j->i', cell._h, ewxyz[ic])

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
    #                         ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)
    #         else:
    #             for ia in range(cell.natm):
    #                 qi = chargs[ia]
    #                 ri = coords[ia]
    #                 for ja in range(cell.natm):
    #                     qj=chargs[ja]
    #                     rj=coords[ja]
    #                     r=np.linalg.norm(ri-rj+L)
    #                     ewovrl += qi * qj / r * scipy.special.erfc(ew_eta * r)

    # # else:
    nx = len(ewxrange)
    ny = len(ewyrange)
    nz = len(ewzrange)
    Lall = np.einsum('ij,jk->ik', cell._h, ewxyz).reshape(3,nx,ny,nz)
    #exclude the point where Lall == 0
    Lall[:,ew_cut[0],ew_cut[1],ew_cut[2]] = 1e200
    Lall = Lall.reshape(3,nx*ny*nz)
    Lall = Lall.T

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(ia):
            qj = chargs[ja]
            rj = coords[ja]
            r = np.linalg.norm(ri-rj)
            ewovrl += 2 * qi * qj / r * scipy.special.erfc(ew_eta * r)

    for ia in range(cell.natm):
        qi = chargs[ia]
        ri = coords[ia]
        for ja in range(cell.natm):
            qj = chargs[ja]
            rj = coords[ja]
            r1 = ri-rj + Lall
            r = np.sqrt(np.einsum('ji,ji->j', r1, r1))
            ewovrl += (qi * qj / r * scipy.special.erfc(ew_eta * r)).sum()

    ewovrl *= 0.5

    # last line of Eq. (F.5) in Martin 
    ewself  = -1./2. * np.dot(chargs,chargs) * 2 * ew_eta / np.sqrt(np.pi)
    ewself += -1./2. * np.sum(chargs)**2 * np.pi/(ew_eta**2 * cell.vol)
    
    # g-space sum (using g grid) (Eq. (F.6) in Martin, but note errors as below)
    SI = cell.get_SI()
    ZSI = np.einsum("i,ij->j", chargs, SI)

    # Eq. (F.6) in Martin is off by a factor of 2, the
    # exponent is wrong (8->4) and the square is in the wrong place
    #
    # Formula should be
    #   1/2 * 4\pi / Omega \sum_I \sum_{G\neq 0} |ZS_I(G)|^2 \exp[-|G|^2/4\eta^2]
    # where
    #   ZS_I(G) = \sum_a Z_a exp (i G.R_a)
    # See also Eq. (32) of ewald.pdf at 
    #   http://www.fisica.uniud.it/~giannozz/public/ewald.pdf

    coulG = tools.get_coulG(cell)
    absG2 = np.einsum('ij,ij->j',np.conj(cell.Gv),cell.Gv)

    ZSIG2 = np.abs(ZSI)**2
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    JexpG2 = coulG*expG2
    ewgI = np.dot(ZSIG2,JexpG2)
    ewg = .5*np.sum(ewgI)
    ewg /= cell.vol

    log.debug('Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg


class RHF(pyscf.scf.hf.RHF):
    '''RHF class adapted for PBCs.

    TODO: Maybe should create PBC SCF class derived from pyscf.scf.hf.SCF, then
          inherit from that.

    '''
    def __init__(self, cell, kpt=None, analytic_int=None):
        self.cell = cell
        pyscf.scf.hf.RHF.__init__(self, cell)
        self.grids = pyscf.pbc.dft.gen_grid.UniformGrids(cell)
        self.mol_ex = False

        if kpt is None:
            kpt = np.array([0,0,0])

        self.kpt = np.array([0,0,0])

        if analytic_int == None:
            self.analytic_int = False
        else:
            self.analytic_int = True

        self._keys = self._keys.union(['cell', 'grids', 'mol_ex', 'kpt', 'analytic_int'])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'Grid size = (%d, %d, %d)', 
                    self.cell.gs[0], self.cell.gs[1], self.cell.gs[2])
        logger.info(self, 'Use molecule exchange = %s', self.mol_ex)

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt

        if self.analytic_int:
            print "USING ANALYTIC INTS"
            return scfint.get_hcore(cell, np.reshape(kpt, (3,1)))
        else:
            return get_hcore(cell, np.reshape(kpt, (3,1)))

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        
        if self.analytic_int:
            print "USING ANALYTIC INTS"
            return scfint.get_ovlp(cell, np.reshape(kpt, (3,1)))
        else:
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
            if lib.norm(kpt) > 1.e-15:
                raise RuntimeError("Non-zero k points not implemented for exchange")
            self._eri = np.real(ao2mo.get_ao_eri(cell))

        vj, vk = pyscf.scf.hf.RHF.get_jk_(self, cell, dm, hermi) 
        
        if self.mol_ex: # use molecular exchange, but periodic J
            log.debug('K PBC build: using molecular integrals')
            mol_eri = pyscf.scf._vhf.int2e_sph(cell._atm, cell._bas, cell._env)
            mol_vj, vk = pyscf.scf.hf.dot_eri_dm(mol_eri, dm, hermi)

        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_elec=self.energy_elec(dm, h1e, vhf)[0]
        #print "ELEC ENERGY", e_elec, np.dot(dm.ravel(),h1e.ravel())
        return self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
    
    def ewald_nuc(self):
        return ewald(self.cell, self.cell.ew_eta, self.cell.ew_cut)
        
    def get_band_fock_ovlp(self, fock, ovlp, band_kpt):
        '''Reconstruct Fock operator at a given band kpt 
           (not necessarily in list of k pts)

        Returns:
            fock : (nao, nao) ndarray
            ovlp : (nao, nao) ndarray
        '''
        iS = scipy.linalg.inv(ovlp)
        iSfockiS = np.dot(np.conj(iovlp.T), np.dot(fock, iovlp))

        # band_ovlp[p,q] = <p(0)|q(k)>
        band_ovlp = mf.get_ovlp(band_kpt)
        # Fb[p,q] = \sum_{rs} <p(k)|_r(0)> <r(0)|F|s(0)> <_s(0)|q(k>
        Fb = np.dot(np.conj(band_ovlp.T), np.dot(isFockiS, band_ovlp))
        # Sb[p,q] = \sum_{rs} <p(k)|_r(0)> <r(0)|s(0)> <_s(0)|q(k>
        Sb = np.dot(np.conj(band_ovlp.T), np.dot(iovlp, band_ovlp))

        return Fb, Sb

