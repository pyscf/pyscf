'''
Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import sys
import time
import numpy as np
import scipy.linalg
import pyscf.lib
import pyscf.scf
import pyscf.scf.hf
import pyscf.gto
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.dft.numint
import pyscf.pbc.scf
from pyscf.lib import logger
from pyscf.lib.numpy_helper import cartesian_prod
from pyscf.pbc import tools
from pyscf.pbc import ao2mo
from pyscf.pbc.gto import pseudo
from pyscf.pbc.scf import scfint
import pyscf.pbc.scf.chkfile


def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    ngs = len(aoR)

    s = (cell.vol/ngs) * np.dot(aoR.T.conj(), aoR)
    return s


def get_hcore(cell, kpt=np.zeros(3)):
    '''Get the core Hamiltonian AO matrix.
    '''
    hcore = get_t(cell, kpt)
    if cell.pseudo:
        hcore += ( get_pp(cell, kpt) + get_jvloc_G0(cell, kpt) )
    else:
        hcore += get_nuc(cell, kpt)

    return hcore


def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.

    Note: Evaluated in real space using orbital gradients, for improved accuracy.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, deriv=1)
    ngs = aoR.shape[1]  # because we requested deriv=1, aoR.shape[0] = 4

    t = 0.5*(np.dot(aoR[1].T.conj(), aoR[1]) +
             np.dot(aoR[2].T.conj(), aoR[2]) +
             np.dot(aoR[3].T.conj(), aoR[3]))
    t *= (cell.vol/ngs)

    return t


def get_t_pw(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix using the PW resolution.

    Note: Incurs error due to finite resolution of the gradient operator.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt, deriv=0)
    nao = cell.nao_nr()

    kG = kpt + cell.Gv
    abskG2 = np.einsum('gi,gi->g', kG, kG)

    aokG = np.empty(aoR.shape, np.complex128)
    TaokG = np.empty(aoR.shape, np.complex128)
    nao = cell.nao_nr()
    for i in range(nao):
        aokG[:,i] = tools.fftk(aoR[:,i], cell.gs, coords, kpt)
        TaokG[:,i] = 0.5*abskG2*aokG[:,i]

    ngs = len(aokG)
    t = np.dot(aokG.T.conj(), TaokG)
    t *= (cell.vol/ngs**2)

    return t


def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    chargs = [cell.atom_charge(i) for i in range(cell.natm)]
    SI = cell.get_SI()
    coulG = tools.get_coulG(cell)
    vneG = -np.dot(chargs,SI) * coulG
    vneR = tools.ifft(vneG, cell.gs)

    vne = np.dot(aoR.T.conj(), vneR.reshape(-1,1)*aoR)
    return vne

def get_pp(cell, kpt=np.zeros(3)):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
    nao = cell.nao_nr()

    SI = cell.get_SI()
    vlocG = pseudo.get_vlocG(cell)
    vpplocG = -np.sum(SI * vlocG, axis=0)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs)
    vpploc = np.dot(aoR.T.conj(), vpplocR.reshape(-1,1)*aoR)

    # vppnonloc evaluated in reciprocal space
    aokG = np.empty(aoR.shape, np.complex128)
    for i in range(nao):
        aokG[:,i] = tools.fftk(aoR[:,i], cell.gs, coords, kpt)
    ngs = len(aokG)

    fakemol = pyscf.gto.Mole()
    fakemol._atm = np.zeros((1,pyscf.gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,pyscf.gto.BAS_SLOTS), dtype=np.int32)
    ptr = pyscf.gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,pyscf.gto.NPRIM_OF ] = 1
    fakemol._bas[0,pyscf.gto.NCTR_OF  ] = 1
    fakemol._bas[0,pyscf.gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,pyscf.gto.PTR_COEFF] = ptr+4
    Gv = np.asarray(cell.Gv+kpt)
    G_rad = pyscf.lib.norm(Gv, axis=1)

    vppnl = np.zeros((nao,nao), dtype=np.complex128)
    for ia in range(cell.natm):
        pp = cell._pseudo[cell.atom_symbol(ia)]
        for l, proj in enumerate(pp[5:]):
            rl, nl, hl = proj
            if nl > 0:
                hl = np.asarray(hl)
                fakemol._bas[0,pyscf.gto.ANG_OF] = l
                fakemol._env[ptr+3] = .5*rl**2
                fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                pYlm_part = pyscf.dft.numint.eval_ao(fakemol, Gv, deriv=0)

                pYlm = np.empty((nl,l*2+1,ngs))
                for k in range(nl):
                    qkl = pseudo.pp._qli(G_rad*rl, l, k)
                    pYlm[k] = pYlm_part.T * qkl
                # pYlm is real
                SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
    vppnl *= (1./ngs**2)

    return vpploc + vppnl


def get_jvloc_G0(cell, kpt=np.zeros(3)):
    '''Get the (separately divergent) Hartree + Vloc G=0 contribution.
    '''
    return 1./cell.vol * np.sum(pseudo.get_alphas(cell)) * get_ovlp(cell, kpt)


def get_j(cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpt_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix.
    '''
    if kpt_band is None:
        kpt1 = kpt2 = kpt
    else:
        kpt1 = kpt_band
        kpt2 = kpt

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR_k1 = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt1)
    aoR_k2 = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt2)
    ngs, nao = aoR_k1.shape

    vjR_k2 = get_vjR(cell, dm, aoR_k2)
    vj = (cell.vol/ngs) * np.dot(aoR_k1.T.conj(), vjR_k2.reshape(-1,1)*aoR_k1)

    return vj


def get_jk(mf, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpt_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpt_band : (3,) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix.
    '''
    if kpt_band is None:
        kpt1 = kpt2 = kpt
    else:
        kpt1 = kpt_band
        kpt2 = kpt

    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR_k1 = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt1)
    aoR_k2 = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt2)
    ngs, nao = aoR_k1.shape

    vjR_k2 = get_vjR(cell, dm, aoR_k2)
    vj = (cell.vol/ngs) * np.dot(aoR_k1.T.conj(), vjR_k2.reshape(-1,1)*aoR_k1)

    vkR_k1k2 = get_vkR(mf, cell, aoR_k1, aoR_k2, kpt1, kpt2)
    aoR_dm_k2 = np.dot(aoR_k2, dm)
    tmp_Rq = np.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dm_k2)
    vk = (cell.vol/ngs) * np.dot(aoR_k1.T.conj(), tmp_Rq)
    #vk = (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq', dm, aoR_k1.conj(),
    #                                vkR_k1k2, aoR_k2)
    return vj, vk


def get_vjR(cell, dm, aoR):
    '''Get the real-space Hartree potential of the given density matrix.

    Returns:
        vR : (ngs,) ndarray
            The real-space Hartree potential at every grid point.
    '''
    coulG = tools.get_coulG(cell)

    rhoR = pyscf.pbc.dft.numint.eval_rho(cell, aoR, dm)
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)
    return vR


def get_vkR(mf, cell, aoR_k1, aoR_k2, kpt1, kpt2):
    '''Get the real-space 2-index "exchange" potential V_{i,k1; j,k2}(r).

    Kwargs:
        kpts : (nkpts, 3) ndarray
            The sampled k-points; may be required for G=0 correction.

    Returns:
        vR : (ngs, nao, nao) ndarray
            The real-space "exchange" potential at every grid point, for all
            AO pairs.

    Note:
        This is essentially a density-fitting or resolution-of-the-identity.
        The returned object is of size ngs*nao**2 and could be precomputed and
        saved in vhfopt.
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    ngs, nao = aoR_k1.shape

    coulG = tools.get_coulG(cell, kpt1-kpt2, exx=True, mf=mf)

    vR = np.zeros((ngs,nao,nao), dtype=np.complex128)
    for i in range(nao):
        for j in range(nao):
            rhoR = aoR_k1[:,i] * aoR_k2[:,j].conj()
            rhoG = tools.fftk(rhoR, cell.gs, coords, kpt1-kpt2)
            vG = coulG*rhoG
            vR[:,i,j] = tools.ifftk(vG, cell.gs, coords, kpt1-kpt2)
    return vR


def ewald(cell, ew_eta, ew_cut, verbose=logger.NOTE):
    '''Perform real (R) and reciprocal (G) space Ewald sum for the energy.

    Formulation of Martin, App. F2.

    Returns:
        float
            The Ewald energy consisting of overlap, self, and G-space sum.

    See Also:
        pyscf.pbc.gto.get_ewald_params
    '''
    #if isinstance(verbose, logger.Logger):
    #    log = verbose
    #else:
    #    log = logger.Logger(cell.stdout, verbose)

    chargs = [cell.atom_charge(i) for i in range(len(cell._atm))]
    coords = [cell.atom_coord(i) for i in range(len(cell._atm))]

    ewovrl = 0.

    # set up real-space lattice indices [-ewcut ... ewcut]
    ewxrange = range(-ew_cut[0],ew_cut[0]+1)
    ewyrange = range(-ew_cut[1],ew_cut[1]+1)
    ewzrange = range(-ew_cut[2],ew_cut[2]+1)
    ewxyz = cartesian_prod((ewxrange,ewyrange,ewzrange)).T

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
    absG2 = np.einsum('gi,gi->g', cell.Gv, cell.Gv)

    ZSIG2 = np.abs(ZSI)**2
    expG2 = np.exp(-absG2/(4*ew_eta**2))
    JexpG2 = coulG*expG2
    ewgI = np.dot(ZSIG2,JexpG2)
    ewg = .5*np.sum(ewgI)
    ewg /= cell.vol

    #log.debug('Ewald components = %.15g, %.15g, %.15g', ewovrl, ewself, ewg)
    return ewovrl + ewself + ewg


#FIXME: project initial guess for k-point
def init_guess_by_chkfile(cell, chkfile_name, project=True):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 2D ndarray
    '''
    from pyscf.pbc.scf import addons
    chk_cell, scf_rec = pyscf.pbc.scf.chkfile.load_scf(chkfile_name)

    def fproj(mo):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell)
        else:
            return mo
    if scf_rec['mo_coeff'].ndim == 2:
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = pyscf.scf.hf.make_rdm1(fproj(mo), mo_occ)
    else:  # UHF
        mo = scf_rec['mo_coeff']
        mo_occ = scf_rec['mo_occ']
        dm = pyscf.scf.hf.make_rdm1(fproj(mo[0]), mo_occ[0]) \
           + pyscf.scf.hf.make_rdm1(fproj(mo[1]), mo_occ[1])
    return dm


def dot_eri_dm_complex(eri, dm, hermi=0):
    '''Compute J, K matrices in terms of the given 2-electron integrals and
    density matrix if either eri or dm is complex.

    Args:
        eri : ndarray
            8-fold or 4-fold ERIs
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        Depending on the given dm, the function returns one J and one K matrix,
        or a list of J matrices and a list of K matrices, corresponding to the
        input density matrices.
    '''
    eri_re = np.ascontiguousarray(eri.real)
    eri_im = np.ascontiguousarray(eri.imag)

    dm_re = np.ascontiguousarray(dm.real)
    dm_im = np.ascontiguousarray(dm.imag)

    vj_rr, vk_rr = pyscf.scf.hf.dot_eri_dm(eri_re, dm_re, hermi)
    vj_ir, vk_ir = pyscf.scf.hf.dot_eri_dm(eri_im, dm_re, hermi)
    vj_ri, vk_ri = pyscf.scf.hf.dot_eri_dm(eri_re, dm_im, hermi)
    vj_ii, vk_ii = pyscf.scf.hf.dot_eri_dm(eri_im, dm_im, hermi)

    vj = vj_rr - vj_ii + 1j*(vj_ir + vj_ri)
    vk = vk_rr - vk_ii + 1j*(vk_ir + vk_ri)

    return vj, vk


# TODO: Maybe should create PBC SCF class derived from pyscf.scf.hf.SCF, then
# inherit from that.
class RHF(pyscf.scf.hf.RHF):
    '''RHF class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.
    '''
    def __init__(self, cell, kpt=None, exxdiv='ewald'):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        pyscf.scf.hf.RHF.__init__(self, cell)

        if kpt is None:
            self.kpt = np.zeros(3)
        else:
            self.kpt = kpt
        if np.allclose(self.kpt, np.zeros(3)):
            self._dtype = np.float64
        else:
            self._dtype = np.complex128

        self.exxdiv = exxdiv

        self._keys = self._keys.union(['cell', 'kpt', 'exxdiv'])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'Exchange divergence treatment = %s', self.exxdiv)

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt

        return scfint.get_hcore(cell, kpt)

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt

        return scfint.get_ovlp(cell, kpt)

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        return self.get_jk_(cell, dm, hermi, kpt, kpt_band)
    def get_jk_(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        Note the incore version, which initializes an _eri array in memory.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (time.clock(), time.time())
        vj, vk = get_jk(self, cell, dm, hermi, self.opt, kpt, kpt_band)
        # TODO: Check incore, direct_scf, _eri's, etc
        #if self._eri is not None or cell.incore_anyway or self._is_mem_enough():
        #    print "self._is_mem_enough() =", self._is_mem_enough()
        #    if self._eri is None:
        #        logger.debug(self, 'Building PBC AO integrals incore')
        #        if kpt is not None and pyscf.lib.norm(kpt) > 1.e-15:
        #            raise RuntimeError("Non-zero kpts not implemented for incore eris")
        #        self._eri = ao2mo.get_ao_eri(cell)
        #    if np.iscomplexobj(dm) or np.iscomplexobj(self._eri):
        #        vj, vk = dot_eri_dm_complex(self._eri, dm, hermi)
        #    else:
        #        vj, vk = pyscf.scf.hf.dot_eri_dm(self._eri, dm, hermi)
        #else:
        #    if self.direct_scf:
        #        self.opt = self.init_direct_scf(cell)
        #    vj, vk = get_jk(cell, dm, hermi, self.opt, kpt)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Compute J matrix for the given density matrix.
        '''
        #return self.get_jk(cell, dm, hermi, kpt, kpt_band)[0]
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        cpu0 = (time.clock(), time.time())
        vj = get_j(cell, dm, hermi, self.opt, kpt, kpt_band)
        logger.timer(self, 'vj', *cpu0)
        return vj

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpt_band=None):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpt_band)[1]

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpt_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        # TODO: Check incore, direct_scf, _eri's, etc
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpt_band)
        return vj - vk * .5

    def get_jk_incore(self, cell=None, dm=None, hermi=1, verbose=logger.DEBUG, kpt=None):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        log = logger.Logger
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(cell.stdout, verbose)

        log.debug('JK PBC build: incore only with PBC integrals')

        if self._eri is None:
            log.debug('Building PBC AO integrals')
            if kpt is not None and pyscf.lib.norm(kpt) > 1.e-15:
                raise RuntimeError("Non-zero k points not implemented for exchange")
            self._eri = ao2mo.get_ao_eri(cell)

        if np.iscomplexobj(dm) or np.iscomplexobj(self._eri):
            vj, vk = dot_eri_dm_complex(self._eri, dm, hermi)
        else:
            vj, vk = pyscf.scf.hf.dot_eri_dm(self._eri, dm, hermi)

        return vj, vk

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        etot = self.energy_elec(dm, h1e, vhf)[0] + self.ewald_nuc()
        return etot.real

    def ewald_nuc(self, cell=None):
        if cell is None: cell = self.cell
        return ewald(cell, cell.ew_eta, cell.ew_cut, self.verbose)

    def get_bands(self, kpt_band, cell=None, dm=None, kpt=None):
        '''Get energy bands at a given (arbitrary) 'band' k-point.

        Returns:
            mo_energy : (nao,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nao) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        fock = self.get_hcore(kpt=kpt_band) \
                + self.get_veff(cell, dm, kpt=kpt, kpt_band=kpt_band)
        s1e = self.get_ovlp(kpt=kpt_band)
        mo_energy, mo_coeff = self.eig(fock, s1e)
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=True):
        if chk is None: chk = self.chkfile
        return init_guess_by_chkfile(self.cell, chk, project)
    def from_chk(self, chk=None, project=True):
        return self.init_guess_by_chkfile(chk, project)

