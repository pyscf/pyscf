import numpy as np
import scipy.linalg
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.scf.hf as pbchf
import pyscf.pbc.dft.rks as pbcrks
from pyscf.pbc import tools
from pyscf.pbc import gto as pbcgto

from pyscf.lib import logger
import pyscf.pbc.scf.scfint as scfint
pi = np.pi

def get_ovlp(mf, cell, kpts):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray                    
    '''
    nkpts = len(kpts)
    nao = cell.nao_nr()
    ovlp_kpts = np.zeros((nkpts,nao,nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            ovlp_kpts[k,:,:] = scfint.get_ovlp(cell, kpt)
        else:
            ovlp_kpts[k,:,:] = pbchf.get_ovlp(cell, kpt)
    return ovlp_kpts

def get_hcore(mf, cell, kpts):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    nao = cell.nao_nr()
    nkpts = len(kpts)
    hcore = np.zeros((nkpts, nao, nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            hcore[k,:,:] = scfint.get_hcore(cell, kpt)
        else:
            hcore[k,:,:] = pbchf.get_hcore(cell, kpt)
    return hcore

def get_j(mf, cell, dm_kpts, kpts):
    '''Get the Coulomb (J) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Returns: 
        vj : (nkpts, nao, nao) ndarray

    Note: This changes the mf object (mf._ecoul)
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngs = len(coords)
    nao = cell.nao_nr()

    coulG = tools.get_coulG(cell)

    aoR_kpts = np.zeros((nkpts, ngs, nao), np.complex128)
    rhoR = np.zeros(ngs)
    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
        rhoR += 1./nkpts*pyscf.pbc.dft.numint.eval_rho(cell, aoR_kpts[k,:,:], dm_kpts[k,:,:])
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)

    # TODO: REPLACE by eval_mat here (with non0tab)
    vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
    ecoul = 0.
    for k in range(nkpts):
        vj_kpts[k,:,:] = cell.vol/ngs * np.dot(aoR_kpts[k,:,:].T.conj(),
                                               vR.reshape(-1,1)*aoR_kpts[k,:,:])
        # TODO: energy is normally evaluated in dft.rks.get_veff
        ecoul += 1./nkpts * 0.5 * np.einsum('ij,ji', dm_kpts[k,:,:], vj_kpts[k,:,:])
    if abs(ecoul.imag > 1.e-12):
        raise RuntimeError("Coulomb energy has imaginary part, " 
                           "something is wrong!", ecoul.imag)
    mf._ecoul = ecoul.real
    return vj_kpts

def get_fock_(mf, h1e_kpts, s1e_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''Get the Fock matrices at sampled k-points.

    This is a k-point version of pyscf.scf.hf.get_fock_

    Returns:
       fock : (nkpts, nao, nao) ndarray
    '''
    fock = np.zeros_like(h1e_kpts)
    nkpts = len(mf.kpts)
    fock = pbchf.RHF.get_fock_(mf, h1e_kpts, s1e_kpts,
                               vhf_kpts, dm_kpts,
                               cycle, adiis, diis_start_cycle, 
                               level_shift_factor, damp_factor)
    return fock


class KRHF(pbchf.RHF):
    '''RHF class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points, 
    e.g. mo_coeff is (nkpts, nao, nao) ndarray
    '''
    def __init__(self, cell, kpts):
        pbchf.RHF.__init__(self, cell,kpts)
        self.kpts = kpts
        self.mo_occ = []
        self.mo_coeff_kpts = []

        if cell.ke_cutoff is not None:
            raise RuntimeError("ke_cutoff not supported with K pts yet")

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell

        if key.lower() == '1e':
            return self.init_guess_by_1e(cell)
        else:
            dm = pyscf.scf.hf.get_init_guess(cell, key)
            nao = cell.nao_nr()
            nkpts = len(self.kpts)
            dm_kpts = np.zeros((nkpts,nao,nao))

            # Use the molecular "unit cell" dm for each k-point
            for k in range(nkpts):
                dm_kpts[k,:,:] = dm

        return dm_kpts

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_hcore(self, cell, kpts)

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_ovlp(self, cell, kpts)

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        return get_j(self, cell, dm_kpts, kpts)

    def get_fock_(self, h1e_kpts, s1e, vhf, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=None, level_shift_factor=None, damp_factor=None):

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift_factor
        if damp_factor is None:
            damp_factor = self.damp_factor

        return get_fock_(self, h1e_kpts, s1e, vhf, dm_kpts, cycle, adiis,
                         diis_start_cycle, level_shift_factor, damp_factor)

    def get_veff(self, cell=None, dm=None, dm_last=None,
                 vhf_last=0, hermi=1):
        '''
        Args:
            cell : instance of :class:`Cell`
            dm : (nkpts, nao, nao) ndarray
                Density matrix at each k-point
                Note: Because get_veff is often called in pyscf.scf.hf with
                kwargs, we have to use the same argument name, 'dm', as in
                pyscf.scf.hf. 
            dm_last : (nkpts, nao, nao) ndarray
                Previous density matrix at each k-point
            vhf_last: (nkpts, nao, nao) ndarray
                Previous vhf at each k-point
        '''
        raise NotImplementedError

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts = len(self.kpts)

        # make this closer to the non-kpt one
        grad_kpts = np.empty(0,)

        for k in range(nkpts):
            grad = pyscf.scf.hf.RHF.get_grad(self, 
                        mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:], fock[k,:,:])
            grad_kpts = np.hstack((grad_kpts, grad))
        return grad_kpts

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        nao = h_kpts.shape[1]
        eig_kpts = np.zeros((nkpts,nao))
        mo_coeff_kpts = np.zeros_like(h_kpts)

        # TODO: should use superclass eig fn here?
        for k in range(nkpts):
            eig_kpts[k,:], mo_coeff_kpts[k,:,:] = pyscf.scf.hf.eig(h_kpts[k,:,:], s_kpts[k,:,:])
        return eig_kpts, mo_coeff_kpts

    def get_occ(self, mo_energy_kpts, mo_coeff_kpts):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy_kpts
        mo_occ_kpts = np.zeros_like(mo_energy_kpts)

        nkpts, nao = mo_coeff_kpts.shape[:2]
        nocc = (self.cell.nelectron * nkpts) // 2

        # Sort eigs in each kpt
        mo_energy = np.reshape(mo_energy_kpts, [nkpts*nao])
        # TODO: store mo_coeff correctly (for later analysis)
        #self.mo_coeff = np.reshape(mo_coeff_kpts, [nao, nao*nkpts])
        mo_idx = np.argsort(mo_energy)
        mo_energy = mo_energy[mo_idx]
        for ix in mo_idx[:nocc]:
            k, ikx = divmod(ix, nao)
            # TODO: implement Fermi smearing
            mo_occ_kpts[k, ikx] = 2

        if nocc < mo_energy.size:
            logger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, '!! HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'HOMO = %.12g', mo_energy[nocc-1])
        if self.verbose >= logger.DEBUG:
            np.set_printoptions(threshold=len(mo_energy))
            logger.debug(self, '  mo_energy = %s', mo_energy)
            np.set_printoptions()

        self.mo_energy = mo_energy_kpts
        self.mo_occ = mo_occ_kpts

        return mo_occ_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        '''One particle density matrix at each k-point.

        Returns:
            dm_kpts : (nkpts, nao, nao) ndarray
        '''
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff 
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ 

        nkpts = len(mo_occ_kpts)
        dm_kpts = np.zeros_like(mo_coeff_kpts)
        for k in range(nkpts):
            dm_kpts[k,:,:] = pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:]).T.conj()
        return dm_kpts

    def energy_elec(self,dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        raise NotImplementedError

    def get_band_fock_ovlp(self, fock, ovlp, band_kpt):
        '''Reconstruct Fock operator at a given 'band' k-point, not necessarily 
        in list of k-points.

        Returns:
            fock : (nao, nao) ndarray
            ovlp : (nao, nao) ndarray
        '''
        # To implement this analogously to
        # get_band_fock_ovlp for a single cell, one needs
        # the ovlp between basis functions at the kpts used in
        # the KSCF calculation, and the band_kpt at which one is 
        # evaluating, i.e. one needs 
        #      
        #      < p(k_pt) | q (band_kpt) >
        #
        # This involves two lattice sums.
        #
        # The other way, is to take the converged real-space
        # density on the grid from the KSCF calculation, and compute
        # hcore (band_kpt) and veff (band_kpt)
        # 
        # This can only reasonably be done for DFT, and using the 
        # real-space grid
        
        raise NotImplementedError


class KRKS(KRHF):
    def __init__(self, cell, kpts):
        KRHF.__init__(self, cell, kpts)
        self._numint = _KNumInt(kpts) # use periodic images of AO in
                                      # numerical integration
        self.xc = 'LDA,VWN'
        self._ecoul = 0
        self._exc = 0

    def dump_flags(self):
        KRHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        Args:
             See kscf.KRHF.get_veff

        Returns:
             vhf : (nkpts, nao, nao) ndarray
                Effective potential corresponding to input density matrix at
                each k-point
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()

        dm = np.array(dm, np.complex128) # e.g. if passed initial DM

        vhf = pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last,
                                      hermi)
        return vhf

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):

        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        nkpts = len(dm_kpts)
        e1 = 0.
        for k in range(nkpts):
            e1 += 1./nkpts*np.einsum('ij,ji', h1e_kpts[k,:,:], dm_kpts[k,:,:]).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul + self._exc


class _KNumInt(pyscf.dft.numint._NumInt):
    '''
    DFT KNumInt class

    Generalization of standard NumInt class for multiple k-points and 
    periodic images.
    '''
    def __init__(self, kpts=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpts = kpts

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        '''
        Returns:
            ao_kpts: (nkpts, ngs, nao) ndarray 
                AO values at each k-point
        '''
        nkpts = len(self.kpts)
        ngs = len(coords)
        nao = mol.nao_nr()

        ao_kpts = np.empty([nkpts, ngs, nao],np.complex128)
        for k in range(nkpts):
            kpt = self.kpts[k,:]
            ao_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(mol, coords, kpt, isgga,
                                  relativity, bastart, bascount,
                                  non0tab, verbose)
        return ao_kpts

    def eval_rho(self, mol, ao_kpts, dm_kpts, non0tab=None,
             isgga=False, verbose=None):
        '''
        Args:
            mol : Mole or Cell object
            ao_kpts : (nkpts, ngs, nao) ndarray
                AO values at each k-point
            dm_kpts: (nkpts, nao, nao) ndarray
                Density matrix at each k-point

        Returns:
           rhoR : (ngs,) ndarray
        '''
        nkpts, ngs, nao = ao_kpts.shape
        rhoR = np.zeros(ngs)
        for k in range(nkpts):
            rhoR += 1./nkpts*pyscf.pbc.dft.numint.eval_rho(mol, ao_kpts[k,:,:], dm_kpts[k,:,:])
        return rhoR

    def eval_rho2(self, mol, ao, dm, non0tab=None, isgga=False,
                  verbose=None):
        raise NotImplementedError

    def nr_rks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        '''
        Use slow function in numint, which only calls eval_rho, eval_mat.
        Faster function uses eval_rho2 which is not yet implemented.
        '''
        # TODO: fix spin, relativity
        spin=0; relativity=0
        return pyscf.dft.numint.nr_rks_vxc(self, mol, grids, x_id, c_id, dms,
                                           spin, relativity, hermi,
                                           max_memory, verbose)

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        raise NotImplementedError

    def eval_mat(self, mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                 isgga=False, verbose=None):
        # use local function for complex eval_mat
        nkpts = len(self.kpts)
        nao = ao.shape[2]

        mat = np.zeros((nkpts, nao, nao), dtype=ao.dtype)
        for k in range(nkpts):
            mat[k,:,:] = pyscf.pbc.dft.numint.eval_mat(mol, ao[k,:,:], weight,
                                    rho, vrho, vsigma, non0tab,
                                    isgga, verbose)
        return mat

#test_kscf_kgamma()
