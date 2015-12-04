'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import time
import numpy as np
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.scf.hf as pbchf
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.scf import scfint


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


def get_j(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngs = len(coords)
    nao = cell.nao_nr()

    aoR_kpts = np.zeros((nkpts, ngs, nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    vjR = get_vjR_(cell, dm_kpts, aoR_kpts)
    if kpt_band is not None:
        aoR_kband = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt_band)
        vj_kpts = cell.vol/ngs * np.dot(aoR_kband.T.conj(),
                                        vjR.reshape(-1,1)*aoR_kband)
    else:
        vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k in range(nkpts):
            vj_kpts[k,:,:] = cell.vol/ngs * np.dot(aoR_kpts[k,:,:].T.conj(),
                                                   vjR.reshape(-1,1)*aoR_kpts[k,:,:])

    return vj_kpts


def get_jk(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray

    Note: This changes the mf object (mf._ecoul)
    '''
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = len(kpts)
    ngs = len(coords)
    nao = cell.nao_nr()

    aoR_kpts = np.zeros((nkpts, ngs, nao), np.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)

    vjR = get_vjR_(cell, dm_kpts, aoR_kpts)
    if kpt_band is not None:
        aoR_kband = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt_band)
        vj_kpts = cell.vol/ngs * np.dot(aoR_kband.T.conj(),
                                        vjR.reshape(-1,1)*aoR_kband)
        vk_kpts = np.zeros((nao,nao), np.complex128)
        for k2 in range(nkpts):
            kpt2 = kpts[k2,:]
            vkR_k1k2 = pbchf.get_vkR_(cell, aoR_kband, aoR_kpts[k2,:,:], kpt_band, kpt2)
            # TODO: Break up the einsum
            vk_kpts += 1./nkpts * (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq', 
                        dm_kpts[k2,:,:], aoR_kband.conj(), 
                        vkR_k1k2, aoR_kpts[k2,:,:])
    else:
        vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k in range(nkpts):
            vj_kpts[k,:,:] = cell.vol/ngs * np.dot(aoR_kpts[k,:,:].T.conj(),
                                                   vjR.reshape(-1,1)*aoR_kpts[k,:,:])
        vk_kpts = np.zeros((nkpts,nao,nao), np.complex128)
        for k1 in range(nkpts):
            kpt1 = kpts[k1,:]
            for k2 in range(nkpts):
                kpt2 = kpts[k2,:]
                # TODO: Break up the einsum
                vkR_k1k2 = pbchf.get_vkR_(cell, aoR_kpts[k1,:,:], aoR_kpts[k2,:,:], kpt1, kpt2)
                vk_kpts[k1,:,:] += 1./nkpts * (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq', 
                                    dm_kpts[k2,:,:], aoR_kpts[k1,:,:].conj(), 
                                    vkR_k1k2, aoR_kpts[k2,:,:])

    # The BAD way to do vj (but like vk) -- gives identical results.
    #vj_kpts = np.zeros((nkpts,nao,nao), np.complex128)
    #for k1 in range(nkpts):
    #    kpt1 = kpts[k1,:]
    #    for k2 in range(nkpts):
    #        kpt2 = kpts[k2,:]
    #        vjR_k2k2 = pbchf.get_vkR_(cell, aoR_kpts[k2,:,:], aoR_kpts[k2,:,:], kpt2, kpt2)
    #        vj_kpts[k1,:,:] += 1./nkpts * (cell.vol/ngs) * np.einsum('rs,Rp,Rqs,Rr->pq', 
    #                            dm_kpts[k2,:,:], aoR_kpts[k1,:,:].conj(), 
    #                            vjR_k2k2, aoR_kpts[k1,:,:])

    return vj_kpts, vk_kpts


def get_vjR_(cell, dm_kpts, aoR_kpts):
    '''Get the real-space Hartree potential of the k-point sampled density matrix.

    Returns:
        vR : (ngs,) ndarray
            The real-space Hartree potential at every grid point.
    '''
    nkpts, ngs, nao = aoR_kpts.shape 
    coulG = tools.get_coulG(cell)

    rhoR = np.zeros(ngs)
    for k in range(nkpts):
        rhoR += 1./nkpts*pyscf.pbc.dft.numint.eval_rho(cell, aoR_kpts[k,:,:], dm_kpts[k,:,:])
    rhoG = tools.fft(rhoR, cell.gs)

    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)
    return vR


def get_fock_(mf, h1e_kpts, s1e_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''Get the Fock matrices at sampled k-points.

    This is a k-point version of pyscf.scf.hf.get_fock_

    Returns:
       fock : (nkpts, nao, nao) ndarray
    '''
    fock = np.zeros_like(h1e_kpts)
    # By inheritance, this is just pyscf.scf.hf.get_fock_
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

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    def __init__(self, cell, kpts):
        pbchf.RHF.__init__(self, cell, kpts)
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

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpt=None, kpt_band=None):
        # Must use 'kpt' kwarg
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpts
        kpts = kpt
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj = get_j(self, cell, dm_kpts, kpts, kpt_band)
        logger.timer(self, 'vj', *cpu0)
        return vj

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpt=None, kpt_band=None):
        # Must use 'kpt' kwarg
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpts
        kpts = kpt
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj, vk = get_jk(self, cell, dm_kpts, kpts, kpt_band)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk
        
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

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, 
                 kpts=None, kpt_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpts is None: kpts = self.kpts
        # TODO: Check incore, direct_scf, _eri's, etc
        vj, vk = self.get_jk(cell, dm, hermi, kpts, kpt_band)
        return vj - vk * .5

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
            dm_kpts[k,:,:] = pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], 
                                                    mo_occ_kpts[k,:]).T.conj()
        return dm_kpts

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        '''Following pyscf.scf.hf.energy_elec()
        '''
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if h1e_kpts is None: h1e_kpts = self.get_hcore()
        if vhf_kpts is None: vhf_kpts = self.get_veff(self.cell, dm_kpts)

        nkpts = len(dm_kpts)
        e1 = e_coul = 0.
        for k in range(nkpts):
            e1 += 1./nkpts * np.einsum('ij,ji', dm_kpts[k,:,:], h1e_kpts[k,:,:])
            e_coul += 1./nkpts * 0.5 * np.einsum('ij,ji', dm_kpts[k,:,:], vhf_kpts[k,:,:])
        if abs(e_coul.imag > 1.e-12):
            raise RuntimeError("Coulomb energy has imaginary part, "
                               "something is wrong!", e_coul.imag)
        e1 = e1.real
        e_coul = e_coul.real
        logger.debug(self, 'E_coul = %.15g', e_coul)
        return e1+e_coul, e_coul

    def get_bands(self, kpt_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at a given (arbitrary) 'band' k-point.

        Returns:
            mo_energy : (nao,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nao) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        fock = pbchf.get_hcore(cell, kpt_band) \
                + self.get_veff(kpts=kpts, kpt_band=kpt_band)
        s1e = pbchf.get_ovlp(cell, kpt_band)
        mo_energy, mo_coeff = pyscf.scf.hf.eig(fock, s1e)
        return mo_energy, mo_coeff 

