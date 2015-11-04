import numpy
import scipy.linalg
import pyscf.dft
import pyscf.pbc.dft
import pyscf.pbc.scf.hf as pbchf
import pyscf.pbc.dft.rks as pbcrks
from pyscf.pbc import tools
from pyscf.pbc import gto as pbcgto

from pyscf.lib import logger
import pyscf.pbc.scf.scfint as scfint
pi = numpy.pi

def get_ovlp(mf, cell, kpts):
    '''Get overlap AO matrices at sampled k-points.

    Args:
        mf : KRHF or KRKS object
        cell : Cell object
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray                    
    '''
    nkpts = kpts.shape[0]
    nao = cell.nao_nr()
    ovlp_kpts = numpy.zeros([nkpts,nao,nao], numpy.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            ovlp_kpts[k,:,:] = scfint.get_ovlp(cell, kpt)
        else:
            ovlp_kpts[k,:,:] = pbchf.get_ovlp(cell, kpt)
    return ovlp_kpts

def get_j(mf, cell, dm_kpts, kpts):
    '''Get Coulomb (J) AO matrices at sampled k-points.

    Args:
        mf : KRHF or KRKS object
        cell : Cell object
        dm_kpts : (nkpts, nao, nao) ndarray
                  DM at each kpt
        kpts : (nkpts, 3) ndarray

    Returns: 
        vj : (nkpts, nao, nao) ndarray
        Coulomb matrix

    Note: *changes mf object* (mf._ecoul)
    '''
    coulG = tools.get_coulG(cell)
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts = kpts.shape[0]

    ngs = coords.shape[0]
    nao = cell.nao_nr()
    aoR_kpts = numpy.zeros((nkpts, ngs, nao),numpy.complex128)
    rhoR = numpy.zeros([ngs])

    for k in range(nkpts):
        kpt = kpts[k,:]
        aoR_kpts[k,:,:] = pyscf.pbc.dft.numint.eval_ao(cell, coords, kpt)
        rhoR += 1./nkpts*pyscf.pbc.dft.numint.eval_rho(cell, aoR_kpts[k,:,:], dm_kpts[k,:,:])

    rhoG = tools.fft(rhoR, cell.gs)
    vG = coulG*rhoG
    vR = tools.ifft(vG, cell.gs)

    #:vj = numpy.zeros([nao,nao])
    #:for i in range(nao):
    #:    for j in range(nao):
    #:        vj[i,j] = cell.vol/ngs*numpy.dot(aoR[:,i],vR*aoR[:,j])
    # TODO: REPLACE by eval_mat here (with non0tab)
    vj_kpts = numpy.zeros([nkpts,nao,nao],numpy.complex128)

    mf._ecoul = 0.
    for k in range(nkpts):
        vj_kpts[k,:,:] = cell.vol/ngs * numpy.dot(aoR_kpts[k,:,:].T.conj(),
                                                  vR.reshape(-1,1)*aoR_kpts[k,:,:])

        mf._ecoul += 1./nkpts*numpy.einsum('ij,ji', dm_kpts[k,:,:], vj_kpts[k,:,:]) * .5
    # TODO: energy is normally evaluated in dft.rks.get_veff
    if abs(mf._ecoul.imag > 1.e-12):
        raise RuntimeError("Coulomb energy has imaginary part, sth is wrong!", mf._ecoul.imag)

    mf._ecoul = mf._ecoul.real

    return vj_kpts

def get_hcore(mf, cell, kpts):
    '''
    K pt version of get_hcore

    Args:
        mf : KRKS or KRHF object
        cell: Cell object
        kpts: (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
                Core Hamiltonian at each kpt
    '''
    nao = cell.nao_nr()
    nkpts = kpts.shape[0]
    hcore = numpy.zeros([nkpts, nao, nao],numpy.complex128)
    for k in range(nkpts):
        kpt = kpts[k,:]
        if mf.analytic_int:
            hcore[k,:,:] = scfint.get_hcore(cell, kpt)
        else:
            hcore[k,:,:] = pbchf.get_hcore(cell, kpt)
    return hcore

def get_fock_(mf, h1e_kpts, s1e_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''
    K pt version of pyscf.scf.hf.get_fock_

    Returns:
       fock : (nkpts, nao, nao) ndarray
    '''
    fock = numpy.zeros_like(h1e_kpts)
    nkpts = mf.kpts.shape[0]
    fock = pbchf.RHF.get_fock_(mf, h1e_kpts, s1e_kpts,
                                      vhf_kpts, dm_kpts,
                                      cycle, adiis, diis_start_cycle, 
                                      level_shift_factor, damp_factor)
    return fock


class KRHF(pbchf.RHF):
    '''
    RHF class with k points

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k pts, e.g.
    mo_coeff is dim [nkpts, nao, nao]
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
            nao = dm.shape[0]
            nkpts = self.kpts.shape[0]
            dm_kpts = numpy.zeros([nkpts,nao,nao])

            for k in range(nkpts):
                dm_kpts[k,:,:] = dm

        return dm_kpts

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        return get_ovlp(self, cell, kpts)

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None):
        if cell is None: cell=self.cell
        if kpts is None: kpts=self.kpts
        if dm_kpts is None: dm_kpts=self.make_rdm1()
        return get_j(self, cell, dm_kpts, kpts)

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell=self.cell
        if kpts is None: kpts=self.kpts
        return get_hcore(self, cell, kpts)

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
            cell : Cell object

            dm : (nkpts, nao, nao) ndarray
                dm *at each k-point*
                *Note* because get_veff is often called in pyscf.scf.hf
                with kwargs, we have to use the same argument name as in
                pyscf.scf.hf. 

            dm_last : (nkpts, nao, nao) ndarray
                previous dm *at each k-point*

            vhf_last: (nkpts, nao, nao) ndarray
                previous vhf *at each k-point*
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

        nkpts = self.kpts.shape[0]

        # make this closer to the non-kpt one
        grad_kpts = numpy.empty(0,)

        for k in range(nkpts):
            grad = pyscf.scf.hf.RHF.get_grad(self, 
                        mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:], fock[k,:,:])
            grad_kpts = numpy.hstack((grad_kpts, grad))
        return grad_kpts

    def eig(self, h_kpts, s_kpts):
        nkpts = h_kpts.shape[0]
        nao = h_kpts.shape[1]
        eig_kpts = numpy.zeros([nkpts,nao])
        mo_coeff_kpts = numpy.zeros_like(h_kpts)

        # TODO: should use superclass eig fn here?
        for k in range(nkpts):
            eig_kpts[k,:], mo_coeff_kpts[k,:,:] = pyscf.scf.hf.eig(h_kpts[k,:,:], s_kpts[k,:,:])
        return eig_kpts, mo_coeff_kpts

    def get_occ(self, mo_energy_kpts, mo_coeff_kpts):
        '''
        K pt version of scf.hf.SCF.get_occ.
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy_kpts
        mo_occ_kpts = numpy.zeros_like(mo_energy_kpts)

        nkpts = mo_coeff_kpts.shape[0]
        nocc = (self.cell.nelectron * nkpts) // 2

        # have to sort eigs in each kpt
        # TODO: implement Fermi smearing

        nao = mo_coeff_kpts.shape[1]

        mo_energy = numpy.reshape(mo_energy_kpts, [nkpts*nao])
        # TODO: store mo_coeff correctly (for later analysis)
        #self.mo_coeff = numpy.reshape(mo_coeff_kpts, [nao, nao*nkpts])


        mo_idx = numpy.argsort(mo_energy)
        mo_energy = mo_energy[mo_idx]
        for ix in mo_idx[:nocc]:
            k, ikx = divmod(ix, nao)
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
            numpy.set_printoptions(threshold=len(mo_energy))
            logger.debug(self, '  mo_energy = %s', mo_energy)
            numpy.set_printoptions()

        self.mo_energy = mo_energy_kpts
        self.mo_occ = mo_occ_kpts

        return mo_occ_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        '''
        DM at each kpt.

        Returns:
            dm_kpts : (nkpts, nao, nao) ndarray
        '''
        if mo_coeff_kpts is None:
            mo_coeff_kpts = self.mo_coeff # Note: this is actually "self.mo_coeff_kpts"
                                        # which is stored in self.mo_coeff of the scf.hf.RHF superclass
        if mo_occ_kpts is None:
            mo_occ_kpts = self.mo_occ # Note: this is actually "self.mo_occ_kpts"
                                    # which is stored in self.mo_occ of the scf.hf.RHF superclass

        nkpts = mo_occ_kpts.shape[0]
        # dm = numpy.zeros_like(mo_coeff_kpts[:,:,0])

        # for k in range(nkpts):
        #     # should this call pbc.scf version?
        #     dm += pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:])
        dm_kpts = numpy.zeros_like(mo_coeff_kpts)

        for k in range(nkpts):
            dm_kpts[k,:,:] = pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:]).T.conj()
        return dm_kpts

    def energy_elec(self,dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        raise NotImplementedError

    def get_band_fock_ovlp(self, fock, ovlp, band_kpt):
        '''Reconstruct Fock operator at a given band kpt 
           (not necessarily in list of k pts)

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

             Effective potential corresponding to input DM at each kpt.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()

        dm=numpy.array(dm, numpy.complex128) # e.g. if passed initial DM

        vhf = pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last,
                                      hermi)
        return vhf

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):

        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        nkpts = dm_kpts.shape[0]
        e1 = 0.
        for k in range(nkpts):
            e1 += 1./nkpts*numpy.einsum('ij,ji', h1e_kpts[k,:,:], dm_kpts[k,:,:]).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul + self._exc


class _KNumInt(pyscf.dft.numint._NumInt):
    '''
    DFT KNumInt class: 

    Generalization of standard NumInt class for a multiple k-pts, 
    and periodic images.
    '''
    def __init__(self, kpts=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpts = kpts

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        '''
        Returns:
            ao_kpts: (k, N, nao) ndarray 
                     AO values at each k pt
        '''
        nkpts = self.kpts.shape[0]
        ngs = coords.shape[0]
        nao = mol.nao_nr()

        ao_kpts = numpy.empty([nkpts, ngs, nao],numpy.complex128)
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
           ao_kpts : (k, N, nao) ndarray
                     AO values at each kpoint
           dm_kpts: (k, nao, nao) ndarray
                     DM at each kpt

        Returns:
           rhoR : (N, ) ndarray
        '''
        nkpts = self.kpts.shape[0]
        ngs = ao_kpts.shape[1]
        rhoR = numpy.zeros([ngs])
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
        nkpts = self.kpts.shape[0]
        nao = ao.shape[2]

        mat = numpy.zeros([nkpts, nao, nao], dtype=ao.dtype)
        for k in range(nkpts):
            mat[k,:,:] = pyscf.pbc.dft.numint.eval_mat(mol, ao[k,:,:], weight,
                                    rho, vrho, vsigma, non0tab,
                                    isgga, verbose)
        return mat



#test_kscf_kgamma()
