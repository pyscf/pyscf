import numpy
import scipy
import pbc.dft.rks
import scipy.linalg
import pyscf.pbc as pbc
import pbc.dft
import pbc.dft.numint
import pbc.dft.gen_grid
import pbc.tools.pbc as helpers
import pyscf.pbc.scf
from pyscf.pbc import gto as pbcgto
import pyscf.pbc.gto.cell as cl
import pyscf.pbc.gto
import pbc
import pyscf
import pyscf.scf.hf


from pyscf.lib import logger
pi=numpy.pi

def get_ovlp(mf, cell, kpts):
    '''Get overlap AO matrices at sampled k-points.
    '''
    nkpts=kpts.shape[0]
    nao=cell.nao_nr()
    ovlp_kpts=numpy.zeros([nkpts,nao,nao], numpy.complex128)
    for k in range(nkpts):
        kpt=numpy.reshape(kpts[k,:], (3,1))
        ovlp_kpts[k,:,:]=pbc.scf.hf.get_ovlp(cell, kpt)

    return ovlp_kpts


def get_j(mf, cell, dm_kpts, kpts):
    '''Get Coulomb (J) AO matrices at sampled k-points.

    kpts: ndarray [nkpts, 3]

    TODO: Note - changes mf object (mf._ecoul)
    '''
    gs=cell.gs
    coulG=helpers.get_coulG(cell)
    coords=pbc.dft.gen_grid.gen_uniform_grids(cell)
    nkpts=kpts.shape[0]

    ngs=coords.shape[0]
    nao=cell.nao_nr()
    aoR_kpts=numpy.zeros((nkpts, ngs, nao),numpy.complex128)
    rhoR=numpy.zeros([ngs])

    for k in range(nkpts):
        kpt=numpy.reshape(kpts[k,:], (3,1))
        aoR_kpts[k,:,:]=pbc.dft.numint.eval_ao(cell, coords, kpt)
        rhoR+=1./nkpts*pbc.dft.numint.eval_rho(cell, aoR_kpts[k,:,:], dm_kpts[k,:,:])

    rhoG=helpers.fft(rhoR, gs)
    vG=coulG*rhoG
    vR=helpers.ifft(vG, gs)

    #:vj=numpy.zeros([nao,nao])
    #:for i in range(nao):
    #:    for j in range(nao):
    #:        vj[i,j]=cell.vol()/ngs*numpy.dot(aoR[:,i],vR*aoR[:,j])
    # TODO: REPLACE by eval_mat here (with non0tab)
    vj_kpts=numpy.zeros([nkpts,nao,nao],numpy.complex128)

    mf._ecoul=0.
    for k in range(nkpts):
        vj_kpts[k,:,:] = cell.vol()/ngs * numpy.dot(aoR_kpts[k,:,:].T.conj(),
                                                  vR.reshape(-1,1)*aoR_kpts[k,:,:])

        mf._ecoul+=1./nkpts*numpy.einsum('ij,ji', dm_kpts[k,:,:], vj_kpts[k,:,:]) * .5
    # TODO: energy is normally evaluated in dft.rks.get_veff
    mf._ecoul=mf._ecoul.real
    return vj_kpts

def get_hcore(mf, cell, kpts):
    '''
    K pt version of get_hcore

    kpts: ndarray [3, nkpts]

    Returns
        Core Hamiltonian at each kpt
        ndarray[nao, nao, nkpts]
    '''
    vne=pbc.scf.hf.get_nuc(cell)
    nao=vne.shape[0]
    nkpts=kpts.shape[0]
    hcore=numpy.zeros([nkpts, nao, nao],numpy.complex128)
    for k in range(nkpts):
        # below reshape is a bit of a hack
        kpt=numpy.reshape(kpts[k,:], (3,1))
        hcore[k,:,:]=(pbc.scf.hf.get_nuc(cell, kpt)+
                      pbc.scf.hf.get_t(cell, kpt))
    return hcore

def get_fock_(mf, h1e_kpts, s1e_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''
    K pt version of scf.hf.get_fock_

    Main difference is that h1e_kpts is now dim [nao, nao, nkpts]
    and the returned fock is dim [nao, nao, nkpts]
    '''
    fock=numpy.zeros_like(h1e_kpts)
    nkpts=mf.kpts.shape[0]
    for k in range(nkpts):
        fock[k,:,:] = pbc.scf.hf.RHF.get_fock_(mf, h1e_kpts[k,:,:], s1e_kpts[k,:,:],
                                        vhf_kpts[k,:,:], dm_kpts[k,:,:],
                                        cycle=-1, adiis=None,
                                        diis_start_cycle=0, level_shift_factor=0,
                                        damp_factor=0)
    return fock


class KRHF(pbc.scf.hf.RHF):
    '''
    RHF class with k points

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional last dimension for the k pts, e.g.
    mo_coeff is dim [nao, nao, nkpts]
    '''
    def __init__(self, cell, kpts):
        pbc.scf.hf.RHF.__init__(self, cell,kpts)
        self.kpts=kpts
        self.mo_occ=[]
        self.mo_coeff_kpts=[]

    # TODO: arglist must use "mol" because of KW args used in kernel fn in hf.py
    def get_init_guess(self, mol=None, key='minao'):
        dm=pyscf.scf.hf.get_init_guess(mol, key)
        nao=dm.shape[0]
        nkpts=self.kpts.shape[0]
        dm_kpts=numpy.zeros([nkpts,nao,nao])

        for k in range(nkpts):
            dm_kpts[k,:,:]=dm

        return dm_kpts

    def get_ovlp(self, cell=None, kpts=None):
        if cell is None: cell=self.cell
        if kpts is None: kpts=self.kpts
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

    # this function should look the same as scf.hf.SCF.get_fock_
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
        Note: get_veff is often called with kwargs, we have to use here
         the same names as in pyscf.scf.hf. Note, however, that
         the input dm, dm_last should more accurately be dm_kpts, dm_last_kpts
        '''
        raise NotImplementedError

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1=self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock=self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts=self.kpts.shape[0]

        # make this closer to the non-kpt one
        grad_kpts=numpy.empty(0,)

        for k in range(nkpts):
            grad=pyscf.scf.hf.RHF.get_grad(self, mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:],
                                           fock[k,:,:])
            grad_kpts=numpy.hstack((grad_kpts, grad))
        return grad_kpts

    def eig(self, h_kpts, s_kpts):
        nkpts=h_kpts.shape[0]
        nao=h_kpts.shape[1]
        eig_kpts=numpy.zeros([nkpts,nao])
        mo_coeff_kpts=numpy.zeros_like(h_kpts)

        # TODO: should use superclass eig fn here?
        for k in range(nkpts):
            eig_kpts[k,:], mo_coeff_kpts[k,:,:]=pyscf.scf.hf.eig(h_kpts[k,:,:], s_kpts[k,:,:])
        return eig_kpts, mo_coeff_kpts

    def get_occ(self, mo_energy_kpts, mo_coeff_kpts):
        '''
        K pt version of scf.hf.SCF.get_occ.
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy_kpts
        mo_occ_kpts = numpy.zeros_like(mo_energy_kpts)

        nkpts=mo_coeff_kpts.shape[0]
        nocc = (self.cell.nelectron * nkpts) // 2
        # have to sort eigs in each kpt
        # TODO: implement Fermi smearing

        nao=mo_coeff_kpts.shape[1]

        mo_energy=numpy.reshape(mo_energy_kpts, [nkpts*nao])
        # TODO: store mo_coeff correctly (for later analysis)
        #self.mo_coeff=numpy.reshape(mo_coeff_kpts, [nao, nao*nkpts])

        mo_idx=numpy.argsort(mo_energy)
        print mo_energy_kpts
        for ix in mo_idx[:nocc]:
            k, ikx=divmod(ix, nao)
            mo_occ_kpts[k, ikx] = 2
            print "occupying", k, ikx, mo_energy_kpts[k,ikx]

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

        self.mo_energy=mo_energy_kpts
        self.mo_occ=mo_occ_kpts

        return mo_occ_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        '''
        One-particle DM

        TODO: check complex conjugation in scf.hf
        '''
        if mo_coeff_kpts is None:
            mo_coeff_kpts=self.mo_coeff # Note: this is actually "self.mo_coeff_kpts"
                                        # which is stored in self.mo_coeff of the scf.hf.RHF superclass
        if mo_occ_kpts is None:
            mo_occ_kpts=self.mo_occ # Note: this is actually "self.mo_occ_kpts"
                                    # which is stored in self.mo_occ of the scf.hf.RHF superclass
        print "MO OCC", self.mo_occ

        nkpts=mo_occ_kpts.shape[0]
        # dm=numpy.zeros_like(mo_coeff_kpts[:,:,0])

        # for k in range(nkpts):
        #     # should this call pbc.scf version?
        #     dm+=pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:])
        dm_kpts=numpy.zeros_like(mo_coeff_kpts)
        for k in range(nkpts):
            dm_kpts[k,:,:]=pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:])
        return dm_kpts

    def energy_elec(self,dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        raise NotImplementedError

class KRKS(KRHF):
    def __init__(self, cell, gs, ew_eta, ew_cut, kpts):
        KRHF.__init__(self, cell, kpts)
        self._numint = _KNumInt(kpts) # use periodic images of AO in
                                      # numerical integration
        self.xc = 'LDA,VWN'
        self._ecoul = 0
        self._exc = 0
        #self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        KRHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=0, hermi=1):
        '''
        Returns:
             effective potential corresponding
             to an input DM at each kpt.

        See also: kscf.RHF.get_veff
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()

        vhf= pyscf.dft.rks.get_veff_(self, cell, dm, dm_last, vhf_last,
                                       hermi)

        return vhf

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):

        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts=self.make_rdm1()

        nkpts=dm_kpts.shape[0]
        e1=0.
        for k in range(nkpts):
            e1+=1./nkpts*numpy.einsum('ij,ji', h1e_kpts[k,:,:], dm_kpts[k,:,:]).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul+self._exc


class _KNumInt(pyscf.dft.numint._NumInt):
    '''
    DFT KNumInt class: generalization for a multiple k-pts, and
    periodic images
    '''
    def __init__(self, kpts=None):
        pyscf.dft.numint._NumInt.__init__(self)
        self.kpts=kpts

    def eval_ao(self, mol, coords, isgga=False, relativity=0, bastart=0,
                bascount=None, non0tab=None, verbose=None):
        '''
        Returns shape (k, N, nao) (ao at each k point)
        '''
        nkpts=self.kpts.shape[0]
        ngs=coords.shape[0]
        nao=mol.nao_nr()

        ao_kpts=numpy.empty([nkpts, ngs, nao],numpy.complex128)
        for k in range(nkpts):
            kpt=numpy.reshape(self.kpts[k,:], (3,1))
            ao_kpts[k,:,:]=pbc.dft.numint.eval_ao(mol, coords, kpt, isgga,
                                  relativity, bastart, bascount,
                                  non0tab, verbose)
        return ao_kpts

    def eval_rho(self, mol, ao_kpts, dm_kpts, non0tab=None,
             isgga=False, verbose=None):
        '''
        Modified to take in (k, N, nao) ao vector, and dm at each kpt
        '''
        nkpts=self.kpts.shape[0]
        ngs=ao_kpts.shape[1]
        rhoR=numpy.zeros([ngs])
        for k in range(nkpts):
            rhoR+=1./nkpts*pbc.dft.numint.eval_rho(mol, ao_kpts[k,:,:], dm_kpts[k,:,:])
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
        return pyscf.dft.numint.nr_rks_vxc(self, mol, grids, x_id, c_id, dms,
                                           spin=0, relativity=0, hermi=1,
                                           max_memory=2000, verbose=None)

    def nr_uks(self, mol, grids, x_id, c_id, dms, hermi=1,
               max_memory=2000, verbose=None):
        raise NotImplementedError

    def eval_mat(self, mol, ao, weight, rho, vrho, vsigma=None, non0tab=None,
                 isgga=False, verbose=None):
        # use local function for complex eval_mat
        nkpts=self.kpts.shape[0]
        nao=ao.shape[2]

        mat=numpy.zeros([nkpts, nao, nao], numpy.complex128)
        for k in range(nkpts):
            mat[k,:,:]=pbc.dft.numint.eval_mat(mol, ao[k,:,:], weight,
                                    rho, vrho, vsigma=None, non0tab=None,
                                    isgga=False, verbose=None)
        return mat



def test_kscf_components():
    pass

    #t=scf.get_t(cell, gs)
    #dm=kmf.make_rdm1()
    # print t
    # print dm
    # print np.einsum('ij,ij',t,dm)
    # s=scf.get_ovlp(cell, gs)
    # print kmf.eig(t, s)[0]
    # print kmf.get_init_guess()
    # print "overlap eigs", scipy.linalg.eigh(s)
    t=[scf.get_t(cell, gs, kpt) for kpt in kpts]
    s=[scf.get_ovlp(cell, gs, kpt) for kpt in kpts]
    # print t
    # print s
    print t[0]/s[0], t[1]/s[1]


def test_kscf_gamma(atom, ncells):

    import numpy as np
    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR

    import warnings

    B=BOHR
    mol = gto.Mole()
    #mol.verbose = 7
    mol.output = None

    # As this is increased, we can see convergence between the molecule
    # and cell calculation
    Lunit=2
    Ly=Lz=2
    Lx=ncells*Lunit

    h=np.diag([Lx,Ly,Lz])

    # place atom in middle of big box
    for i in range(ncells):
        mol.atom.extend([[atom, ((.5+i)*Lunit*B,0.5*Ly*B,0.5*Lz*B)]])

    # these are some exponents which are
    # not hard to integrate
    mol.basis = { atom: [[0, (1.0, 1.0)]] }
    #mol.basis = { 'He': '6-31G' }
    mol.build()


    pseudo = None
    n = 40
    cell = pbcgto.Cell()
    cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = mol.unit
    cell.h = h
    cell.gs = [n*ncells,n,n]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    cell.h=h
    cell.nimgs = [2,2,2]
    cell.output=None
    cell.verbose=7
    cell.build()

    #warnings.simplefilter("error", np.ComplexWarning)

    kmf=pbc.dft.rks.RKS(cell)
    kmf.init_guess="atom"
    return kmf.scf()

def test_kscf_kpoints(atom, ncells):

    import numpy as np
    from pyscf import gto
    from pyscf.dft import rks
    from pyscf.lib.parameters import BOHR

    B=BOHR
    mol = gto.Mole()
    #mol.verbose = 7
    mol.output = None

    Lunit=2
    Ly=Lz=2
    Lx=Lunit


    h=np.diag([Lx,Ly,Lz])

    # place atom in middle of big box
    mol.atom.extend([[atom, (0.5*Lunit*B,0.5*Ly*B,0.5*Lz*B)]])

    # these are some exponents which are
    # not hard to integrate
    mol.basis = { atom: [[0, (1.0, 1.0)]] }
    #mol.basis = { 'He': '6-31G' }
    mol.build()

    # this is the PBC DFT calc!!
    pseudo = None
    n = 40
    cell = pbcgto.Cell()
    cell.output = '/dev/null'
    cell.verbose = 5
    cell.unit = mol.unit
    cell.h = h
    cell.gs = [n,n,n]
    cell.nimgs = [2,2,2]

    cell.atom = mol.atom
    cell.basis = mol.basis
    cell.pseudo = pseudo
    cell.build()
    #cell=cl.Cell()
    #cell.__dict__=mol.__dict__ # hacky way to make a cell
    #cell.h=h
    ##cell.vol=scipy.linalg.det(cell.h)
    ##cell.nimgs = [0,0,0]
    #cell.pseudo=None
    #cell.output=None
    #cell.verbose=7
    #cell.build()

    # points in grid (x,y,z)
    #gs=np.array([40,40,40])

    # Ewald parameters
    precision=1.e-9

    # make Monkhorst-Pack (equally spaced k points along x direction)
    invhT=scipy.linalg.inv(numpy.asarray(cell.h).T)
    kGvs=[]
    for i in range(ncells):
        kGvs.append(i*1./ncells*2*pi*np.dot(invhT,(1,0,0)))
    kpts=numpy.vstack(kGvs)

    kmf=KRKS(cell, cell.gs, cell.ew_eta, cell.ew_cut, kpts)
    kmf.init_guess="atom"
    return kmf.scf()

def test_kscf_kgamma():
    # tests equivalence of gamma supercell and kpoints calculations

    emf_gamma=[]
    emf_kpt=[]

    # TODO: currently works only if atom has an even #of electrons
    # *only* reason is that Mole checks this against spin,
    # all other parts of the code
    # works so long as cell * nelectron is even, e.g. 4 unit cells of H atoms
    for ncell in range(1,6):
        emf_gamma.append(test_kscf_gamma("He", ncell)/ncell)
        emf_kpt.append(test_kscf_kpoints("He", ncell))
        print "COMPARISON", emf_gamma[-1], emf_kpt[-1] # should be the same up to integration error

    # each entry should be the same up to integration error (abt 5 d.p.)
    print "ALL ENERGIES, GAMMA", emf_gamma
    print "ALL ENERGIES, KPT", emf_kpt

test_kscf_kgamma()
