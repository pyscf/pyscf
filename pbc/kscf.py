import numpy
import scipy
import scipy.linalg
import pbc
import scf
import cell as cl
import pyscf
import pyscf.scf.hf

from pyscf.lib import logger

def get_hcore(mf, cell, kpts):
    '''
    K pt version of get_hcore

    kpts: ndarray [3, nkpts]

    Returns
        Core Hamiltonian at each kpt
        ndarray[nao, nao, nkpts]
    '''
    vne=scf.get_nuc(cell, mf.gs) 
    nao=vne.shape[0]
    nkpts=kpts.shape[0]
    hcore=numpy.zeros([nkpts, nao, nao])
    for k in range(nkpts):
        print kpts[k,:].shape
        # below reshape is a bit of a hack
        hcore[k,:,:]=vne+scf.get_t(cell, mf.gs, numpy.reshape(kpts[k,:], (3,1))) 
    return hcore

def get_fock_(mf, h1e_kpts, s1e, vhf, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):
    '''
    K pt version of scf.hf.get_fock_

    Main difference is that h1e_kpts is now dim [nao, nao, nkpts]
    and the returned fock is dim [nao, nao, nkpts]
    '''
    fock=numpy.zeros_like(h1e_kpts)
    nkpts=mf.kpts.shape[0]
    for k in range(nkpts):
        fock[k,:,:] = scf.RHF.get_fock_(mf, h1e_kpts[k,:,:], s1e, vhf, 
                                        dm_kpts[k,:,:], cycle=-1, adiis=None,
                                        diis_start_cycle=0, level_shift_factor=0, 
                                        damp_factor=0)
    return fock
    
class KRHF(scf.RHF):
    '''
    RHF class with k points
    
    Compared to molecular SCF, some members such as mo_coeff, mo_occ 
    now have an additional last dimension for the k pts, e.g.
    mo_coeff is dim [nao, nao, nkpts]
    '''
    def __init__(self, cell, gs, ew_eta, ew_cut, kpts):
        scf.RHF.__init__(self, cell, gs, ew_eta, ew_cut)
        self.kpts=kpts
        self.mo_occ=[]
        self.mo_coeff_kpts=[]

    # TODO arglist must use "mol" because of KW args used in kernel fn in hf.py
    def get_init_guess(self, mol=None, key='minao'):
        # fix this
        dm=pyscf.scf.hf.get_init_guess(mol, key)
        nao=dm.shape[0]
        nkpts=self.kpts.shape[0]
        dm_kpts=numpy.zeros([nkpts,nao,nao])
        
        # check normalization of FT
        for k in range(nkpts):
            dm_kpts[k,:,:]=1./nkpts*dm
        return dm_kpts

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell=self.cell
        if kpts is None: kpts=self.kpts
        return get_hcore(self, cell, kpts)
 
    # this function should look the same as scf.hf.SCF.get_fock_
    def get_fock_(self, h1e_kpts, s1e, vhf, dm_kpts, cycle=-1, adiis=None,
              diis_start_cycle=0, level_shift_factor=0, damp_factor=0):

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
        if dm_last is None:
            dm_last=numpy.zeros_like(dm)
        dm=numpy.einsum('kij->ij',dm)
        dm_last=numpy.einsum('kij->ij',dm_last)

        return pyscf.scf.hf.RHF.get_veff(self, cell, dm, dm_last, vhf_last, hermi)

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

    def eig(self, h_kpts, s):
        nkpts=h_kpts.shape[0]
        nao=h_kpts.shape[1]
        eig_kpts=numpy.zeros([nkpts,nao])
        mo_coeff_kpts=numpy.zeros_like(h_kpts)

        # TODO: should use superclass eig fn here?
        for k in range(nkpts):             
            eig_kpts[k,:], mo_coeff_kpts[k,:,:]=pyscf.scf.hf.eig(h_kpts[k,:,:], s)
        return eig_kpts, mo_coeff_kpts

    def get_occ(self, mo_energy_kpts, mo_coeff_kpts):
        '''
        K pt version of scf.hf.SCF.get_occ.
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy_kpts
        mo_occ_kpts = numpy.zeros_like(mo_energy_kpts)

        nocc = self.cell.nelectron // 2
        # have to sort eigs in each kpt
        # TODO: implement Fermi smearing
        nkpts=mo_coeff_kpts.shape[0]
        nao=mo_coeff_kpts.shape[1]

        mo_energy=numpy.reshape(mo_energy_kpts, [nkpts*nao])
        #mo_coeff=numpy.reshape(mo_coeff_kpts, [nao, nao*nkpts])

        mo_idx=numpy.argsort(mo_energy)
        for ix in mo_idx[:nocc]:
            k, ikx=divmod(ix, nkpts)
            mo_occ_kpts[ikx, k] = 2

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
        return mo_occ_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        '''
        One-particle DM

        TODO: check complex conjugation in scf.hf
        '''
        if mo_coeff_kpts is None:
            mo_coeff_kpts=self.mo_coeff_kpts
        if mo_occ_kpts is None:
            mo_occ_kpts=self.mo_occ_kpts

        nkpts=mo_occ_kpts.shape[0]
        # dm=numpy.zeros_like(mo_coeff_kpts[:,:,0])

        # for k in range(nkpts):
        #     # should this call pbc.scf version?
        #     dm+=pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:])
        dm_kpts=numpy.zeros_like(mo_coeff_kpts)
        for k in range(nkpts):
            dm_kpts[k,:,:]=pyscf.scf.hf.make_rdm1(mo_coeff_kpts[k,:,:], mo_occ_kpts[k,:])
        return dm_kpts

    def energy_tot(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if dm_kpts is None: dm_kpts=self.make_rdm1()
        energy_elec=0.
        nkpts=self.kpts.shape[0]
        for k in range(nkpts):
            energy_elec+=pyscf.scf.hf.energy_elec(self, dm_kpts[k,:,:], h1e_kpts[k,:,:], vhf)[0]
        return energy_elec + self.ewald_nuc()

def test_kscf_sc():            
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    Lx=2
    L=60
    h=numpy.eye(3.)*L
    mol.atom.extend([['He', (0.5,L/2.,L/2.)], ])
    mol.atom.extend([['He', (1.5,L/2.,L/2.)], ])

    mol.basis = { 'He': [[0,(1.0, 1.0)]] }
    mol.build()

    m=pyscf.scf.hf.RHF(mol)
    print (m.scf())

    cell=cl.Cell()
    cell.__dict__=mol.__dict__
    cell.h=h
    cell.vol=scipy.linalg.det(cell.h)
    cell.output=None
    cell.verbose=7
    cell.build()

def test_kscf_kpts():

    gs=numpy.array([80,80,80])
    ew_eta=0.05
    ew_cut=(40,40,40)
    kpts=numpy.vstack(([0,0,0],
                       [0,0,1]))
    print kpts.shape
    #mf=scf.RHF(cell, gs, ew_eta, ew_cut)
    kmf=KRHF(cell, gs, ew_eta, ew_cut, kpts)


def test_kscf():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = None

    L=60
    h=numpy.eye(3.)*L

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
    cell.output=None
    cell.verbose=7
    cell.build()
    
    gs=numpy.array([80,80,80])
    ew_eta=0.05
    ew_cut=(40,40,40)
    kpts=numpy.vstack(([0,0,0],
                       [0,0,1]))
    print kpts.shape
    #mf=scf.RHF(cell, gs, ew_eta, ew_cut)
    kmf=KRHF(cell, gs, ew_eta, ew_cut, kpts)
    #print (mf.scf()) # -2.58766850182551: doesn't look good, but this is due
                     # to interaction of the exchange hole with its periodic
                     # image, which can only be removed with *very* large boxes.
    print (kmf.scf())

    # Now try molecular type integrals for the exchange operator, 
    # and periodic integrals for Coulomb. This effectively
    # truncates the exchange operator. 
    kmf.mol_ex=True 
    print (kmf.scf()) # -2.63493445685: much better!
