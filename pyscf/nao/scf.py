from __future__ import print_function, division
import sys, numpy as np
from pyscf.nao.tddft_iter import tddft_iter
from pyscf.scf import hf, uhf
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u
from pyscf.nao.m_vhartree_coo import vhartree_coo
from timeit import default_timer as timer

#
#
#
class scf(tddft_iter):

  def __init__(self, **kw):
    """
    Constructor a self-consistent field
    """

    self.perform_scf = kw['perform_scf'] if 'perform_scf' in kw else False
    self.kmat_algo = kw['kmat_algo'] if 'kmat_algo' in kw else None
    self.kmat_timing = 0.0 if 'kmat_timing' in kw else None
    for x in ['xc_code', 'dealloc_hsx', 'dtype']: kw.pop(x,None)
    tddft_iter.__init__(self, dtype=np.float64, xc_code='RPA', dealloc_hsx=False, **kw)
    #print(__name__, ' dtype ', self.dtype)

    self.xc_code_kernel = copy(self.xc_code)
    self.xc_code = self.xc_code_mf
    self.dm_mf   = self.make_rdm1() # necessary to get_hcore(...) in case of pp starting point

    if self.gen_pb:
      self.hkernel_den = pack2den_u(self.kernel)

    if self.nspin==1:
      self.pyscf_scf = hf.SCF(self)
    else:
      self.pyscf_scf = uhf.UHF(self)
      
    self.pyscf_scf.direct_scf = False # overriding the attributes from hf.SCF ...
    self.pyscf_scf.get_hcore = self.get_hcore
    self.pyscf_scf.get_ovlp = self.get_ovlp
    self.pyscf_scf.get_j = self.get_j
    self.pyscf_scf.get_jk = self.get_jk
    self.pyscf_scf.energy_nuc = self.energy_nuc
    if self.perform_scf : self.kernel_scf(**kw)

  def kernel_scf(self, dump_chk=False, **kw):
    """ This does the actual SCF loop so far only HF """
    from pyscf.nao.m_fermi_energy import fermi_energy as comput_fermi_energy
    dm0 = self.get_init_guess()
    if (self.nspin==2 and dm0.ndim==5): dm0=dm0[0,...,0] 
    etot = self.pyscf_scf.kernel(dm0=dm0, dump_chk=dump_chk, **kw)
    #print(__name__, self.mo_energy.shape, self.pyscf_hf.mo_energy.shape)

    if self.nspin==1:
      self.mo_coeff[0,0,:,:,0] = self.pyscf_scf.mo_coeff.T
      self.mo_energy[0,0,:] = self.pyscf_scf.mo_energy
      self.ksn2e = self.mo_energy
      self.mo_occ[0,0,:] = self.pyscf_scf.mo_occ
    elif self.nspin==2:
      for s in range(self.nspin):
        self.mo_coeff[0,s,:,:,0] = self.pyscf_scf.mo_coeff[s].T
        self.mo_energy[0,s,:] = self.pyscf_scf.mo_energy[s]
        self.ksn2e = self.mo_energy
        self.mo_occ[0,s,:] = self.pyscf_scf.mo_occ[s]
    else:
      raise RuntimeError('0>nspin>2?')
      
    self.xc_code_previous = copy(self.xc_code)
    self.xc_code = "HF"
    self.fermi_energy = comput_fermi_energy(self.mo_energy, sum(self.nelec), self.telec)
    return etot

  def get_hcore(self, mol=None, **kw):
    hcore = -0.5*self.laplace_coo().toarray()
    hcore += self.vnucele_coo(**kw).toarray()
    return hcore

  def vnucele_coo(self, **kw): # Compute matrix elements of nuclear-electron interaction (attraction)
    '''
    it subtracts the computed matrix elements from the total Hamiltonian to find out the 
    nuclear-electron interaction in case of SIESTA import. This means that Vne is defined by 
    Vne = H_KS - T - V_H - V_xc
    '''
    if self.pseudo:
      # This is wrong after a repeated SCF. A better way would be to use pseudo-potentials and really recompute.
      tkin = (-0.5*self.laplace_coo()).tocsr()
      vxc  = self.vxc_lil(dm=self.dm_mf, xc_code=self.xc_code_mf, **kw)[0].tocsr()
      ham  = self.get_hamiltonian()[0].tocsr()
      vhar = self.vhartree_coo(dm=self.dm_mf, **kw)  

      #vhar for spin has two components, so for tocsr() must be split 
      if (self.nspin==1): vha = vhar.tocsr()
      elif (self.nspin==2): vha = vhar[0].tocsr()+vhar[1].tocsr()
    
      vne  = ham-tkin-vha-vxc
      
    else :
      vne  = self.vnucele_coo_coulomb(**kw)
    return vne.tocoo()

  def add_pb_hk(self, **kw): return self.pb,self.hkernel_den

  def get_ovlp(self, sv=None):
    from pyscf.nao.m_overlap_am import overlap_am
    sv = self if sv is None else sv
    return sv.overlap_coo(funct=overlap_am).toarray()

  def vhartree_coo(self, **kw):
    return vhartree_coo(self, **kw)

  def vhartree_den(self, **kw):
    '''Compute matrix elements of the Hartree potential and return dense matrix compatible with RHF or UHF'''
    co = self.vhartree_coo(**kw)
    if self.nspin==1:
      vh = co.toarray()
    elif self.nspin==2:
      vh = np.stack((co[0].toarray(), co[1].toarray() ))
    else:
      raise RuntimeError('nspin>2?')
    return vh

  def get_j(self, dm=None, **kw):
    '''Compute J matrix for the given density matrix (matrix elements of the Hartree potential).'''
    if dm is None: dm = self.make_rdm1()
    return self.vhartree_den(dm=dm)

  def get_k(self, dm=None, **kw):
    '''Compute K matrix for the given density matrix.'''
    from pyscf.nao.m_kmat_den import kmat_den
    if dm is None: dm = self.make_rdm1()
    
    if False:
      print(__name__, ' get_k: self.kmat_algo ', self.kmat_algo, dm.shape)
      if len(dm.shape)==5:
        print(__name__, 'nelec dm', (dm[0,:,:,:,0]*self.overlap_lil().toarray()).sum())
      elif len(dm.shape)==2 or len(dm.shape)==3:
        print(__name__, 'nelec dm', (dm*self.overlap_lil().toarray()).sum())
      else:
        print(__name__, dm.shape)
    
    kmat_algo = kw['kmat_algo'] if 'kmat_algo' in kw else self.kmat_algo

    #if self.verbosity>1: print(__name__, "\t\t====> Matrix elements of Fock exchange operator will be calculated by using '{}' algorithm.\f".format(kmat_algo))
    return kmat_den(self, dm=dm, algo=kmat_algo, **kw)

    if self.kmat_timing is not None: t1 = timer()
    kmat = kmat_den(self, dm=dm, algo=kmat_algo, **kw)
    if self.kmat_timing is not None: self.kmat_timing += timer()-t1
    return kmat


  def get_jk(self, mol=None, dm=None, hermi=1, **kw):
    if dm is None: dm = self.make_rdm1()
    j = self.get_j(dm, **kw)
    k = self.get_k(dm, **kw)
    return j,k

  def get_veff(self, dm=None, **kw):
    '''Hartree-Fock potential matrix for the given density matrix'''
    if dm is None: dm = self.make_rdm1()
    vj, vk = self.get_jk (dm, **kw)
    if (self.nspin==1): v_eff = vj - vk * .5
    if (self.nspin==2): v_eff = vj[0] + vj[1] - vk
    return v_eff

  def get_fock(self, h1e=None, dm=None, **kw):
    '''Fock matrix for the given density matrix (matrix elements of the Fockian).'''
    if dm is None: dm = self.make_rdm1()
    if h1e is None: h1e = self.get_hcore()
    fock = h1e +  self.get_veff(dm, **kw) 
    return fock
