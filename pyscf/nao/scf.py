from __future__ import print_function, division
import sys, numpy as np
from pyscf.nao import tddft_iter
from pyscf.scf import hf
from copy import copy
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l

#
#
#
class scf(tddft_iter):

  def __init__(self, **kw):
    """ Constructor a self-consistent field """
    self.perform_scf = kw['perform_scf'] if 'perform_scf' in kw else False
    ips = map(lambda x: kw.pop(x,None), ['xc_code', 'dealloc_hsx', 'dtype'])
    tddft_iter.__init__(self, dtype=np.float64, xc_code='RPA', dealloc_hsx=False, **kw)
    #print(__name__, ' dtype ', self.dtype)

    self.xc_code_kernel = copy(self.xc_code)
    self.xc_code = self.xc_code_mf
    self.dm_mf   = self.make_rdm1() # necessary to get_hcore(...) in case of pp starting point
    self.hkernel_den = pack2den_l(self.kernel)
    self.pyscf_hf = hf.SCF(self)
    self.pyscf_hf.direct_scf = False # overriding the attributes from hf.SCF ...
    self.pyscf_hf.get_hcore = self.get_hcore
    self.pyscf_hf.get_ovlp = self.get_ovlp
    self.pyscf_hf.get_j = self.get_j
    self.pyscf_hf.get_jk = self.get_jk
    self.pyscf_hf.energy_nuc = self.energy_nuc
    if self.perform_scf : self.kernel_scf(**kw)

  def kernel_scf(self, dump_chk=False, **kw):
    """ This does the actual SCF loop so far only HF """
    from pyscf.nao.m_fermi_energy import fermi_energy as comput_fermi_energy
    dm0 = self.get_init_guess()
    etot = self.pyscf_hf.kernel(dm0=dm0, dump_chk=dump_chk, **kw)
    self.mo_coeff[0,0,:,:,0] = self.pyscf_hf.mo_coeff.T
    self.mo_energy[0,0,:] = self.pyscf_hf.mo_energy
    self.ksn2e = self.mo_energy
    self.mo_occ[0,0,:] = self.pyscf_hf.mo_occ
    self.xc_code_previous = copy(self.xc_code)
    self.xc_code = "HF"
    #print('self.fermi_energy is not updated in scf!!!')
    self.fermi_energy = comput_fermi_energy(self.mo_energy, self.nelec, self.telec)
    return etot

  def get_hcore(self, mol=None, **kw):
    hcore = 0.5*self.laplace_coo().toarray()
    hcore += self.vnucele_coo(**kw).toarray()
    return hcore

  def vnucele_coo(self, **kw): # Compute matrix elements of nuclear-electron interaction (attraction)
    if self.pseudo:
      # This is wrong after a repeated SCF. A better way would be to use pseudo-potentials and really recompute.
      tkin = (0.5*self.laplace_coo()).tocsr()
      vhar = self.vhartree_coo(dm=self.dm_mf, **kw).tocsr()
      vxc  = self.vxc_lil(dm=self.dm_mf, xc_code=self.xc_code_mf, **kw).tocsr()
      vne  = self.get_hamiltonian()[0].tocsr()-tkin-vhar-vxc
    else :
      vne  = self.vnucele_coo_coulomb(**kw)
    return vne.tocoo()

  def vhartree_coo(self, **kw):
    from pyscf.nao.m_vhartree_coo import vhartree_coo
    return vhartree_coo(self, **kw)

  def add_pb_hk(self, **kw): return self.pb,self.hkernel_den

  def get_ovlp(self, sv=None):
    from pyscf.nao.m_overlap_am import overlap_am
    sv = self if sv is None else sv
    return sv.overlap_coo(funct=overlap_am).toarray()

  def get_j(self, dm=None, **kw):
    '''Compute J matrix for the given density matrix (matrix elements of the Hartree potential).'''
    if dm is None: dm = self.make_rdm1()
    from pyscf.nao.m_vhartree_coo import vhartree_coo
    return vhartree_coo(self, dm=dm).toarray()

  def get_k(self, dm=None, **kw):
    '''Compute K matrix for the given density matrix.'''
    from pyscf.nao.m_kmat_den import kmat_den
    if dm is None: dm = self.make_rdm1()
    return kmat_den(self, dm=dm, **kw)

  def get_jk(self, sv=None, dm=None, **kvargs):
    if sv is None: sv = self.sv
    if dm is None: dm = self.make_rdm1()
    j = self.get_j(dm, **kvargs)
    k = self.get_k(dm, **kvargs)
    return j,k
