#!/usr/bin/env python
'''
Hartree-Fock with numerical atomic orbitals
'''

import sys
import numpy as np
from pyscf.scf import hf
from pyscf.lib import logger

  
class SCF(hf.SCF):
  '''SCF class adapted for NAOs.'''
  def __init__(self, sv, pseudo=None, **kvargs):
    from pyscf.nao.m_prod_basis import prod_basis_c
    self.sv = sv
    hf.SCF.__init__(self, sv)
    self.direct_scf = False # overriding the attribute from hf.SCF ...
    self.pb = prod_basis_c().init_prod_basis_pp(sv, **kvargs)
    self.hkernel_den = self.pb.comp_coulomb_den(**kvargs)
    self.pseudo = hasattr(sv, 'xml_dict') if pseudo is None else pseudo 

  def add_pb_hk(self, **kvargs):
    """ This is adding a product basis attribute to the class and making possible then to compute the matrix elements of Hartree potential or Fock exchange."""
    from pyscf.nao.m_prod_basis import prod_basis_c
    if hasattr(self, 'pb'):
      pb = self.pb
      hk = self.hkernel_den
    else:
      pb = self.pb = prod_basis_c().init_prod_basis_pp(self, **kvargs)
      hk = self.hkernel_den = pb.comp_coulomb_den(**kvargs)
    return pb,hk

  def kernel(self, dump_chk=False, **kwargs):
    return hf.SCF.kernel(self, dump_chk=dump_chk, **kwargs)

  def get_hcore(self, sv=None):
    sv = self.sv if sv is None else sv
    hcore = 0.5*sv.laplace_coo().toarray()
    hcore += self.vnucele_coo().toarray()
    return hcore

  def vnucele_coo(self, **kvargs): # Compute matrix elements of nuclear-electron interaction (attraction)
    if self.pseudo:
      tkin = (0.5*self.sv.laplace_coo()).tocsr()
      dm = self.sv.comp_dm()
      vhar = self.vhartree_coo(dm=dm, **kvargs).tocsr()
      vxc  = self.sv.vxc_lil(dm=dm, **kvargs).tocsr()
      vne  = self.sv.get_hamiltonian(**kvargs)[0].tocsr()-tkin-vhar-vxc
    else :
      vne  = self.sv.vnucele_coo_coulomb(**kvargs)
    return vne.tocoo()

  def vhartree_coo(self, **kvargs):
    from pyscf.nao.m_vhartree_coo import vhartree_coo
    return vhartree_coo(self, **kvargs)
    
  def get_ovlp(self, sv=None):
    from pyscf.nao.m_overlap_am import overlap_am
    sv = self.sv if sv is None else sv
    return sv.overlap_coo(funct=overlap_am).toarray()
  
  def get_init_guess(self, sv=None, key='minao'):
    sv = self.sv if sv is None else sv
    return sv.get_init_guess(key=key)

  def get_j(self, dm=None, **kvargs):
    '''Compute J matrix for the given density matrix.'''
    if dm is None: dm = self.make_rdm1()
    from pyscf.nao.m_vhartree_coo import vhartree_coo
    return vhartree_coo(self, dm=dm).toarray()

  def get_k(self, dm=None, **kvargs):
    '''Compute K matrix for the given density matrix.'''
    from pyscf.nao.m_kmat_den import kmat_den
    if dm is None: dm = self.make_rdm1()
    return kmat_den(self, dm=dm, **kvargs)

  def get_jk(self, sv=None, dm=None, **kvargs):
    if sv is None: sv = self.sv
    if dm is None: dm = self.make_rdm1()
    j = self.get_j(dm, **kvargs)
    k = self.get_k(dm, **kvargs)
    return j,k

  def energy_nuc(self):
    return self.sv.energy_nuc()

RHF = SCF
