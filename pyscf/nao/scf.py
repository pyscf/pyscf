from __future__ import print_function, division
import sys, numpy as np
from pyscf.nao import nao, prod_basis_c

#
#
#
class scf(nao):

  def __init__(self, **kw):
    """ Constructor a self-consistent field calculation class """
    nao.__init__(self, **kw)
    mf = self.mf = kw['mf']
    self.mo_coeff = mf.mo_coeff
    self.mo_energy = mf.mo_energy
    self.mo_occ = mf.mo_occ
    self.pb = prod_basis_c()
    self.pb.init_prod_basis_pp_batch(nao=self, **kw)
    self.kernel = None # I am not initializing it here because different methods need different kernels...
    

#
# Example of reading pySCF mean-field calculation.
#
if __name__=="__main__":
  from pyscf import gto, scf as scf_gto
  from pyscf.nao import nao, scf
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvdz') # coordinates in Angstrom!
  dft = scf_gto.RKS(mol)
  dft.kernel()
  
  sv = scf(mf=dft, gto=dft.mol, rcut_tol=1e-9, nr=512, rmin=1e-6)
  
  print(sv.ao_log.sp2norbs)
  print(sv.ao_log.sp2nmult)
  print(sv.ao_log.sp2rcut)
  print(sv.ao_log.sp_mu2rcut)
  print(sv.ao_log.nr)
  print(sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1])
  print(sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape)
  print(dir(sv.pb))
  print(sv.pb.norbs)
  print(sv.pb.npdp)
  print(sv.pb.c2s[-1])
  

