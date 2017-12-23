from __future__ import print_function, division
import sys, numpy as np
from pyscf.nao import mf

#
#
#
class umf(mf):
  """ Unrestricted Mean Field calc, i.e. spin-saturated"""

  def __init__(self, **kw):
    """ Constructor a mean field class (store result of a mean-field calc, deliver density matrix etc) """
    mf.__init__(self, **kw)
    assert self.nspin==2
    if 'mf' in kw: self.init_umf_from_pyscf(**kw)

  def init_umf_from_pyscf(self, **kw):
    """ Constructor a self-consistent field calculation class """
    from pyscf.nao import conv_yzx2xyz_c
    from pyscf.nao.m_fermi_energy import fermi_energy as comput_fermi_energy

    mf = self.mf = kw['mf']
    self.mo_coeff = np.require(np.zeros((1,self.nspin,self.norbs,self.norbs,1), dtype=self.dtype), requirements='CW')
    conv = conv_yzx2xyz_c(kw['gto'])
    
    for s in range(self.nspin):
      self.mo_coeff[0,s,:,:,0] = conv.conv_yzx2xyz_1d(mf.mo_coeff[s], conv.m_xyz2m_yzx).T

    self.mo_energy = np.require(mf.mo_energy, dtype=self.dtype, requirements='CW')
    
    self.mo_occ = np.require(np.zeros((1,self.nspin,self.norbs),dtype=self.dtype), requirements='CW')
    self.mo_occ[0,0:self.nspin,:] = mf.mo_occ
    np.require(mf.mo_occ, dtype=self.dtype, requirements='CW')
    nelec = self.mo_occ.sum()
    #print(__name__, nelec, self.telec)
    fermi = comput_fermi_energy(self.mo_energy, nelec, self.telec)
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else fermi
    self.xc_code = mf.xc if hasattr(mf, 'xc') else 'HF'

  def make_rdm1(self, mo_coeff=None, mo_occ=None):
    # from pyscf.scf.hf import make_rdm1 -- different index order here
    if mo_occ is None: mo_occ = self.mo_occ[0,:,:]
    if mo_coeff is None: mo_coeff = self.mo_coeff[0,:,:,:,0]
    dm = np.zeros((1,self.nspin,self.norbs,self.norbs,1))
    for s in range(self.nspin):
      mocc = mo_coeff[s,mo_occ[s]>0,:]
      dm[0,s,:,:,0] = np.dot(mocc.T.conj()*mo_occ[s,mo_occ>0], mocc)
    return dm

#
# Example of reading pySCF mean-field calculation.
#
if __name__=="__main__":
  from pyscf import gto, scf as scf_gto
  from pyscf.nao import rmf
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvdz') # coordinates in Angstrom!
  dft = scf_gto.UKS(mol)
  dft.kernel()
  
  sv = umf(mf=dft, gto=dft.mol, rcut_tol=1e-9, nr=512, rmin=1e-6)
  
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
  

