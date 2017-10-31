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
    self.dtype = kw['dtype'] if 'dtype' in kw else np.float32
    if 'mf' in kw:
      self.init_mf(**kw)
    elif 'label' in kw:
      self.init_mo_coeff_label(**kw)
    elif 'gpaw' in kw:
      self.init_mo_coeff_label(**kw)
    elif 'openmx' in kw:
      pass
    else:
      raise RuntimeError('unknown constructor')
    self.xc_code = 'LDA,PZ' # just a guess...
    self.init_libnao()
    self.pb = prod_basis_c()
    self.pb.init_prod_basis_pp_batch(nao=self, **kw)
    self.kernel = None # I am not initializing it here because different methods need different kernels...

  def init_mf(self, **kw):
    """ Constructor a self-consistent field calculation class """
    from pyscf.nao import conv_yzx2xyz_c
    mf = self.mf = kw['mf']
    self.mo_coeff = np.require(np.zeros((1,self.nspin,self.norbs,self.norbs,1), dtype=self.dtype), requirements='CW')
    conv = conv_yzx2xyz_c(kw['gto'])
    self.mo_coeff[0,0,:,:,0] = conv.conv_yzx2xyz_1d(mf.mo_coeff, conv.m_xyz2m_yzx).T
    
    self.mo_energy = np.require(mf.mo_energy, dtype=self.dtype, requirements='CW')
    self.mo_occ = np.require(np.zeros((1,self.nspin,self.norbs),dtype=self.dtype), requirements='CW')
    self.mo_occ[0,0,:] = mf.mo_occ
    np.require(mf.mo_occ, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else 0.0000317 # 10K
    nelec = self.mo_occ.sum()
    assert int(nelec) % 2 == 0
    nocc = int(nelec/2)
    fermi_energy = (self.mo_energy[nocc]+self.mo_energy[nocc-1])/2.0
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else fermi_energy

  def init_mo_coeff_label(self, **kw):
    """ Constructor a self-consistent field calculation class """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    self.mo_coeff = np.require(self.wfsx.x, dtype=self.dtype, requirements='CW')
    self.mo_energy = np.require(self.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else self.hsx.telec
    self.nelec = kw['nelec'] if 'nelec' in kw else self.hsx.nelec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd

  def diag_check(self, atol=1e-5, rtol=1e-4):
    from pyscf.nao.m_sv_diag import sv_diag 
    ksn2e = self.xml_dict['ksn2e']
    ac = True
    for k,kvec in enumerate(self.xml_dict["k2xyzw"]):
      for spin in range(self.nspin):
        e,x = sv_diag(self, kvec=kvec[0:3], spin=spin)
        eref = ksn2e[k,spin,:]
        acks = np.allclose(eref,e,atol=atol,rtol=rtol)
        ac = ac and acks
        if(not acks):
          aerr = sum(abs(eref-e))/len(e)
          print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
    return ac

  def get_occupations(self, telec=None, ksn2e=None, fermi_energy=None):
    """ Compute occupations of electron levels according to Fermi-Dirac distribution """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    Telec = self.hsx.telec if telec is None else telec
    ksn2E = self.wfsx.ksn2e if ksn2e is None else ksn2e
    Fermi = self.fermi_energy if fermi_energy is None else fermi_energy
    ksn2fd = fermi_dirac_occupations(Telec, ksn2E, Fermi)
    ksn2fd = (3.0-self.nspin)*ksn2fd
    return ksn2fd

  def init_libnao(self, wfsx=None):
    """ Initialization of data on libnao site """
    from pyscf.nao.m_libnao import libnao
    from pyscf.nao.m_sv_chain_data import sv_chain_data
    from ctypes import POINTER, c_double, c_int64, c_int32, byref

    if wfsx is None:
        data = sv_chain_data(self)
        # (nkpoints, nspin, norbs, norbs, nreim)
        #print(' data ', sum(data))
        size_x = np.array([1, self.nspin, self.norbs, self.norbs, 1], dtype=np.int32)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True
    else:
        size_x = np.zeros(len(self.wfsx.x.shape), dtype=np.int32)
        for i, sh in enumerate(self.wfsx.x.shape):
            size_x[i] = sh

        data = sv_chain_data(self)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True

    libnao.init_aos_libnao.argtypes = (POINTER(c_int64), POINTER(c_int64))
    info = c_int64(-999)
    libnao.init_aos_libnao(c_int64(self.norbs), byref(info))
    if info.value!=0: raise RuntimeError("info!=0")
    return self

  def vxc_lil(self, **kw):   # Compute exchange-correlation potentials
    from pyscf.nao.m_vxc_lil import vxc_lil
    return vxc_lil(self, deriv=1, **kw)

  #def comp_dm(self):  # Computes the density matrix
  #  from pyscf.nao.m_comp_dm import comp_dm
  #  dm = comp_dm(self.wfsx.x, self.get_occupations())
  #  return dm

  def make_rdm1(self, mo_coeff=None, mo_occ=None):
    # from pyscf.scf.hf import make_rdm1 -- different index order here
    if mo_occ is None: mo_occ = self.mo_occ[0,0,:]
    if mo_coeff is None: mo_coeff = self.mo_coeff[0,0,:,:,0]
    mocc = mo_coeff[mo_occ>0,:]
    dm = np.zeros((1,self.nspin,self.norbs,self.norbs,1))
    dm[0,0,:,:,0] = np.dot(mocc.T.conj()*mo_occ[mo_occ>0], mocc)
    return dm

  def dens_elec(self, coords, dm): # Compute electronic density for a given density matrix and on a given set of coordinates
    from pyscf.nao.m_dens_libnao import dens_libnao
    from pyscf.nao.m_init_dm_libnao import init_dm_libnao
    from pyscf.nao.m_init_dens_libnao import init_dens_libnao
    if not self.init_sv_libnao : raise RuntimeError('not self.init_sv_libnao')
    if init_dm_libnao(dm) is None : raise RuntimeError('init_dm_libnao(dm) is None')
    if init_dens_libnao()!=0 : raise RuntimeError('init_dens_libnao()!=0')
    return dens_libnao(coords, self.nspin)

  def exc(self, dm, xc_code, **kw):   # Compute exchange-correlation energies
    from pyscf.nao.m_exc import exc
    return exc(self, dm, xc_code, **kw)

  def get_init_guess(self, key=None):
    """ Compute an initial guess for the density matrix. """
    from pyscf.scf.hf import init_guess_by_minao
    if hasattr(self, 'mol'):
      dm = init_guess_by_minao(self.mol)
    else:
      dm = self.make_rdm1()  # the loaded ks orbitals will be used
      if dm.shape[0:2]==(1,1) and dm.shape[4]==1 : dm = dm.reshape((self.norbs,self.norbs))
    return dm

  def get_hamiltonian(self): # Returns the stored matrix elements of current hamiltonian 
    return self.hsx.spin2h4_csr

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
  

