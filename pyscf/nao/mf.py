from __future__ import print_function, division
import sys, numpy as np
from pyscf.nao import nao, prod_basis_c

#
#
#
class mf(nao):

  def __init__(self, **kw):
    """ Constructor a mean field class (store result of a mean-field calc, deliver density matrix etc) """
    #print(__name__, 'before construct')
    nao.__init__(self, **kw)
    self.dtype = kw['dtype'] if 'dtype' in kw else np.float64
    self.pseudo = hasattr(self, 'sp2ion') 
    self.verbosity = kw['verbosity'] if 'verbosity' in kw else 0
    self.gen_pb = kw['gen_pb'] if 'gen_pb' in kw else True
    
    if 'mf' in kw:
      #print(__name__, 'init_mf')
      self.init_mf(**kw)
    elif 'label' in kw:
      self.init_mo_coeff_label(**kw)
      self.xc_code = 'LDA,PZ' # just a guess...
    elif 'gpaw' in kw:
      self.init_mo_coeff_label(**kw)
      self.xc_code = 'LDA,PZ' # just a guess, but in case of GPAW there is a field...
    elif 'openmx' in kw:
      self.xc_code = 'GGA,PBE' # just a guess...
      pass
    else:
      raise RuntimeError('unknown constructor')
    if self.verbosity>0: print(__name__, ' pseudo ', self.pseudo)
    self.init_libnao()
    if self.gen_pb:
      self.pb = prod_basis_c()
      if self.verbosity>0: print(__name__, ' dtype ', self.dtype, ' norbs ', self.norbs)
      self.pb.init_prod_basis_pp_batch(nao=self, **kw)

  def make_rdm1(self, mo_coeff=None, mo_occ=None):
    """ This is spin-saturated case"""
    # from pyscf.scf.hf import make_rdm1 -- different index order here
    if mo_occ is None: mo_occ = self.mo_occ[0,0,:]
    if mo_coeff is None: mo_coeff = self.mo_coeff[0,0,:,:,0]
    mocc = mo_coeff[mo_occ>0,:]
    dm = np.zeros((1,self.nspin,self.norbs,self.norbs,1))
    dm[0,0,:,:,0] = np.dot(mocc.T.conj()*mo_occ[mo_occ>0], mocc)
    return dm

  def init_mf(self, **kw):
    """ Constructor a self-consistent field calculation class """
    #print(__name__, 'mf init_mf>>>>>>')

    self.telec = kw['telec'] if 'telec' in kw else 0.0000317 # 10K
    mf = self.mf = kw['mf']
    self.xc_code = mf.xc if hasattr(mf, 'xc') else 'HF'

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
        for i, sh in enumerate(self.wfsx.x.shape): size_x[i] = sh

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

  def get_init_guess(self, mol=None, key=None):
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
  
  def dos(self, comegas, **kw):
    """ Ordinary Density of States (from the current mean-field eigenvalues) """
    from pyscf.nao.scf_dos import scf_dos
    return scf_dos(self, comegas, **kw)

  def pdos(self, comegas, **kw):
    """ Partial Density of States (resolved in angular momentum of atomic orbitals) """
    from pyscf.nao.pdos import pdos
    return pdos(self, comegas, **kw)

  def lsoa_dos(self, comegas, **kw):
    """ Partial Density of States (contributions from a given list of atoms) """
    from pyscf.nao.pdos import lsoa_dos
    return lsoa_dos(self, comegas, **kw)

  def gdos(self, comegas, **kw):
    """ Some molecular orbital population analysis """
    from pyscf.nao.pdos import gdos
    return gdos(self, comegas, **kw)

  def read_wfsx(self, fname, **kw):
    """ An occasional reading of the SIESTA's .WFSX file """
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    self.wfsx = siesta_wfsx_c(fname=fname, **kw)
    
    assert self.nkpoints == self.wfsx.nkpoints
    assert self.norbs == self.wfsx.norbs 
    assert self.nspin == self.wfsx.nspin
    orb2m = self.get_orb2m()
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.x[k,s,n,:,:])

    self.mo_coeff = np.require(self.wfsx.x, dtype=self.dtype, requirements='CW')
    self.mo_energy = np.require(self.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else self.hsx.telec
    self.nelec = kw['nelec'] if 'nelec' in kw else self.hsx.nelec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd
    return self


  def plot_contour(self, w=0.0):
    """
      Plot contour with poles of Green's function in the self-energy 
      SelfEnergy(w) = G(w+w')W(w')
      with respect to w' = Re(w')+Im(w')
      Poles of G(w+w') are located: w+w'-(E_n-Fermi)+i*eps sign(E_n-Fermi)==0 ==> 
      w'= (E_n-Fermi) - w -i eps sign(E_n-Fermi)
    """
    try :
      import matplotlib.pyplot as plt
      from matplotlib.patches import Arc, Arrow 
    except:
      print('no matplotlib?')
      return

    fig,ax = plt.subplots()
    fe = self.fermi_energy
    ee = self.mo_energy
    iee = 0.5-np.array(ee>fe)
    eew = ee-fe-w
    ax.plot(eew, iee, 'r.', ms=10.0)
    pp = list()
    pp.append(Arc((0,0),4,4,angle=0, linewidth=2, theta1=0, theta2=90, zorder=2, color='b'))
    pp.append(Arc((0,0),4,4,angle=0, linewidth=2, theta1=180, theta2=270, zorder=2, color='b'))
    pp.append(Arrow(0,2,0,-4,width=0.2, color='b', hatch='o'))
    pp.append(Arrow(-2,0,4,0,width=0.2, color='b', hatch='o'))
    for p in pp: ax.add_patch(p)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.ylim(-3.0,3.0)
    plt.show()

#
# Example of reading pySCF mean-field calculation.
#
if __name__=="__main__":
  from pyscf import gto, scf as scf_gto
  from pyscf.nao import nao, mf
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvdz') # coordinates in Angstrom!
  dft = scf_gto.RKS(mol)
  dft.kernel()
  
  sv = mf(mf=dft, gto=dft.mol, rcut_tol=1e-9, nr=512, rmin=1e-6)
  
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
