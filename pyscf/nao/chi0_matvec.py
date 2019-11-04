from __future__ import print_function, division
from copy import copy
import numpy as np
from numpy import array, argmax
from scipy.sparse import csr_matrix, coo_matrix
from timeit import default_timer as timer
from scipy.linalg import blas

from pyscf.nao import mf
from pyscf.nao.m_tddft_iter_gpu import tddft_iter_gpu_c
from pyscf.nao.m_chi0_noxv import chi0_mv_gpu, chi0_mv
from pyscf.data.nist import HARTREE2EV

class chi0_matvec(mf):
  """ A class to organize the application of non-interacting response to a vector """

  def __init__(self, **kw):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    self.dtype = kw['dtype'] if 'dtype' in kw else np.float32
    for x in ['dtype']: kw.pop(x, None)
    mf.__init__(self, dtype=self.dtype, **kw)

    if self.dtype == np.float32:
      self.gemm = blas.sgemm
    elif self.dtype == np.float64:
      self.gemm = blas.dgemm
    else:
      raise ValueError("dtype can be only float32 or float64")

    self.dealloc_hsx = kw['dealloc_hsx'] if 'dealloc_hsx' in kw else True
    self.eps = kw['iter_broadening'] if 'iter_broadening' in kw else 0.00367493
    self.GPU = GPU = kw['GPU'] if 'GPU' in kw else None
    self.nfermi_tol = nfermi_tol = kw['nfermi_tol'] if 'nfermi_tol' in kw else 1e-5
    self.telec = kw['telec'] if 'telec' in kw else self.telec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy

    assert type(self.eps)==float

    self.div_numba = None
    if self.use_numba:
      from pyscf.nao.m_div_eigenenergy_numba import div_eigenenergy_numba
      self.div_numba = div_eigenenergy_numba

    if hasattr(self, 'hsx') and self.dealloc_hsx: self.hsx.deallocate()     # deallocate hsx

    self.ksn2e = self.mo_energy # Just a pointer here. Is it ok?
    
    if 'fermi_energy' in kw:
      if self.verbosity>0: print(__name__, 'Fermi energy is specified => recompute occupations')
      ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
      for s,n2fd in enumerate(ksn2fd[0]):
        if not all(n2fd>self.nfermi_tol): continue
        print(self.telec, s, self.nfermi_tol, n2fd)
        raise RuntimeError(__name__, 'telec is too high?')
      ksn2f = self.ksn2f = (3-self.nspin)*ksn2fd
    else:
      ksn2f = self.ksn2f = self.mo_occ
    
    self.nfermi = array([argmax(ksn2f[0,s]<self.nfermi_tol) for s in range(self.nspin)], dtype=int)
    self.vstart = array([argmax(1.0-ksn2f[0,s]>=self.nfermi_tol) for s in range(self.nspin)], dtype=int)
    self.xocc = [self.mo_coeff[0,s,:nfermi,:,0] for s,nfermi in enumerate(self.nfermi)]
    self.xvrt = [self.mo_coeff[0,s,vstart:,:,0] for s,vstart in enumerate(self.vstart)]
 
    if self.verbosity>4 :
      #print(__name__, '\t====> self.dtype ', self.dtype)
      print(__name__, '\t====> self.xocc[0].dtype ', self.xocc[0].dtype)
      print(__name__, '\t====> self.xvrt[0].dtype ', self.xvrt[0].dtype)
      print(__name__, '\t====> MO energies (ksn2e) (eV):\n{},\tType: {}'.format(self.ksn2e*HARTREE2EV,self.ksn2e.dtype))
      print(__name__, '\t====> Occupations (ksn2f):\n{},\tType: {}'.format(self.ksn2f,self.ksn2f.dtype))

    self.rf0_ncalls = 0
            
    if not hasattr(self, 'pb'):
      print('no pb?')
      return
      
    pb = self.pb
    self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)
    self.td_GPU = tddft_iter_gpu_c(GPU, self.mo_coeff[0,0,:,:,0], self.ksn2f, self.ksn2e, self.norbs, self.nfermi, self.nprod, self.vstart)

  def apply_rf0(self, sp2v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    expect_shape=tuple([self.nspin*self.nprod])
    assert np.all(sp2v.shape==expect_shape), "{} {}".format(sp2v.shape,expect_shape)
    self.rf0_ncalls+=1

    if self.td_GPU.GPU is None:
      return chi0_mv(self, sp2v, comega)
    else:
      return chi0_mv_gpu(self, sp2v, comega)

  def comp_polariz_nonin_xx(self, comegas):
    """  Compute the non interacting polarizability along the xx direction """
    pxx = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      dn0 = self.apply_rf0(vext[0], comega)
      pxx[iw] = np.dot(dn0, vext[0])
    return pxx

  def comp_polariz_nonin_xx_atom_split(self, comegas):
    """  Compute the non interacting polarizability along the xx direction """
    aw2pxx = np.zeros((self.natoms, comegas.shape[0]), dtype=self.dtypeComplex)

    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      dn0 = self.apply_rf0(vext[0], comega)
      for ia in range(self.natoms):
          dn0_atom = np.zeros(self.nprod, dtype=complex)
          dn0_atom[self.pb.c2s[ia]:self.pb.c2s[ia+1]] = dn0[self.pb.c2s[ia]:self.pb.c2s[ia+1]]
          aw2pxx[ia, iw] = np.dot(dn0_atom, vext[0])
    return aw2pxx

  def comp_polariz_nonin_ave(self, comegas , **kw):
    """  Compute the average non-interacting polarizability """
    p_avg = np.zeros(comegas.shape, dtype=self.dtypeComplex)
    verbosity = kw['verbosity'] if 'verbosity' in kw else self.verbosity
    nww = len(comegas)
    for xyz in range(3):
      vext = np.concatenate([self.moms1[:,xyz] for s in range(self.nspin)])
      for iw, comega in enumerate(comegas):
        if verbosity>1: print(__name__, xyz, iw, nww, comega*HARTREE2EV)
        dn0 = self.apply_rf0(vext, comega)
        p_avg[iw] += (dn0*vext).sum()
    return p_avg/3.0

  def comp_dens_nonin_along_Eext(self, comegas, Eext = np.array([1.0, 0.0, 0.0])):
    """ 
        Compute a the average non-interacting polarizability along the Eext direction
        for the frequencies comegas.
        
        Input Parameters:
            comegas (1D array, complex): the real part contains the frequencies at which the polarizability
                        should be computed. The imaginary part id the width of the polarizability define as self.eps
            Eext (1D xyz array, real): direction of the external field


        Calculated quantity:
            self.p_mat (complex array, dim: [3, 3, comega.size]): store the (3, 3) polarizability matrix 
                                [[Pxx, Pxy, Pxz],
                                 [Pyx, Pyy, Pyz],
                                 [Pzx, Pzy, Pzz]] for each frequency.
            self.dn (complex array, dim: [3, comegas.size, self.nprod]): store the density change
    """
 
    assert Eext.size == 3
    
    self.p0_mat = np.zeros((3, 3, comegas.size), dtype=self.dtypeComplex)
    self.dn0 = np.zeros((3, comegas.size, self.nprod), dtype=self.dtypeComplex)

    Edir = Eext/np.dot(Eext, Eext)

    vext = np.transpose(self.moms1)
    for xyz, Exyz in enumerate(Edir):
        if Exyz == 0.0: continue
        for iw, comega in enumerate(comegas):
            self.dn0[xyz, iw, :] = self.apply_rf0(vext[xyz], comega)

    self.p0_mat = np.einsum("iwp,jp->ijw", self.dn0, vext)
