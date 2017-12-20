from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
from pyscf.nao import mf
from pyscf.nao.m_tddft_iter_gpu import tddft_iter_gpu_c
from pyscf.nao.m_chi0_noxv import chi0_mv_gpu, chi0_mv
from pyscf.nao.m_blas_wrapper import spmv_wrapper
from copy import copy

import scipy
if int(scipy.__version__[0]) > 0:
    scipy_ver = 1
else:
    scipy_ver = 0

try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_iter(mf):
  """ 
    Iterative TDDFT a la PK, DF, OC JCTC
    
    Input Parameters:
    -----------------
        kw: keywords arguments:
            * tddft_iter_tol (real, default: 1e-3): tolerance to reach for 
                            convergency in the iterative procedure.
  """

  def __init__(self, **kw):
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    mf.__init__(self, **kw)
    self.xc_code_mf = copy(self.xc_code)
    self.dealloc_hsx = kw['dealloc_hsx'] if 'dealloc_hsx' in kw else True
    self.maxiter = kw['maxiter'] if 'maxiter' in kw else 1000
    self.tddft_iter_tol = kw['tddft_iter_tol'] if 'tddft_iter_tol' in kw else 1e-3
    self.eps = kw['iter_broadening'] if 'iter_broadening' in kw else 0.00367493
    self.GPU = GPU = kw['GPU'] if 'GPU' in kw else None
    self.xc_code = xc_code = kw['xc_code'] if 'xc_code' in kw else self.xc_code
    self.nfermi_tol = nfermi_tol = kw['nfermi_tol'] if 'nfermi_tol' in kw else 1e-5
    self.dtype = kw['dtype'] if 'dtype' in kw else np.float32
    self.telec = kw['telec'] if 'telec' in kw else self.telec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    
    self.spmv = spmv_wrapper
    if self.dtype == np.float32:
      self.dtypeComplex = np.complex64
      self.gemm = blas.sgemm
      if scipy_ver > 0: self.spmv = blas.sspmv
    elif self.dtype == np.float64:
      self.dtypeComplex = np.complex128
      self.gemm = blas.dgemm
      if scipy_ver > 0: self.spmv = blas.dspmv
    else:
      raise ValueError("dtype can be only float32 or float64")
    self.load_kernel = load_kernel = kw['load_kernel'] if 'load_kernel' in kw else False
    
    assert self.tddft_iter_tol>1e-6
    assert type(self.eps)==float
    
    pb = self.pb

    # deallocate hsx
    if hasattr(self, 'hsx') and self.dealloc_hsx: self.hsx.deallocate()
    
    self.rf0_ncalls = 0
    self.matvec_ncalls = 0

    self.v_dab = pb.get_dp_vertex_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()
    self.cc_da = pb.get_da2cc_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()

    self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)
    self.nprod = self.moms0.size

    if self.verbosity>0 : print(__name__, ' nprod ', self.nprod, ' cc_da.shape ', self.cc_da.shape)

    if load_kernel:
      self.load_kernel_method(**kw)
    else:
      self.kernel,self.kernel_dim = pb.comp_coulomb_pack(dtype=self.dtype) # Lower Triangular Part of the kernel
      assert self.nprod==self.kernel_dim, "%r %r "%(self.nprod, self.kernel_dim)

      xc = xc_code.split(',')[0]
      if xc=='RPA' or xc=='HF': pass
      elif xc=='LDA' or xc=='GGA': self.comp_fxc_pack(kernel=self.kernel, **kw)
      else:
        print(' xc_code', xc_code, xc, xc_code.split(','))
        raise RuntimeError('unkn xc_code')

    if self.verbosity>0 : print(__name__, '      xc_code ', self.xc_code)
        
    # probably unnecessary, require probably does a copy
    # problematic for the dtype, must there should be another option 
    #self.x  = np.require(sv.wfsx.x, dtype=self.dtype, requirements='CW')

    self.ksn2e = np.require(np.zeros((1,self.nspin,self.norbs)), dtype=self.dtype, requirements='CW')
    self.ksn2e[0,0,:] = self.mo_energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    if all(ksn2fd[0,0,:]>nfermi_tol):
      print(self.telec, nfermi_tol, ksn2fd[0,0,:])
      raise RuntimeError(__name__, 'telec is too high?')
    
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>=nfermi_tol)
    
    #print('tddft_iter.__init__')
    #print(self.telec)
    #print(self.ksn2f)
    #print(self.ksn2e)
    #print(self.fermi_energy)
    #print(self.nfermi)
    #print(self.vstart)
    #print(ksn2fd[0,0,:], ksn2fd[0,0,:]<nfermi_tol)
    #print(1.0-ksn2fd[0,0,:], 1.0-ksn2fd[0,0,:]<nfermi_tol)
    #raise RuntimeError('tddft_iter nfermi?')

    self.xocc = self.mo_coeff[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.mo_coeff[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?

    #print(self.xocc.shape)
    #print(self.xvrt.shape)
    
    self.td_GPU = tddft_iter_gpu_c(GPU, self.mo_coeff[0, 0, :, :, 0], self.ksn2f, self.ksn2e, 
            self.norbs, self.nfermi, self.nprod, self.vstart)

  def load_kernel_method(self, kernel_fname, kernel_format="npy", kernel_path_hdf5=None, **kwargs):

      if kernel_format == "npy":
          self.kernel = self.dtype(np.load(kernel_fname))
      elif kernel_format == "txt":
          self.kernel = np.loadtxt(kernel_fname, dtype=self.dtype)
      elif kernel_format == "hdf5":
          import h5py
          if kernel_path_hdf5 is None:
              raise ValueError("kernel_path_hdf5 not set while trying to read kernel from hdf5 file.")
          self.kernel = h5py.File(kernel_fname, "r")[kernel_path_hdf5].value
      else:
          raise ValueError("Wrong format for loading kernel, must be: npy, txt or hdf5, got " + kernel_format)

      if len(self.kernel.shape) > 1:
          raise ValueError("The kernel must be saved in packed format in order to be loaded!")
      
      assert self.nprod*(self.nprod+1)//2 == self.kernel.size, "wrong size for loaded kernel: %r %r "%(self.nprod*(self.nprod+1)//2, self.kernel.size)
      self.kernel_dim = self.nprod

  def comp_fxc_lil(self, **kw): 
    """Computes the sparse version of the TDDFT interaction kernel"""
    from pyscf.nao.m_vxc_lil import vxc_lil
    return vxc_lil(self, deriv=2, ao_log=self.pb.prod_log, **kw)
  
  def comp_fxc_pack(self, **kw): 
    """Computes the packed version of the TDDFT interaction kernel """
    from pyscf.nao.m_vxc_pack import vxc_pack
    vxc_pack(self, deriv=2, ao_log=self.pb.prod_log, **kw)

  def apply_rf0(self, v, comega=1j*0.0):
    """ 
        This applies the non-interacting response function to a vector (a set of vectors?) 
    """
    
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    no = self.norbs

    if self.td_GPU.GPU is None:
        return chi0_mv(self, v, comega)
    else:
        return chi0_mv_gpu(self, v, comega) 

  def comp_veff(self, vext, comega=1j*0.0, x0=None):
    #from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    from scipy.sparse.linalg import lgmres, LinearOperator
    
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)
    resgm, info = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, 
        requirements='C'), x0=x0, tol=self.tddft_iter_tol, maxiter=self.maxiter)
    if info != 0:
        print("LGMRES Warning: info = {0}".format(info))
    return resgm
  
  def vext2veff_matvec(self, v):
    self.matvec_ncalls+=1 
    chi0 = self.apply_rf0(v, self.comega_current)
    
    # For some reason it is very difficult to pass only one dimension
    # of an array to the fortran routines?? matvec[0, :].ctypes.data_as(POINTER(c_float))
    # is not working!!!

    # real part
    chi0_reim = np.require(chi0.real, dtype=self.dtype, requirements=["A", "O"])
    matvec_real = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)
    
    # imaginary part
    chi0_reim = np.require(chi0.imag, dtype=self.dtype, requirements=["A", "O"])
    matvec_imag = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)

    return v - (matvec_real + 1.0j*matvec_imag)


  def kchi0_mv(self, v, comega=None):
    """ Operator O = K chi0 """
    self.matvec_ncalls+=1
    w = self.comega_current if comega is None else comega
    chi0 = self.apply_rf0(v, w)
    
    chi0_reim = np.require(chi0.real, dtype=self.dtype, requirements=["A", "O"])
    matvec_real = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)
    
    chi0_reim = np.require(chi0.imag, dtype=self.dtype, requirements=["A", "O"])
    matvec_imag = self.spmv(self.nprod, 1.0, self.kernel, chi0_reim, lower=1)

    return (matvec_real + 1.0j*matvec_imag)

  def umkckc_mv(self, v):
    """ Operator O = [1- K chi0 K chi0] """
    return v - self.kchi0_mv(self.kchi0_mv(v))

  def upkc_mv(self, v):
    """ Operator O = [1 + K chi0] """
    return v + self.kchi0_mv(v)

  def solve_umkckc(self, vext, comega=1j*0.0, x0=None):
    from scipy.sparse.linalg import lgmres, LinearOperator
    """ This solves a system of linear equations 
           (1 - K chi0 K chi0 ) X = vext 
     or computes 
           X = (1 - K chi0 K chi0 )^{-1} vext 
    """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff2_op = LinearOperator((self.nprod,self.nprod), matvec=self.umkckc_mv, dtype=self.dtypeComplex)
    resgm,info = lgmres(veff2_op, np.require(vext, dtype=self.dtypeComplex, requirements='C'), x0=x0, tol=self.tddft_iter_tol, maxiter=self.maxiter)
    if info != 0:  print("LGMRES Warning: info = {0}".format(info))
    return resgm

  def comp_polariz_nonin_xx(self, comegas):
    """  Compute the non interacting polarizability along the xx direction """
    pxx = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      dn0 = self.apply_rf0(vext[0], comega)
      pxx[iw] = np.dot(dn0, vext[0])
    return pxx

  def comp_polariz_inter_xx(self, comegas):
    """  Compute the interacting polarizability along the xx direction  """
    pxx = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      veff = self.comp_veff(vext[0], comega)
      dn = self.apply_rf0(veff, comega)
      pxx[iw] = np.dot(vext[0], dn)
    return pxx


  def polariz_upkc(self, comegas):
    """ Compute interacting polarizability along the xx direction using an alternative algorighm with chi = chi0 (1+K chi0) [1-K chi0 K chi0]^(-1) """
    pxx = np.zeros((len(comegas),3), dtype=self.dtypeComplex)
    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      v1 = self.solve_umkckc(vext[0], comega)
      dn1 = self.apply_rf0(v1, comega)
      dn2 = self.apply_rf0(self.kchi0_mv(v1), comega)
      pxx[iw,1] = np.dot(vext[0], dn1)
      pxx[iw,2] = np.dot(vext[0], dn2)
      pxx[iw,0] = np.dot(vext[0], dn1+dn2)
    return pxx

  def polariz_dckcd(self, comegas):
    """ Compute a term <d chi0 K chi0 d> """
    pxx = np.zeros(len(comegas), dtype=self.dtypeComplex)
    vext = np.transpose(self.moms1)
    for iw, w in enumerate(comegas):
      v1 = self.kchi0_mv(vext[0], comega=w)
      dn1 = self.apply_rf0(v1, w)
      pxx[iw] = np.dot(vext[0], dn1)
    return pxx

  def comp_polariz_nonin_ave(self, comegas , **kw):
    """  Compute the average non-interacting polarizability """
    p_avg = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    verbosity = kw['verbosity'] if 'verbosity' in kw else self.verbosity
    nww, eV = len(comegas), 27.2114
    vext = np.transpose(self.moms1)
    for xyz in range(3):
        for iw, comega in enumerate(comegas):
          if verbosity>0: print(xyz, iw, nww, comega*eV)
          dn0 = self.apply_rf0(vext[xyz], comega)
          p_avg[iw] += np.dot(dn0, vext[xyz])
    return p_avg/3.0

  def comp_polariz_inter_ave(self, comegas, **kw):
    """  Compute average interacting polarizability  """
    p_avg = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    verbosity = kw['verbosity'] if 'verbosity' in kw else self.verbosity
    vext = np.transpose(self.moms1)
    nww, eV = len(comegas), 27.2114
    for xyz in range(3):
      for iw, comega in enumerate(comegas):
        if verbosity>0: print(xyz, iw, nww, comega*eV)
        veff = self.comp_veff(vext[xyz], comega)
        dn = self.apply_rf0(veff, comega)
        p_avg[iw] += np.dot(vext[xyz], dn)
    return p_avg/3.0

  polariz_inter_ave = comp_polariz_inter_ave

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

  def comp_dens_inter_along_Eext(self, comegas, Eext = np.array([1.0, 0.0, 0.0])):
    """ 
        Compute a the average interacting polarizability along the Eext direction
        for the frequencies comegas.
        
        Input Parameters:
            comegas (1D array, complex): the real part contains the frequencies at which the polarizability
                        should be computed. The imaginary part id the width of the polarizability define as self.eps
            Eext (1D xyz array, real): direction of the external field
            maxiter (integer): max number of iteration before to exit iteration loop in GMRES
        
        Other Calculated quantity:
            self.p_mat (complex array, dim: [3, 3, comega.size]): store the (3, 3) polarizability matrix 
                                [[Pxx, Pxy, Pxz],
                                 [Pyx, Pyy, Pyz],
                                 [Pzx, Pzy, Pzz]] for each frequency.
            self.dn (complex array, dim: [3, comegas.size, self.nprod]): store the density change
    """
    
    assert Eext.size == 3
    self.p_mat = np.zeros((3, 3, comegas.size), dtype=self.dtypeComplex)
    self.dn = np.zeros((3, comegas.size, self.nprod), dtype=self.dtypeComplex)

    Edir = Eext/np.dot(Eext, Eext)
    
    vext = np.transpose(self.moms1)
    for xyz, Exyz in enumerate(Edir):
      if Exyz == 0.0: continue

      for iw,comega in enumerate(comegas):
        veff = self.comp_veff(vext[xyz], comega)
        self.dn[xyz, iw, :] = self.apply_rf0(veff, comega)
            
    self.p_mat = np.einsum("jp,iwp->ijw", vext, self.dn)
