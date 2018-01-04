from __future__ import print_function, division
import numpy as np
from timeit import default_timer as timer
from pyscf.nao.chi0_matvec import chi0_matvec
from copy import copy

class tddft_iter(chi0_matvec):
  """ 
    Iterative TDDFT a la PK, DF, OC JCTC
    
    Input Parameters:
    -----------------
        kw: keywords arguments:
            * tddft_iter_tol (real, default: 1e-3): tolerance to reach for 
                            convergency in the iterative procedure.
  """

  def __init__(self, **kw):
    chi0_matvec.__init__(self, **kw)

    self.xc_code_mf = copy(self.xc_code)
    self.xc_code = xc_code = kw['xc_code'] if 'xc_code' in kw else self.xc_code
    self.load_kernel = load_kernel = kw['load_kernel'] if 'load_kernel' in kw else False
    self.maxiter = kw['maxiter'] if 'maxiter' in kw else 1000
    self.tddft_iter_tol = kw['tddft_iter_tol'] if 'tddft_iter_tol' in kw else 1e-3
    assert self.tddft_iter_tol>1e-6

    self.matvec_ncalls = 0

    if not hasattr(self, 'pb'):
      print('no pb?')
      return
      
    pb = self.pb

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

  def comp_polariz_inter_xx(self, comegas):
    """  Compute the interacting polarizability along the xx direction  """
    pxx = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    vext = np.transpose(self.moms1)
    for iw, comega in enumerate(comegas):
      veff = self.comp_veff(vext[0], comega)
      dn = self.apply_rf0(veff, comega)
      pxx[iw] = np.dot(vext[0], dn)
    return pxx

  def comp_polariz_inter_ave(self, comegas, **kw):
    """  Compute average interacting polarizability  """
    p_avg = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    verbosity = kw['verbosity'] if 'verbosity' in kw else self.verbosity
    vext = np.transpose(self.moms1)
    nww, eV = len(comegas), 27.2114
    for xyz in range(3):
      for iw, comega in enumerate(comegas):
        if verbosity>1: print(xyz, iw, nww, comega*eV)
        veff = self.comp_veff(vext[xyz], comega)
        dn = self.apply_rf0(veff, comega)
        p_avg[iw] += np.dot(vext[xyz], dn)
    return p_avg/3.0

  polariz_inter_ave = comp_polariz_inter_ave

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
