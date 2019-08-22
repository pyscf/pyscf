from __future__ import print_function, division
import sys
from copy import copy
import numpy as np
from numpy import require, zeros_like
from scipy.linalg import blas

from timeit import default_timer as timer

from pyscf.data.nist import HARTREE2EV
from pyscf.nao.chi0_matvec import chi0_matvec
from pyscf.nao.m_blas_wrapper import spmv_wrapper
from pyscf.nao.m_pack2den import pack2den_u

    
class tddft_iter(chi0_matvec):
  """ 
    Iterative TDDFT a la PK, DF, OC JCTC
    
    Input Parameters:
    -----------------
        kw: keywords arguments:
            * tddft_iter_tol (real, default: 1e-3): tolerance to reach for 
                            convergency in the iterative procedure.
            * tmp_fname (string, default None): temporary file to save polarizability
                            at each frequency. Can be a life saver for large systems.
  """

  def __init__(self, **kw):

    self.load_kernel = load_kernel = kw['load_kernel'] if 'load_kernel' in kw else False
    self.maxiter = kw['maxiter'] if 'maxiter' in kw else 1000
    self.tddft_iter_tol = kw['tddft_iter_tol'] if 'tddft_iter_tol' in kw else 1e-3
    self.res_method = kw["res_method"] if "res_method" in kw else "both"
    assert self.tddft_iter_tol>1e-6

    # better to check input before to initialize calculations
    chi0_matvec.__init__(self, **kw)
    if self.scipy_ver < 1 and self.res_method != "both":
        import warnings
        warnings.warn("scipy.__version__ < 1, the res_method both will be used!")

    self.xc_code_mf = copy(self.xc_code)
    self.xc_code = xc_code = kw['xc_code'] if 'xc_code' in kw else self.xc_code

    self.matvec_ncalls = 0

    if not hasattr(self, 'pb'):
      print(__name__, 'no pb?')
      print(__name__, kw.keys())
      return
      
    self.spmv = spmv_wrapper
    if self.scipy_ver > 0:
      if self.dtype == np.float32: self.spmv = blas.sspmv
      elif self.dtype == np.float64: self.spmv = blas.dspmv
      else: raise ValueError("dtype can be only float32 or float64")

    xc = xc_code.split(',')[0]

    if load_kernel:
      self.load_kernel_method(**kw)

      if self.nspin==1:
        self.ss2kernel = [[self.kernel]]
      elif self.nspin==2:
        self.ss2kernel = [[self.kernel,self.kernel], [self.kernel,self.kernel]]

      if xc!='RPA' and self.nspin!=1: raise RuntimeError('not sure it would work')

    else:
      self.kernel,self.kernel_dim = self.pb.comp_coulomb_pack(dtype=self.dtype) # Lower Triangular
      assert self.nprod==self.kernel_dim,"{} {}".format(self.nprod,self.kernel_dim)
      
      if self.nspin==1:
        self.ss2kernel = [[self.kernel]]
      elif self.nspin==2:
        self.ss2kernel = [[self.kernel,self.kernel], [self.kernel,self.kernel]]
        
      # List of POINTERS !!! of kernel [[(up,up), (up,dw)], [(dw,up), (dw,dw)]] TAKE CARE!!!
      
      if xc=='RPA' or xc=='HF': 
        pass
      elif xc=='LDA' or xc=='GGA': 
        if self.nspin==1:
          self.comp_fxc_pack(kernel=self.kernel, **kw)
        elif self.nspin==2:
          kkk = self.comp_fxc_pack(**kw) + self.kernel
          self.ss2kernel = [[kkk[0], kkk[1]], [kkk[1],kkk[2]]]
          for s in range(self.nspin):
            for t in range(self.nspin): assert self.ss2kernel[s][t].dtype==self.dtype

      else:
        print(' xc_code', xc_code, xc, xc_code.split(','))
        raise RuntimeError('unkn xc_code')

    if self.verbosity>0 : print(__name__,'\t====> self.xc_code:', self.xc_code)


  def load_kernel_method(self, kernel_fname, kernel_format="npy", kernel_path_hdf5=None, **kw):
      """ Loads from file and initializes .kernel field... Useful? Rewrite?"""
    
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
    return vxc_pack(self, deriv=2, ao_log=self.pb.prod_log, **kw)

  def comp_veff(self, vext, comega=1j*0.0, x0=None):
    """ This computes an effective field (scalar potential) given the external scalar potential """
    from scipy.sparse.linalg import LinearOperator
    nsp = self.nspin*self.nprod
    assert len(vext)==nsp, "{} {}".format(len(vext), nsp)
    self.comega_current = comega
    veff_op = LinearOperator((nsp,nsp), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)

    if self.res_method == "relative" or self.res_method == "absolute":
      from pyscf.nao.m_lgmres import lgmres
      resgm, info = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, 
        requirements='C'), x0=x0, tol=self.tddft_iter_tol, maxiter=self.maxiter, res=self.res_method)
    elif self.res_method == "both":
      # use the non-modified lgmres scipy version
      from scipy.sparse.linalg import lgmres
      resgm, info = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, 
        requirements='C'), x0=x0, tol=self.tddft_iter_tol, maxiter=self.maxiter)
    else:
        raise ValueError("wrong input for res_method")

    if info != 0: print("LGMRES Warning: info = {0}".format(info))
    return resgm

  def vext2veff_matvec(self, vin):
    self.matvec_ncalls+=1
    dn0 = self.apply_rf0(vin, self.comega_current)
    vcre,vcim = self.apply_kernel(dn0)
    return vin - (vcre + 1.0j*vcim)
  
  def vext2veff_matvec2(self, vin):
    self.matvec_ncalls+=1
    dn0 = self.apply_rf0(vin, self.comega_current)
    vcre,vcim = self.apply_kernel(dn0)
    return 1- (vin - (vcre + 1.0j*vcim))

  def apply_kernel(self, dn):
    if self.nspin==1:
      return self.apply_kernel_nspin1(dn)
    elif self.nspin==2:
      return self.apply_kernel_nspin2(dn)

  def apply_kernel_nspin1(self, dn):
    
    daux  = np.zeros(self.nprod, dtype=self.dtype)
    daux[:] = require(dn.real, dtype=self.dtype, requirements=["A","O"])
    vcre = self.spmv(self.nprod, 1.0, self.kernel, daux)
    
    daux[:] = require(dn.imag, dtype=self.dtype, requirements=["A","O"])
    vcim = self.spmv(self.nprod, 1.0, self.kernel, daux)
    return vcre,vcim

  def apply_kernel_nspin2(self, dn):

    vcre = np.zeros((2,self.nspin,self.nprod), dtype=self.dtype)
    daux = np.zeros((self.nprod), dtype=self.dtype)
    s2dn = dn.reshape((self.nspin,self.nprod))

    for s in range(self.nspin):
      for t in range(self.nspin):
        for ireim,sreim in enumerate(('real', 'imag')):
          daux[:] = require(getattr(s2dn[t], sreim), dtype=self.dtype, requirements=["A","O"])
          vcre[ireim,s] += self.spmv(self.nprod, 1.0, self.ss2kernel[s][t], daux)

    return vcre[0].reshape(-1),vcre[1].reshape(-1)


  def comp_polariz_inter_xx(self, comegas, tmp_fname=None):
    """  Compute the interacting polarizability along the xx direction  """
    pxx = np.zeros(comegas.shape, dtype=self.dtypeComplex)

    if tmp_fname is not None:
      assert isinstance(tmp_fname, str), "tmp_fname must be a string"

    vext = np.transpose(self.moms1)
    nww, eV = len(comegas), 27.2114
    for iw, comega in enumerate(comegas):
      if self.verbosity>0: print(iw, nww, comega.real*HARTREE2EV)
      veff = self.comp_veff(vext[0], comega)
      dn = self.apply_rf0(veff, comega)
      pxx[iw] = np.dot(vext[0], dn)
      if tmp_fname is not None:
        tmp = open(tmp_fname, "a")
        tmp.write("{0}   {1}   {2}\n".format(comega.real, pxx[iw].real,
                                                          pxx[iw].imag))
        tmp.close() # Need to open and close the file at every freq, otherwise
                    # tmp is written only at the end of the calculations, therefore,
                    # it is useless

    return pxx

  def comp_polariz_inter_ave(self, comegas, tmp_fname=None, **kw):
    """  Compute average interacting polarizability  """
    
    verbosity = kw['verbosity'] if 'verbosity' in kw else self.verbosity
    sh = comegas.shape if hasattr(comegas, 'shape') else (len(comegas))
    p_avg = np.zeros(sh, dtype=self.dtypeComplex)

    if tmp_fname is not None:
      assert isinstance(tmp_fname, str), "tmp_fname must be a string"

    nww = len(comegas)
    for iw, comega in enumerate(comegas):
      for xyz in range(3):
        vext = np.concatenate([self.moms1[:,xyz] for s in range(self.nspin)])
        if verbosity>0: print(__name__, xyz, iw, nww, comega*HARTREE2EV)
        veff = self.comp_veff(vext, comega)
        dn = self.apply_rf0(veff, comega)
        p_avg[iw] += np.dot(vext, dn)

      if tmp_fname is not None:
        tmp = open(tmp_fname, "a")
        tmp.write("{0}   {1}   {2}\n".format(comega.real, p_avg[iw].real/3.0,
                                                              p_avg[iw].imag/3.0))
        tmp.close() # Need to open and close the file at every freq, otherwise
                    # tmp is written only at the end of the calculations, therefore,
                    # it is useless
    return p_avg/3.0


  polariz_inter_ave = comp_polariz_inter_ave
  

  def comp_dens_inter_along_Eext(self, comegas, Eext=np.array([1.0,0.0,0.0]),tmp_fname=None):
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

    if tmp_fname is not None:
        if not isinstance(tmp_fname, str):
            raise ValueError("tmp_fname must be a string")
        else:
            tmp_re = open(tmp_fname+".real", "a")
            tmp_re.write("# All atomic units\n")
            tmp_re.write("# w (Ha)    Pxx    Pxy    Pxz    Pyx    Pyy    Pyz    Pzx    Pzy    Pzz\n")
            
            tmp_im = open(tmp_fname+".imag", "a")
            tmp_im.write("# All atomic units\n")
            tmp_im.write("# w    Pxx    Pxy    Pxz    Pyx    Pyy    Pyz    Pzx    Pzy    Pzz\n")

   
    assert Eext.size == 3
    self.p_mat = np.zeros((3, 3, comegas.size), dtype=self.dtypeComplex)
    self.dn = np.zeros((3, comegas.size, self.nprod), dtype=self.dtypeComplex)

    Edir = Eext/np.dot(Eext, Eext)
    
    vext = np.transpose(self.moms1)
    nww, eV = len(comegas), 27.211386024367243
    
    if tmp_fname is not None:
        for iw,comega in enumerate(comegas):
            for xyz, Exyz in enumerate(Edir):
                if Exyz == 0.0: continue
                
                if self.verbosity>0: 
                    print("dir: {0}, w: {1}/{2}: ".format(xyz, iw, nww), comega*eV)
                veff = self.comp_veff(vext[xyz], comega)
                self.dn[xyz, iw, :] = self.apply_rf0(veff, comega)
            
                for xyzp, Exyzp in enumerate(Edir):
                    self.p_mat[xyz, xyzp, iw] = np.dot(vext[xyzp], self.dn[xyz, iw, :])
                
            tmp_re = open(tmp_fname+".real", "a")
            tmp_re.write("{0}   ".format(comega.real))

            tmp_im = open(tmp_fname+".imag", "a")
            tmp_im.write("{0}   ".format(comega.real))
            
            for i in range(3):
                for j in range(3):
                    tmp_re.write("{0}    ".format(self.p_mat[i, j, iw].real))
                    tmp_im.write("{0}    ".format(self.p_mat[i, j, iw].imag))
            tmp_re.write("\n")
            tmp_im.write("\n")
            tmp_re.close()  # Need to open and close the file at every freq, otherwise
                            # tmp is written only at the end of the calculations, therefore,
                            # it is useless
            tmp_im.close()  

    else:
        for xyz, Exyz in enumerate(Edir):
            if Exyz == 0.0: continue

            for iw,comega in enumerate(comegas):
                print(xyz, iw)
                if self.verbosity>0: 
                    print("dir: {0}/3, w: {1}/{2}: ".format(xyz, iw, nww), comega*eV)
                veff = self.comp_veff(vext[xyz], comega)
                self.dn[xyz, iw, :] = self.apply_rf0(veff, comega)
        
        self.p_mat = np.einsum("jp,iwp->ijw", vext, self.dn)
