from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
from pyscf.nao.m_blas_wrapper import spmv_wrapper
from pyscf.nao.m_tddft_iter_gpu import tddft_iter_gpu_c

try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='LDA,PZ',
          GPU=False, precision="single", **kvargs):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_comp_dm import comp_dm
    import sys

    assert tddft_iter_tol>1e-6
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here

    if precision == "single":
        self.dtype = np.float32
        self.dtypeComplex = np.complex64
    elif precision == "double":
        self.dtype = np.float64
        self.dtypeComplex = np.complex128
    else:
        raise ValueError("precision can be only single or double")


    self.rf0_ncalls = 0
    self.l0_ncalls = 0
    self.matvec_ncalls = 0
    self.tddft_iter_tol = tddft_iter_tol
    self.eps = tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    self.v_dab = pb.get_dp_vertex_coo(dtype=self.dtype).tocsr()
    self.cc_da = pb.get_da2cc_coo(dtype=self.dtype).tocsr()
    self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)
    self.nprod = self.moms0.size
    self.kernel, self.kernel_dim = pb.comp_coulomb_pack(dtype=self.dtype)

    if xc_code.upper()!='RPA' :
      dm = comp_dm(sv.wfsx.x, sv.get_occupations())
      
      xc_pack = pb.comp_fxc_pack(dm, xc_code, dtype=self.dtype, **kvargs)
      self.kernel = self.kernel + xc_pack
      
    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy
    self.x  = np.require(sv.wfsx.x, dtype=self.dtype, requirements='CW')
    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)
    self.xocc = self.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?

    self.tddft_iter_gpu = tddft_iter_gpu_c(GPU, self.v_dab, self.ksn2f, self.ksn2e, 
            self.norbs, self.nfermi, self.vstart)

  def apply_rf0(self, v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    # np.require may perform a copy of v, is it really necessary??
    vdp = self.cc_da * np.require(v, dtype=np.complex64)
    no = self.norbs
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([no,no]))

    if self.tddft_iter_gpu.GPU:
        vdp = self.tddft_iter_gpu.apply_rf0_gpu(self.xocc, sab, comega)
    else:
        #
        # WARNING!!!!
        # nb2v is column major, while self.xvrt is row major
        #       What a mess!!
        nb2v = self.xocc*sab
        nm2v = blas.cgemm(1.0, nb2v, np.transpose(self.xvrt))
        
        if use_numba:
            div_eigenenergy_numba(self.ksn2e, self.ksn2f, self.nfermi,
                    self.vstart, comega, nm2v, self.ksn2e.shape[2])
        else:
            for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
              for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
                m = j+n+1-self.vstart
                nm2v[n,m] = nm2v[n,m] * (fn-fm) *\
                  ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

        nb2v = blas.cgemm(1.0, nm2v, self.xvrt)

        ab2v = blas.cgemm(1.0, np.transpose(self.xocc), nb2v).reshape(no*no)

        vdp = self.v_dab*ab2v

    return vdp*self.cc_da 

  def comp_veff(self, vext, comega=1j*0.0):
    from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)
    resgm = gmres_alias(veff_op, np.require(vext, dtype=self.dtypeComplex, 
        requirements='C'), tol=self.tddft_iter_tol)
    return resgm
  
  def vext2veff_matvec(self, v):
    self.matvec_ncalls+=1 
    
    chi0 = self.apply_rf0(v, self.comega_current)
    
    # For some reason it is very difficult to pass only one dimension
    # of an array to the fortran routines?? matvec[0, :].ctypes.data_as(POINTER(c_float))
    # is not working!!!

    # real part
    chi0_reim = np.require(chi0.real, dtype=self.dtype, requirements=["A", "O"])
    matvec_real = spmv_wrapper(1.0, self.kernel, chi0_reim)
    
    # imaginary part
    chi0_reim = np.require(chi0.imag, dtype=self.dtype, requirements=["A", "O"])
    matvec_imag = spmv_wrapper(1.0, self.kernel, chi0_reim)

    return v - (matvec_real + 1.0j*matvec_imag)

  def comp_polariz_xx(self, comegas):
    """ Polarizability """
    polariz = np.zeros_like(comegas, dtype=np.complex64)
    
    for iw,comega in enumerate(comegas):
      veff,info = self.comp_veff(self.moms1[:,0], comega)
      chi0 = self.apply_rf0( veff, comega )

      polariz[iw] = np.dot(self.moms1[:,0], chi0)

    if self.tddft_iter_gpu.GPU:
        self.tddft_iter_gpu.clean_gpu()

    return polariz

  def comp_nonin(self, comegas):
    """
        Non interacting polarizability
    """
    vext = np.transpose(self.moms1)
    pxx = np.zeros(comegas.shape, dtype=np.complex64)

    for iomega, omega in enumerate(comegas):
      chi0 = self.apply_rf0(vext[0,:], omega)
      pxx[iomega] =-np.dot(chi0, vext[0,:])
    return pxx
