from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
from pyscf.nao.m_tddft_iter_gpu import tddft_iter_gpu_c
#from pyscf.nao.m_sparse_blas import csrgemv # not working!
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
import scipy
if int(scipy.__version__[0]) > 0:
    scipy_ver = 1
else:
    scipy_ver = 0
    from pyscf.nao.m_blas_wrapper import spmv_wrapper

try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_tem_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='LDA,PZ',
          GPU=False, precision="single", load_kernel=False, 
          velec = np.array([1.0, 0.0, 0.0]), beam_offset = np.array([0.0, 0.0, 0.0]),
          dr = np.array([0.3, 0.3, 0.3]), freq=np.linspace(0.0, 0.367, 100),
          **kvargs):
    """ 
        EELS version of the Iterative TDDFT a la PK, DF, OC JCTC 

        Input Parameters:
        -----------------
            velec: xyz component of the electron velocity in atomic unit
            beam_offset: xyz components of the beam offset, must be orthogonal 
                    to velec in atomic unit
            dr: spatial resolution for the electron trajectory in atomic unit.
                Warning: This parameter influence the accuracy of the calculations.
                    if it is taken too large the results will be wrong.
            freq: Frequency range (in atomic unit), freq[0] must be 0.0!!
    """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_comp_dm import comp_dm

    assert tddft_iter_tol>1e-6
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here
    assert abs(np.dot(velec, beam_offset)) < 1e-8
    assert freq[0] == 0.0


    if precision == "single":
        self.dtype = np.float32
        self.dtypeComplex = np.complex64
        self.gemm = blas.sgemm
        if scipy_ver > 0:
            self.spmv = blas.sspmv
        else: 
            self.spmv = spmv_wrapper
    elif precision == "double":
        self.dtype = np.float64
        self.dtypeComplex = np.complex128
        self.gemm = blas.dgemm
        if scipy_ver > 0:
            self.spmv = blas.dspmv
        else: 
            self.spmv = spmv_wrapper
    else:
        raise ValueError("precision can be only single or double")

    self.rf0_ncalls = 0
    self.l0_ncalls = 0
    self.matvec_ncalls = 0
    self.tddft_iter_tol = tddft_iter_tol
    self.eps = tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    self.GPU = GPU

    # tem variables
    self.velec = velec
    self.beam_offset = beam_offset
    self.dr = dr
    self.freq = freq
    self.vnorm = np.sqrt(np.dot(velec, velec))
    self.vdir = velec/self.vnorm
    
    self.check_collision(sv.atom2coord)
    self.get_time_range()

    self.v_dab = pb.get_dp_vertex_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()
    self.cc_da = pb.get_da2cc_sparse(dtype=self.dtype, sparseformat=coo_matrix).tocsr()

    self.moms0,self.moms1 = pb.comp_moments(dtype=self.dtype)
    self.nprod = self.moms0.size


    if load_kernel:
        self.load_kernel(**kvargs)
    else:
        self.kernel,self.kernel_dim = pb.comp_coulomb_pack(dtype=self.dtype) # Lower Triangular Part of the kernel
        assert self.nprod==self.kernel_dim, "%r %r "%(self.nprod, self.kernel_dim)
        
        if xc_code.upper()!='RPA' :
          dm = comp_dm(sv.wfsx.x, sv.get_occupations())
          pb.comp_fxc_pack(dm, xc_code, kernel = self.kernel, dtype=self.dtype, **kvargs)

    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy


    self.calc_external_potential(self.sv, self.pb)
    import sys
    sys.exit()

    # probably unnecessary, require probably does a copy
    # problematic for the dtype, must there should be another option 
    #self.x  = np.require(sv.wfsx.x, dtype=self.dtype, requirements='CW')

    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)

    self.xocc = sv.wfsx.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = sv.wfsx.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?

    self.tddft_iter_gpu = tddft_iter_gpu_c(GPU, sv.wfsx.x[0, 0, :, :, 0], self.ksn2f, self.ksn2e,
            self.norbs, self.nfermi, self.nprod, self.vstart)

  def check_collision(self, atom2coord):
    """
    Check if the electron collide with an atom
    """

    R0 = -100.0*np.max(atom2coord)*self.vdir + self.beam_offset

    for atm in range(atom2coord.shape[0]):
        vec = R0 - atom2coord[atm, :]
        
        # unit vector to compare to vdir
        vec = abs(vec/np.sqrt(np.dot(vec, vec)))

        if np.sqrt(np.dot(vec-self.vdir, vec-self.vdir)) < 1e-6:
            mess = """
            Electron is collinding with atom {0}:
            velec = [{1:.3f}, {2:.3f}, {3:.3f}]
            beam_offset = [{4:.3f}, {5:.3f}, {6:.3f}]
            atom coord = [{7:.3f}, {8:.3f}, {9:.3f}]
            impact parameter = {10:.9f} > 1e-6
            """.format(atm, *self.velec, *self.beam_offset, 
                    *atom2coord[atm, :], np.sqrt(np.dot(vec, self.vdir)))

            raise ValueError(mess)

  def get_time_range(self):
      """
        Get the time and symmetric frequency range for the electron passing close
        to the particle. The tim e range is a symmetric array
        around 0.0. At t = 0, the electron is at its closest
        position from the molecule. This array will depend on the
        frequency range and the spatial precision dr.
        To respect the Fourier transform convention, the following
        relationshiip must be fulfill,

        N = 2*pi/(dw*dt)

        with N the number of element of t.
        N must be an odd number in order that t is symmetric
      """
      
      dt = np.min(self.dr)/self.vnorm
      dw = self.freq[1]-self.freq[0]

      N = int(2*np.pi/(dw*dt))
      if N % 2 == 0:
          N +=1

      wmax = 2.0*np.pi*(N-1)/(N*dt)/2.0
      
      self.freq_symm = np.arange(-wmax, wmax+dw, dw)
      
      tmax = (N-1)*dt/2
      self.time = np.arange(-tmax, tmax+dt, dt)


  def calc_external_potential(self, sv, pb):
      """
        Calculate the external potential created by a moving charge
      """
      from pyscf.nao.m_libnao import libnao
      from ctypes import POINTER, c_double, c_int, c_int64, c_float, c_int
      from pyscf.nao.m_ao_matelem import ao_matelem_c

      V_freq_real = np.zeros((self.nprod, self.freq.size), dtype=np.float32)
      V_freq_imag = np.zeros((self.nprod, self.freq.size), dtype=np.float32)

      aome = ao_matelem_c(sv.ao_log.rr, sv.ao_log.pp)
      nc = pb.npairs # sv.natm # ???
      nfmx = 2*sv.ao_log.jmx + 1 # ???
      jcut_lmult = nfmx # ???
      
#      libnao.calculate_potential_pb_test(sv.ao_log.rr.ctypes.data_as(POINTER(c_double)),
#              c_int(sv.ao_log.rr.size))
      libnao.calculate_potential_pb(sv.ao_log.rr.ctypes.data_as(POINTER(c_double)), 
              self.time.ctypes.data_as(POINTER(c_double)), self.freq.ctypes.data_as(POINTER(c_double)), 
              self.freq_symm.ctypes.data_as(POINTER(c_double)), self.velec.ctypes.data_as(POINTER(c_double)), 
              self.beam_offset.ctypes.data_as(POINTER(c_double)), 
              V_freq_real.ctypes.data_as(POINTER(c_double)), V_freq_imag.ctypes.data_as(POINTER(c_double)), 
              c_int(nfmx), c_int(sv.ao_log.jmx), c_int(jcut_lmult), 
              c_int(sv.ao_log.rr.size), c_int(self.time.size), 
              c_int(self.freq.size), c_int(self.nprod), c_int(nc))

      raise ValueError("Euh!!! check how to get nc, nfmx, jcut_lmult!!!")

  def load_kernel(self, kernel_fname, kernel_format="npy", kernel_path_hdf5=None, **kwargs):

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

  def apply_rf0(self, v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    no = self.norbs
    print("vKs = ", np.sum(abs(v.real)), np.sum(abs(v.imag)), v.shape, self.nprod)

    if self.GPU:
        return chi0_mv_gpu(self.tddft_iter_gpu, v, self.cc_da, self.v_dab, no, comega, self.dtype,
                    self.dtypeComplex)
    else:
        return chi0_mv(v, self.xocc, self.xvrt, self.ksn2e[0, 0, :], self.ksn2f[0, 0, :],
                    self.cc_da, self.v_dab, no, self.nfermi, self.nprod, self.vstart, comega, self.dtype,
                    self.dtypeComplex)


  def comp_veff(self, vext, comega=1j*0.0, x0=None):
    #from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    from scipy.sparse.linalg import lgmres, LinearOperator
    
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=self.dtypeComplex)
    resgm = lgmres(veff_op, np.require(vext, dtype=self.dtypeComplex, 
        requirements='C'), x0=x0, tol=self.tddft_iter_tol)
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

  def comp_polariz_xx(self, comegas, x0=False):
    """ 
        Compute interacting polarizability

        Inputs:
        -------
            comegas (complex 1D array): frequency range (in Hartree) for which the polarizability is computed.
                                     The imaginary part control the width of the signal.
                                     For example, 
                                     td = tddft_iter_c(...)
                                     comegas = np.arange(0.0, 10.05, 0.05) + 1j*td.eps
            x0 (boolean, optional): determine if a starting guess array should be use to
                                    guess the solution. if True, it will use the non-interacting 
                                    polarizability as guess.
        Output:
        -------
            polariz (complex 1D array): computed polarizability
            self.dn (complex 2D array): computed density change in prod basis
        
    """
    polariz = np.zeros_like(comegas, dtype=np.complex64)
    self.dn = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)
    
    for iw,comega in enumerate(comegas):
        if x0 == True:
            veff,info = self.comp_veff(self.moms1[:,0], comega, x0=self.dn0[iw, :])
        else:
            veff,info = self.comp_veff(self.moms1[:,0], comega, x0=None)

        self.dn[iw, :] = self.apply_rf0(veff, comega)
     
        polariz[iw] = np.dot(self.moms1[:,0], self.dn[iw, :])

    if self.tddft_iter_gpu.GPU:
        self.tddft_iter_gpu.clean_gpu()

    return polariz

  def comp_nonin(self, comegas):
    """ 
        Compute non-interacting polarizability

        Inputs:
        -------
            comegas (complex 1D array): frequency range (in Hartree) for which the polarizability is computed.
                                     The imaginary part control the width of the signal.
                                     For example, 
                                     td = tddft_iter_c(...)
                                     comegas = np.arange(0.0, 10.05, 0.05) + 1j*td.eps
        Output:
        -------
            pxx (complex 1D array): computed non-interacting polarizability
            self.dn0 (complex 2D array): computed non-interacting density change in prod basis
        
    """

    vext = np.transpose(self.moms1)
    pxx = np.zeros(comegas.shape, dtype=np.complex64)
    self.dn0 = np.zeros((comegas.shape[0], self.nprod), dtype=np.complex64)

    for iw, omega in enumerate(comegas):
        self.dn0[iw, :] = -self.apply_rf0(vext[0, :], omega)
 
        pxx[iw] = np.dot(self.dn0[iw, :], vext[0,:])
    return pxx
