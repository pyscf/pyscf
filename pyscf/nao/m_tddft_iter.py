from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import blas
from timeit import default_timer as timer
import sys

try:
    # try import gpu library
    from pyscf.nao.m_libnao import libnao_gpu
    from ctypes import POINTER, c_double, c_int64, c_float, c_int
    GPU_import = True
except:
    GPU_import = False


try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False


class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2, tddft_iter_broadening=0.00367493,
          nfermi_tol=1e-5, telec=None, nelec=None, fermi_energy=None, xc_code='LDA,PZ',
          GPU=False, **kvargs):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_comp_dm import comp_dm

    assert tddft_iter_tol>1e-6
    assert type(tddft_iter_broadening)==float
    assert sv.wfsx.x.shape[-1]==1 # i.e. real eigenvectors we accept here


    self.rf0_ncalls,self.l0_ncalls = 0
    self.matvec_ncalls = 0
    self.tddft_iter_tol = tddft_iter_tol
    self.eps = tddft_iter_broadening
    self.sv, self.pb, self.norbs, self.nspin = sv, pb, sv.norbs, sv.nspin
    self.v_dab = pb.get_dp_vertex_coo(dtype=np.float32).tocsr()
    self.cc_da = pb.get_da2cc_coo(dtype=np.float32).tocsr()
    self.moms0,self.moms1 = pb.comp_moments(dtype=np.float32)
    self.nprod = self.moms0.size
    t1 = timer()
    self.kernel = pb.comp_coulomb_den(dtype=np.float32)
    t2 = timer()
    #print("Time Hartree kernel: ", t2-t1)

    if xc_code.upper()!='RPA' :
      dm = comp_dm(sv.wfsx.x, sv.get_occupations())
      self.kernel = self.kernel + pb.comp_fxc_lil(dm, xc_code, dtype=np.float32, **kvargs).todense()

    self.telec = sv.hsx.telec if telec is None else telec
    self.nelec = sv.hsx.nelec if nelec is None else nelec
    self.fermi_energy = sv.fermi_energy if fermi_energy is None else fermi_energy
    self.x  = np.require(sv.wfsx.x, dtype=np.float32, requirements='CW')
    self.ksn2e = np.require(sv.wfsx.ksn2e, dtype=np.float32, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
    self.ksn2f = (3-self.nspin)*ksn2fd
    self.nfermi = np.argmax(ksn2fd[0,0,:]<nfermi_tol)
    self.vstart = np.argmax(1.0-ksn2fd[0,0,:]>nfermi_tol)
    self.xocc = self.x[0,0,0:self.nfermi,:,0]  # does python creates a copy at this point ?
    self.xvrt = self.x[0,0,self.vstart:,:,0]   # does python creates a copy at this point ?
    
    if GPU and GPU_import:
        self.GPU=True

        self.block_size = np.array([32, 32], dtype=np.int32) # threads by block
        self.grid_size = np.array([0, 0], dtype=np.int32) # number of blocks
        dimensions = [self.nfermi, self.ksn2f.shape[2]]
        for i in range(2):
            if dimensions[i] <= self.block_size[i]:
                self.block_size[i] = dimensions[i]
                self.grid_size[i] = 1
            else:
                self.grid_size[i] = dimensions[i]/self.block_size[i] + 1

        libnao_gpu.init_iter_gpu(self.x[0, 0, :, :, 0].ctypes.data_as(POINTER(c_float)), c_int64(self.norbs),
                self.ksn2e[0, 0, :].ctypes.data_as(POINTER(c_float)), c_int64(self.ksn2e[0, 0, :].size),
                self.ksn2f[0, 0, :].ctypes.data_as(POINTER(c_float)), c_int64(self.ksn2f[0, 0, :].size),
                c_int64(self.nfermi), c_int64(self.vstart))
    elif GPU and not GPU_import:
        raise ValueError("GPU lib failed to initialize!")
    else:
        self.GPU = False


  def apply_rf0(self, v, comega=1j*0.0):
    """ This applies the non-interacting response function to a vector (a set of vectors?) """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    self.rf0_ncalls+=1
    # np.require may perform a copy of v, is it really necessary??
    vdp = self.cc_da * np.require(v, dtype=np.complex64)
    no = self.norbs
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([no,no]))

    if self.GPU:

        nb2v = self.xocc*sab.real
        libnao_gpu.calc_nm2v_real(nb2v.ctypes.data_as(POINTER(c_float)))
        nb2v = self.xocc*sab.imag
        libnao_gpu.calc_nm2v_imag(nb2v.ctypes.data_as(POINTER(c_float)))

        libnao_gpu.calc_XXVV(c_double(comega.real), c_double(comega.imag),
                self.block_size.ctypes.data_as(POINTER(c_int)), self.grid_size.ctypes.data_as(POINTER(c_int)))

        ab2v = np.zeros([self.norbs*self.norbs], dtype=np.float32)

        libnao_gpu.calc_ab2v_imag(ab2v.ctypes.data_as(POINTER(c_float)))
        vdp = 1j*self.v_dab*ab2v

        libnao_gpu.calc_ab2v_real(ab2v.ctypes.data_as(POINTER(c_float)))
        vdp += self.v_dab*ab2v

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
    res = vdp*self.cc_da
    return res
    
  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting, four-particle Green's function to a two-particle vector (in particular to the dipole matrix elements) """
    assert sab.dim==2, "%r"%(dab.dim)
    assert all(sab.shape==self.norbs), "%r, %r, %r "%(sab.shape, self.norbs)
    self.l0_ncalls+=1
    no = self.norbs
    nb2v = self.xocc*sab
    nm2v = blas.cgemm(1.0, nb2v, np.transpose(self.xvrt))
        
    if use_numba:
      div_eigenenergy_numba(self.ksn2e, self.ksn2f, self.nfermi, self.vstart, comega, nm2v, self.ksn2e.shape[2])
    else:
      for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
        for j,[em,fm] in enumerate(zip(self.ksn2e[0,0,n+1:],self.ksn2f[0,0,n+1:])):
          m = j+n+1-self.vstart
          nm2v[n,m] = nm2v[n,m] * (fn-fm) *\
            ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )

    nb2v = blas.cgemm(1.0, nm2v, self.xvrt)

    ab2v = blas.cgemm(1.0, np.transpose(self.xocc), nb2v).reshape(no*no)
    return ab2v


  def comp_veff(self, vext, comega=1j*0.0):
    from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    
    """ This computes an effective field (scalar potential) given the external scalar potential """
    assert len(vext)==len(self.moms0), "%r, %r "%(len(vext), len(self.moms0))
    self.comega_current = comega
    veff_op = LinearOperator((self.nprod,self.nprod), matvec=self.vext2veff_matvec, dtype=np.complex64)
    resgm = gmres_alias(veff_op, np.require(vext, dtype=np.complex64, requirements='C'), tol=self.tddft_iter_tol)
    return resgm
  
  def vext2veff_matvec(self, v):
    self.matvec_ncalls+=1 
    return v - blas.cgemv(1.0, self.kernel, self.apply_rf0(v, self.comega_current))

  def comp_polariz_xx(self, comegas):
    """ Polarizability """
    polariz = np.zeros_like(comegas, dtype=np.complex64)
    for iw,comega in enumerate(comegas):
      veff,info = self.comp_veff(self.moms1[:,0], comega)
      polariz[iw] = np.dot(self.moms1[:,0], self.apply_rf0( veff, comega ))

    if self.GPU:
        libnao_gpu.clean_gpu()

    return polariz
