from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs
import math
  
def chi0_mv(self, dvin, comega=1j*0.0, dnout=None):
    """
        Apply the non-interacting response function to a vector
        Input Parameters:
        -----------------
            self : tddft_iter or tddft_tem class
            sp2v : vector describing the effective perturbation [spin*product] --> value
            comega: complex frequency
    """
    if dnout is None: dnout = np.zeros_like(dvin, dtype=self.dtypeComplex)

    sp2v  = dvin.reshape((self.nspin,self.nprod))
    sp2dn = dnout.reshape((self.nspin,self.nprod))
    
    for s in range(self.nspin):
      vdp = csr_matvec(self.cc_da, sp2v[s].real)  # real part
      sab = (vdp*self.v_dab).reshape((self.norbs,self.norbs))
    
      nb2v = self.gemm(1.0, self.xocc[s], sab)
      nm2v_re = self.gemm(1.0, nb2v, self.xvrt[s].T)
    
      vdp = csr_matvec(self.cc_da, sp2v[s].imag)  # imaginary
      sab = (vdp*self.v_dab).reshape((self.norbs, self.norbs))
      
      nb2v = self.gemm(1.0, self.xocc[s], sab)
      nm2v_im = self.gemm(1.0, nb2v, self.xvrt[s].T)

      vs,nf = self.vstart[s],self.nfermi[s]
    
      if self.use_numba:
        self.div_numba(self.ksn2e[0,s], self.ksn2f[0,s], nf, vs, comega, nm2v_re, nm2v_im)
      else:
        for n,(en,fn) in enumerate(zip(self.ksn2e[0,s,:nf], self.ksn2f[0,s,:nf])):
          for m,(em,fm) in enumerate(zip(self.ksn2e[0,s,vs:],self.ksn2f[0,s,vs:])):
            nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
            nm2v = nm2v * (fn - fm) * \
              ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
            nm2v_re[n, m] = nm2v.real
            nm2v_im[n, m] = nm2v.imag

        for n in range(vs+1,nf): #padding m<n i.e. negative occupations' difference
          for m in range(n-vs):  nm2v_re[n,m],nm2v_im[n,m] = 0.0,0.0

      nb2v = self.gemm(1.0, nm2v_re, self.xvrt[s]) # real part
      ab2v = self.gemm(1.0, self.xocc[s].T, nb2v).reshape(self.norbs*self.norbs)
      vdp = csr_matvec(self.v_dab, ab2v)
      chi0_re = vdp*self.cc_da

      nb2v = self.gemm(1.0, nm2v_im, self.xvrt[s]) # imag part
      ab2v = self.gemm(1.0, self.xocc[s].T, nb2v).reshape(self.norbs*self.norbs)
      vdp = csr_matvec(self.v_dab, ab2v)    
      chi0_im = vdp*self.cc_da
      
      sp2dn[s] = chi0_re + 1.0j*chi0_im
      
    return dnout

#
#
#

def chi0_mv_gpu(self, v, comega=1j*0.0):
#        tddft_iter_gpu, v, cc_da, v_dab, no,
#        comega=1j*0.0, dtype=np.float32, cdtype=np.complex64):
# check with nspin=2
    """
        Apply the non-interacting response function to a vector using gpu for
        matrix-matrix multiplication
    """
    assert self.nspin==1
    
    if self.dtype != np.float32:
        print(self.dtype)
        raise ValueError("GPU version only with single precision")

    vext = np.zeros((v.shape[0], 2), dtype = self.dtype, order="F")
    vext[:, 0] = v.real
    vext[:, 1] = v.imag

    # real part
    vdp = csr_matvec(self.cc_da, vext[:, 0])
    sab = (vdp*self.v_dab).reshape([self.norbs, self.norbs])

    self.td_GPU.cpy_sab_to_device(sab, Async = 1)
    self.td_GPU.calc_nb2v_from_sab(reim=0)
    # nm2v_real
    self.td_GPU.calc_nm2v_real()


    # start imaginary part
    vdp = csr_matvec(self.cc_da, vext[:, 1])
    sab = (vdp*self.v_dab).reshape([self.norbs, self.norbs])
    self.td_GPU.cpy_sab_to_device(sab, Async = 2)


    self.td_GPU.calc_nb2v_from_sab(reim=1)
    # nm2v_imag
    self.td_GPU.calc_nm2v_imag()

    self.td_GPU.div_eigenenergy_gpu(comega)

    # real part
    self.td_GPU.calc_nb2v_from_nm2v_real()
    self.td_GPU.calc_sab(reim=0)
    self.td_GPU.cpy_sab_to_host(sab, Async = 1)

    # start calc_ imag to overlap with cpu calculations
    self.td_GPU.calc_nb2v_from_nm2v_imag()

    vdp = csr_matvec(self.v_dab, sab)
    
    self.td_GPU.calc_sab(reim=1)

    # finish real part 
    chi0_re = vdp*self.cc_da

    # imag part
    self.td_GPU.cpy_sab_to_host(sab)

    vdp = csr_matvec(self.v_dab, sab)
    chi0_im = vdp*self.cc_da
#    ssum_re = np.sum(abs(chi0_re))
#    ssum_im = np.sum(abs(chi0_im))
#    if math.isnan(ssum_re) or math.isnan(ssum_im):
#      print(__name__)
#      print('comega ', comega)
#      print(v.shape, v.dtype)
#      print("chi0 = ", ssum_re, ssum_im)
#      print("sab = ", np.sum(abs(sab)))
#      raise RuntimeError('ssum == np.nan')


    return chi0_re + 1.0j*chi0_im

