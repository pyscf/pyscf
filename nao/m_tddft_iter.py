from __future__ import print_function, division
import numpy as np
from scipy.sparse import csr_matrix
#from pyscf.nao import system_vars_c, prod_basis_c
from pyscf.nao import coulomb_am, comp_coulomb_den

class tddft_iter_c():

  def __init__(self, sv, pb, tddft_iter_tol=1e-2):
    """ Iterative TDDFT a la PK, DF, OC JCTC """
    assert tddft_iter_tol>1e-12
    assert len(sv.wfsx.x)==1 # i.e. real eigenvectors
    self.tddft_iter_tol = tddft_iter_tol
    
    self.sv = sv
    self.pb = pb
    self.v_dab = pb.get_dp_vertex_coo(dtype=np.float32).tocsr()
    self.cc_da = pb.get_da2cc_coo(dtype=np.float32).tocsr()
    self.x  = np.require(sv.wfsx.x, dtype=np.float32, requirements='CW')
    self.e  = np.require(sv.wfsx.ksn2e, dtype=np.float32, requirements='CW')
    self.moms0,self.moms1 = pb.comp_moments(dtype=np.float32)
    self.kernel = pb.comp_coulomb_pack(dtype=np.float32)
    self.n = sv.norbs

  def apply_rf0(self, v, omega=0.0, eps=0.00367493):
    """ This applies the non-interacting response function to a vector or a set of vectors """
    assert len(v)==len(self.moms0), "%r, %r "%(len(v), len(self.moms0))
    vdp = self.cc_da*v
    n = self.n
    sab = csr_matrix((np.transpose(vdp)*self.v_dab).reshape([n,n]))
    #print(sab.shape, sab.nnz, self.x.shape)

    #print(' tddft_iter ', vdp.shape, len(v), v.shape)
    #v_ab = self.vrtx_psum(vdp)
    #v_nb = self.x*v_ab
    #v_nm = self.x*v_nb
