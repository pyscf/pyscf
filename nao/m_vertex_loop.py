from __future__ import print_function, division
import numpy as np
from numpy import einsum

#
#
#
class vertex_loop_c():
  '''
  Holder of dominant product vertices in the form which must be easy to deploy in iterative tddft or in Fock operator
  Args:
    instance of prod_basis_c
  Returns:
    
  Examples:
  '''
  def __init__(self, pb):
    
    self.pb = pb
    
    self.dpc2info = []
    for atom,sp in enumerate(pb.sv.atom2sp):
      self.dpc2info.append([atom,atom,sp,pb.prod_log.sp2vertex[sp].shape[0], 1])

    for ibp, bp in enumerate(pb.bp2info):
      self.dpc2info.append([bp[0][0], bp[0][1],ibp, bp[3].shape[1], 2])
    
    ndpc = len(self.dpc2info)
    self.dpc2s = np.zeros( ndpc+1, np.int32 )
    for dpc in range(ndpc): self.dpc2s[dpc+1] = self.dpc2s[dpc] + self.dpc2info[dpc][3]

    self.i2ccct = []
    for dpc,i in enumerate(self.dpc2info):
      if i[4]==1:
        self.i2ccct.append([i[0], i[1], dpc, 0])
      elif i[4]==2:
        self.i2ccct.append([i[0], i[1], dpc, 0])
        self.i2ccct.append([i[1], i[0], dpc, 1])
      else:
        raise RuntimeError('type is not 1 or 2')
    
    niter = len(self.i2ccct)
    self.i2s = np.zeros( niter+1, np.int64 )
    for i,ccct in enumerate(self.i2ccct):
      s1,f1 = pb.sv.atom2s[ccct[0]], pb.sv.atom2s[ccct[0]+1]
      s2,f2 = pb.sv.atom2s[ccct[1]], pb.sv.atom2s[ccct[1]+1]
      s3,f3 = self.dpc2s[ccct[2]],self.dpc2s[ccct[2]+1]
      self.i2s[i+1] = self.i2s[i] + (f3-s3)*(f2-s2)*(f1-s1)
    
    self.data = np.zeros(self.i2s[-1])
    for i,[ccct,s,f] in enumerate(zip(self.i2ccct,self.i2s,self.i2s[1:])):
      dpc = ccct[2]
      info = self.dpc2info[dpc]
      self.data[s:f] = 0.0
      
    print(self.i2ccct, self.i2s[-1])
#
#
#
if __name__=='__main__':
  from pyscf.nao import system_vars_c, prod_basis_c
  from pyscf.nao.m_vertex_loop import vertex_loop_c
  from pyscf import gto
  import numpy as np
  from timeit import default_timer as timer
  import matplotlib.pyplot as plt
  
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  pb = prod_basis_c(sv)
  vl = vertex_loop_c(pb)
  
