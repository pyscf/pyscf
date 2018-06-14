# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    %.vdata : all product vertex blocks packed into an one dimensional array
    %.i2inf : integer array iteration --> info (start in vdata, atom1,atom2,product center,start functions,finish functions)
  Examples:
  '''
  def __init__(self, pb):
    
    self.pb = pb  # "copy" of input data
    self.c2d = [] #  product Center -> list of the info Dictionaries
    for atom,sp in enumerate(pb.sv.atom2sp): 
      self.c2d.append({"atoms": (atom,atom), "pspecie": sp, "vertex": pb.prod_log.sp2vertex[sp], "ptype": 1})
    for ibp,[i,vertex] in enumerate(zip(pb.bp2info, pb.bp2vertex)): 
      self.c2d.append({"atoms": (i[0][0], i[0][1]), "pspecie": ibp, "vertex": vertex, "ptype": 2})

    ndpc = len(self.c2d)  # number of product centers in this vertex 
    self.c2t = np.array([self.c2d[c]["ptype"] for c in range(ndpc)]) # product Center -> product specie Type
    self.c2s = np.zeros( ndpc+1, np.int32 ) # product Center -> Start index of a product function in a global counting for this vertex
    for c in range(ndpc): self.c2s[c+1] = self.c2s[c] + self.c2d[c]["vertex"].shape[0]

    niter = sum(self.c2t)
    self.i2inf = np.zeros( (niter+1,11), np.int64 )
    atom2s = pb.sv.atom2s
    i = -1 # iteration in the loop over the whole vertex
    for c,[d,s,f,t] in enumerate(zip(self.c2d, self.c2s,self.c2s[1:], self.c2t)):
      if t==1:
        i = i + 1
        a1, a2 = d["atoms"][0],d["atoms"][1]
        self.i2inf[i+1, 0] = self.i2inf[i, 0] + d["vertex"].size
        self.i2inf[i,1:11] = a1,a2,c,atom2s[a1],atom2s[a2],s,atom2s[a1+1],atom2s[a2+1],f,0
        
      elif t==2:
        i = i + 1
        self.i2inf[i+1, 0] = self.i2inf[i, 0] + d["vertex"].size
        a1, a2 = d["atoms"][0],d["atoms"][1]
        self.i2inf[i+1, 0] = self.i2inf[i, 0] + d["vertex"].size
        self.i2inf[i,1:11] = a1,a2,c,atom2s[a1],atom2s[a2],s,atom2s[a1+1],atom2s[a2+1],f,0
        i = i + 1
        self.i2inf[i+1, 0] = self.i2inf[i, 0] + d["vertex"].size
        a1, a2 = d["atoms"][1],d["atoms"][0]
        self.i2inf[i+1, 0] = self.i2inf[i, 0] + d["vertex"].size
        self.i2inf[i,1:11] = a1,a2,c,atom2s[a1],atom2s[a2],s,atom2s[a1+1],atom2s[a2+1],f,1
      else:
        raise RuntimeError('wrong product center type?')
      
    self.vdata = np.zeros(self.i2inf[-1][0])
    for i in range(niter):
      s,f,c,tr = self.i2inf[i][0], self.i2inf[i+1][0],self.i2inf[i][3],self.i2inf[i][10]
      if tr==0:
        self.vdata[s:f] = self.c2d[c]["vertex"].reshape(f-s)
      elif tr==1:
        self.vdata[s:f] = einsum('pab->pba', self.c2d[c]["vertex"]).reshape(f-s)
      else:
        raise RuntimeError('!tr?')


#
#
#
if __name__=='__main__':
  from pyscf.nao import system_vars_c, prod_basis_c, vertex_loop_c
  from pyscf import gto
  import numpy as np
  from timeit import default_timer as timer
  import matplotlib.pyplot as plt
  
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  pb = prod_basis_c(sv)
  vl = vertex_loop_c(pb)
  print(dir(vl))
  print(vl.vdata.sum())
  
  
