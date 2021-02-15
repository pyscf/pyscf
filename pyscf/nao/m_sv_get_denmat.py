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
import numpy
from pyscf.nao.m_siesta_hsx_bloch_mat import siesta_hsx_bloch_mat
from timeit import default_timer as timer
#
#
#
def sv_get_denmat(sv, mattype='hamiltonian', prec=64, kvec=[0.0,0.0,0.0], spin=0):

  assert(type(prec)==int)

  mat = None
  mtl = mattype.lower()
  
  if(sv.hsx.is_gamma):

    mat = numpy.empty((sv.norbs,sv.norbs), dtype='float'+str(prec))
    if(mtl=='hamiltonian'):
      mat = sv.hsx.spin2h4_csr[spin].todense()
    elif(mtl=='overlap'):
      mat = sv.hsx.s4_csr.todense()
    else: 
      raise SystemError('!mattype')

  else:
    #t1 = timer()
    if(mtl=='hamiltonian'):
      mat = siesta_hsx_bloch_mat(sv.hsx.spin2h4_csr[spin], sv.hsx, kvec=kvec)
    elif(mtl=='overlap'):
      mat = siesta_hsx_bloch_mat(sv.hsx.s4_csr, sv.hsx, kvec=kvec)
    else:
      raise SystemError('!mattype')
    #t2 = timer(); print(t2-t1)
    
  return mat
