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

import numpy as np

#
#
#
def _siesta2blanko_denvec(orb2m, vec, orb_sc2orb_uc=None):

  n,nreim = vec.shape

  if orb_sc2orb_uc is None:
    orb_sc2m = orb2m
  else:
    orb_sc2m = np.zeros_like(orb_sc2orb_uc)
    for orb_sc,orb_uc in enumerate(orb_sc2orb_uc): orb_sc2m[orb_sc] = orb2m[orb_uc]

  orb2ph = (-1.0)**orb_sc2m
  
  if(nreim==1):
    vec[:,0] = vec[:,0]*orb2ph[:]

  elif(nreim==2):

    #print(vec[0:3,:], ' vec')
    cvec = vec.view(dtype=np.complex64)
    #print(cvec[0:3], 'cvec', cvec.shape) # I expected cvec.shape = (n), but got (n,1)...
    cvec[:,0] = cvec[:,0] * orb2ph
    #print(cvec[0:3], ' cvec2')
    vec = cvec.view(dtype=np.float32)
    #print(vec[0:3], ' vec2')

    #raise RuntimeError('debug')

  else:
    raise SystemError('!nreim')

  return(0)
