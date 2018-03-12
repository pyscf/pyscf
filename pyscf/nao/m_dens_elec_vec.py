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
from numpy import require, zeros, linalg, compress
from scipy.spatial.distance import cdist
from timeit import default_timer as timer

def dens_elec_vec(sv, crds, dm):
  """ Compute the electronic density using vectorized oprators """
  from pyscf.nao.m_rsphar_vec import rsphar_vec as rsphar_vec_python

  assert crds.ndim==2  
  assert crds.shape[-1]==3  
  nc = crds.shape[0]
  
  lmax = sv.ao_log.jmx
  for ia,[ra,sp] in enumerate(zip(sv.atom2coord,sv.atom2sp)):
    lngs = cdist(crds, sv.atom2coord[ia:ia+1,:])
    bmask = lngs<sv.ao_log.sp2rcut[sp]
    crds_selec = compress(bmask[:,0], crds, axis=0)
    t1 = timer()
    rsh1 = rsphar_vec_python(crds_selec, lmax)
    t2 = timer(); print(t2-t1); t1 = timer()
        
  return 0
