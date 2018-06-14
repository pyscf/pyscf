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

import numpy
from scipy.linalg import eigh 
from pyscf.nao.m_sv_get_denmat import sv_get_denmat

#
#
#
def sv_diag(sv, kvec=[0.0,0.0,0.0], spin=0, prec=64):
  assert(type(prec)==int)
  h = sv_get_denmat(sv, mattype='hamiltonian', kvec=kvec, spin=spin, prec=prec)
  s = sv_get_denmat(sv, mattype='overlap', kvec=kvec, spin=spin, prec=prec)

  return( eigh(h, s) )
