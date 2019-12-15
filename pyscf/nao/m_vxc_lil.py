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
from numpy import array, int64, zeros, float64

#
#
#
def vxc_lil(self, **kw):
  """
    Computes the exchange-correlation matrix elements
    Args:
      sv : (System Variables), this must have arrays of coordinates and species, etc
    Returns:
      fxc,vxc,exc
  """
  from pyscf.nao.m_xc_scalar_ni import xc_scalar_ni
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from scipy.sparse import lil_matrix

  #dm, xc_code, deriv, ao_log=None, dtype=float64, **kvargs

  dm = kw['dm'] if 'dm' in kw else self.make_rdm1()
  kernel = kw['kernel'] if 'kernel' in kw else None
  ao_log = kw['ao_log'] if 'ao_log' in kw else self.ao_log
  xc_code = kw['xc_code'] if 'xc_code' in kw else self.xc_code
  kw.pop('xc_code',None)
  dtype = kw['dtype'] if 'dtype' in kw else self.dtype
  
  aome = ao_matelem_c(self.ao_log.rr, self.ao_log.pp, self, dm)
  me = aome.init_one_set(self.ao_log) if ao_log is None else aome.init_one_set(ao_log)
  atom2s = zeros((self.natm+1), dtype=int64)
  for atom,sp in enumerate(self.atom2sp): atom2s[atom+1]=atom2s[atom]+me.ao1.sp2norbs[sp]
  
  lil = [lil_matrix((atom2s[-1],atom2s[-1]), dtype=dtype) for i in range((self.nspin-1)*2+1)]

  for atom1,[sp1,rv1,s1,f1] in enumerate(zip(self.atom2sp,self.atom2coord,atom2s,atom2s[1:])):
    for atom2,[sp2,rv2,s2,f2] in enumerate(zip(self.atom2sp,self.atom2coord,atom2s,atom2s[1:])):
      blk = xc_scalar_ni(me,sp1,rv1,sp2,rv2,xc_code=xc_code,**kw)
      for i,b in enumerate(blk): lil[i][s1:f1,s2:f2] = b[:,:]

  return lil
