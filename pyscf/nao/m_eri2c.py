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

from __future__ import division, print_function
import numpy as np

#
#
#
def eri2c(me, sp1,R1,sp2,R2, **kvargs):
    """ Computes two-center Electron Repulsion Integrals. This is normally done between product basis functions"""
    from pyscf.nao.m_ao_matelem import build_3dgrid
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    
    grids = build_3dgrid(me, sp1,np.array(R1), sp2,np.array(R2), **kvargs)

    pf = grids.weights * ao_eval(me.ao2,         np.array(R1), sp1, grids.coords)
    qv =                 ao_eval(me.ao2_hartree, np.array(R2), sp2, grids.coords)

    pq2eri = np.einsum('pr,qr->pq',pf,qv)
    return pq2eri


if __name__=="__main__":
  from pyscf.nao import system_vars_c, ao_matelem_c
  from pyscf.nao.prod_log import prod_log as prod_log_c
  from pyscf.nao.m_eri2c import eri2c
  
  sv = system_vars_c(label='siesta')
  R0 = sv.atom2coord[0,:]
  
  prod_log = prod_log_c(ao_log=sv.ao_log)
  print(prod_log.sp2norbs)
  
  me_prod = ao_matelem_c(prod_log)
  vc_am = me_prod.coulomb_am(0, R0, 0, R0)
  print(vc_am.shape, vc_am.max(), vc_am.min())

  vc_ni = eri2c(me_prod, 0, R0, 0, R0, level=5)
  print(vc_ni.shape, vc_ni.max(), vc_ni.min())
  
  print(abs(vc_ni-vc_am).sum() / vc_am.size, abs(vc_ni-vc_am).max())
  
  
