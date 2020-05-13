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
def eri3c(me, sp1,sp2,R1,R2, sp3,R3, **kvargs):
    """ Computes three-center Electron Repulsion Integrals.
    atomic orbitals of second species must contain the Hartree potentials of radial functions (product orbitals)"""
    from pyscf.nao.m_ao_matelem import build_3dgrid3c
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    #from pyscf.nao.m_ao_eval import ao_eval as ao_eval
    
    grids = build_3dgrid3c(me, sp1,sp2,R1,R2, sp3,R3, **kvargs)

    ao1 = grids.weights * ao_eval(me.aos[0], R1, sp1, grids.coords)
    ao2 = ao_eval(me.aos[0], R2, sp2, grids.coords)
    aoao = np.einsum('ar,br->abr', ao1, ao2)
    pbf = ao_eval(me.aos[1], R3, sp3, grids.coords)
    
    abp2eri = np.einsum('abr,pr->abp',aoao,pbf)
    return abp2eri


if __name__=="__main__":
  from pyscf.nao.m_system_vars import system_vars_c, ao_matelem_c
  from pyscf.nao.prod_log import prod_log as prod_log_c
  from pyscf.nao.m_eri3c import eri3c
  
  sv = system_vars_c(label='siesta')
  R0 = sv.atom2coord[0,:]
  
  prod_log = prod_log_c(ao_log=sv.ao_log)
  me_prod = ao_matelem_c(prod_log)
  vc = me_prod.coulomb_am(0, R0, 0, R0)
  eri_am = np.einsum('pab,pq->abq', prod_log.sp2vertex[0], vc)
  print( eri_am.shape )
  print( eri_am.sum(), eri_am.max(), np.argmax(eri_am), eri_am.min(), np.argmin(eri_am))
  
  vhpf = prod_log.hartree_pot()
  me = ao_matelem_c(sv.ao_log, vhpf)
  eri_ni = eri3c(me, 0, 0, R0, R0, 0, R0, level=4)
  
  print( eri_ni.shape )
  print( eri_ni.sum(), eri_ni.max(), np.argmax(eri_ni), eri_ni.min(), np.argmin(eri_ni))
  
