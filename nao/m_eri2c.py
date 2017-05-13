from __future__ import division, print_function
import numpy as np

#
#
#
def eri2c(me, sp1,R1,sp2,R2, **kvargs):
    """ Computes two-center Electron Repulsion Integrals. This is normally done between product basis functions"""
    from pyscf.nao.m_ao_matelem import build_3dgrid
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    
    grids = build_3dgrid(me, sp1,R1, sp2,R2, **kvargs)

    pf = grids.weights * ao_eval(me.ao2,         R1, sp1, grids.coords)
    qv =                 ao_eval(me.ao2_hartree, R2, sp2, grids.coords)

    pq2eri = np.einsum('pr,qr->pq',pf,qv)
    return pq2eri


if __name__=="__main__":
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  from pyscf.nao.m_prod_log import prod_log_c
  from pyscf.nao.m_eri2c import eri2c
  
  sv = system_vars_c(label='siesta')
  R0 = sv.atom2coord[0,:]
  
  prod_log = prod_log_c(sv.ao_log)
  print(prod_log.sp2norbs)
  
  me_prod = ao_matelem_c(prod_log)
  vc_am = me_prod.coulomb_am(0, R0, 0, R0)
  print(vc_am.shape, vc_am.max(), vc_am.min())

  vc_ni = eri2c(me_prod, 0, R0, 0, R0, level=5)
  print(vc_ni.shape, vc_ni.max(), vc_ni.min())
  
  print(abs(vc_ni-vc_am).sum() / vc_am.size, abs(vc_ni-vc_am).max())
  
  
