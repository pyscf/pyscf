from pyscf.nao.m_system_vars import system_vars_c, diag_check
from pyscf.nao.m_siesta_xml_print import siesta_xml_print
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_local_vertex import local_vertex_c
import numpy as np
import matplotlib.pyplot as plt
import sys

label = 'siesta'
sv  = system_vars_c(label)

print(diag_check(sv))

me = ao_matelem_c(sv.ao_log)
oo = me.get_overlap_ap(0, 0, [0.0,0.0,0.0], [0.0,0.0,0.0])
print(sum(sum(oo)))

lv = local_vertex_c(sv.ao_log)
lv.get_local_vertex(0)




#plt.plot(me.kk, np.log(abs(me.psi_log_mom[0,0,:])),
#  me.kk, me.psi_log_mom[0,1,:], 
#  me.kk, me.psi_log_mom[0,2,:],
#  me.kk, me.psi_log_mom[0,3,:],
#  me.kk, me.psi_log_mom[0,4,:])
#plt.xlim([0.0,10.0])
#plt.show()

