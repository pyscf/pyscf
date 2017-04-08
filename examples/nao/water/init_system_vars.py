from pyscf.nao.m_system_vars import system_vars_c, diag_check
from pyscf.nao.m_siesta_xml_print import siesta_xml_print
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
import numpy as np
import matplotlib.pyplot as plt

label = 'siesta'
sv  = system_vars_c(label)

print(diag_check(sv), dir(sv))

sbt = sbt_c(sv.ao_log.rr, sv.ao_log.pp)

me = ao_matelem_c(sv.ao_log)
print(dir(me))
oo = me.get_overlap(0,0, np.array([0.0,1.0,0.0]), np.array([0.0,0.0,0.0]))

print(oo[0,0])
print(oo)



#plt.plot(pp, gg)
#plt.xlim([0.0,10.0])
#plt.show()

