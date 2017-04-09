from pyscf.nao.m_system_vars import system_vars_c, diag_check
from pyscf.nao.m_siesta_xml_print import siesta_xml_print
from pyscf.nao.m_sbt import sbt_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_log_interp import log_interp
import numpy as np
import matplotlib.pyplot as plt
import sys

label = 'siesta'
sv  = system_vars_c(label)

print(diag_check(sv), dir(sv))

sbt = sbt_c(sv.ao_log.rr, sv.ao_log.pp)
me = ao_matelem_c(sv.ao_log)
print(dir(me))
R1 = sv.xml_dict["atom2coord"][0,:]
R2 = sv.xml_dict["atom2coord"][1,:]
oo = me.get_overlap(0, 1, R1, R2)

print(oo.shape)
print(oo)


#plt.plot(pp, gg)
#plt.xlim([0.0,10.0])
#plt.show()

