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

print(diag_check(sv))

sbt = sbt_c(sv.ao_log.rr, sv.ao_log.pp)
me = ao_matelem_c(sv.ao_log)
R1 = np.array([0.0000000000000000, -3.1143832781848038E-003,   0.0000000000000000])
R2 = np.array([1.4659276763729339,  1.1212154156359782,        0.0000000000000000])
oo = me.get_overlap(0, 1, R1, R2)

#print(oo.shape)
#print(oo)

#plt.plot(me.kk, np.log(abs(me.psi_log_mom[0,0,:])),
#  me.kk, me.psi_log_mom[0,1,:], 
#  me.kk, me.psi_log_mom[0,2,:],
#  me.kk, me.psi_log_mom[0,3,:],
#  me.kk, me.psi_log_mom[0,4,:])
#plt.xlim([0.0,10.0])
#plt.show()

