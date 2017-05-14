from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
from pyscf.nao.m_prod_log import dipole_check
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
from pyscf.nao.m_overlap_am import overlap_am
from pyscf.nao.m_overlap_ni import overlap_ni
import matplotlib.pyplot as plt

#import numpy as np

label = 'siesta'
sv  = system_vars_c(label)
print(diag_check(sv))
print(overlap_check(sv))

prod_log = prod_log_c(sv.ao_log, 1e-6)
print(prod_log.sp2norbs)
#sp = 0
#for j,ff in zip(prod_log.sp_mu2j[sp], prod_log.psi_log[sp]):
  #if j>0 :
    #plt.plot(prod_log.rr, ff, '--', label=str(j))
  #else:
    #plt.plot(prod_log.rr, ff, '-', label=str(j))

#plt.xlim([0.0,0.5])
#plt.legend()
#plt.show()



print( overlap_check_prod_log(prod_log, level=7) )
print( dipole_check(sv, prod_log, level=7) )

me = ao_matelem_c(prod_log)
#pp2o_ni = comp_overlap_coo(sv, prod_log, overlap_funct=overlap_ni, level=5 ).toarray()
#print('ni')
#print(pp2o_ni.sum())

pp2o_am = comp_overlap_coo(sv, prod_log, funct=overlap_am).toarray()
print('am')
print(pp2o_am.sum())


