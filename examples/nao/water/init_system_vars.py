from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
import numpy as np

sv  = system_vars_c('siesta')
print(diag_check(sv))
print(overlap_check(sv))

print(dir(sv.ao_log))
