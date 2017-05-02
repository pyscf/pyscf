from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
from pyscf.nao.m_prod_log import dipole_check
#import numpy as np

label = 'siesta'
sv  = system_vars_c(label)
print(diag_check(sv))
print(overlap_check(sv))

prod_log = prod_log_c(sv, 1e-6)
print( overlap_check_prod_log(sv, prod_log, level=7) )
print( dipole_check(sv, prod_log, level=7) )

