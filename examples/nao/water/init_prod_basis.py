from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
import numpy as np
import matplotlib.pyplot as plt
import sys

label = 'siesta'
sv  = system_vars_c(label)
print(diag_check(sv))
print(overlap_check(sv))

prd_log = prod_log_c(sv.ao_log, 1e-6)

print( overlap_check_prod_log(prd_log) )
