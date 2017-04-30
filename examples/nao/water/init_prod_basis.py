from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
from pyscf.nao.m_ao_eval import ao_eval_
import numpy as np
import matplotlib.pyplot as plt
import sys

label = 'siesta'
sv  = system_vars_c(label)
print(diag_check(sv))
print(overlap_check(sv))

prd_log = prod_log_c(sv.ao_log, 1e-6)
print( overlap_check_prod_log(prd_log) )

coords = np.zeros((5,3))
coords[0,0:3] = 0.4,0.5,0.5
coords[1,0:3] = 1.4,0.5,0.5
coords[2,0:3] = 0.3,1.5,0.5
coords[3,0:3] = 0.4,2.5,0.5
coords[4,0:3] = 0.4,3.5,0.5
ao_val = np.zeros((sv.ao_log.sp2norbs[0],coords.shape[0]))
rv = sv.atom2coord[0,:]
ao_eval_(sv.ao_log, rv, 0, coords, ao_val)
print(ao_val)

