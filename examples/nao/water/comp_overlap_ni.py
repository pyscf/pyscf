from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c, diag_check, overlap_check
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
from pyscf.nao.m_ao_matelem import ao_matelem_c
import numpy as np

sv  = system_vars_c()
print(diag_check(sv))
print(overlap_check(sv))

me = ao_matelem_c(sv)
me.overlap_ni(0, 0, sv.atom2coord[0,:], sv.atom2coord[1,:])

