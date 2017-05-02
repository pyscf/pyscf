from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
import numpy as np
from timeit import default_timer as timer

sv  = system_vars_c()

me = ao_matelem_c(sv)
R1 = sv.atom2coord[0]
R2 = sv.atom2coord[1]

start1 = timer()
overlap_am = me.overlap_am(0, 0, R1, R2)
end1 = timer()

start2 = timer()
overlap_ni = me.overlap_ni(0, 0, R1, R2, level=0)
end2 = timer()

print(abs(overlap_ni-overlap_am).sum()/overlap_am.size)
print(abs(overlap_ni-overlap_am).max())
print(overlap_ni[-1,-1], overlap_am[-1,-1])

print(end1-start1, end2-start2)
