from pyscf.nao.m_system_vars import system_vars_c
import sys

label = 'siesta'
sv  = system_vars_c(label)
assert sv.norbs == 23
