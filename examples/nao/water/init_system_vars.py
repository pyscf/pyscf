from pyscf.nao.m_system_vars import system_vars_c, diag_check
from pyscf.nao.m_siesta_xml_print import siesta_xml_print
from pyscf.nao.m_sv_diag import sv_diag
from pyscf.nao.m_sbt import sbt_c

import matplotlib.pyplot as plt

label = 'siesta'
sv  = system_vars_c(label)
print(diag_check(sv), type(sv.ao_log.rr), type(sv.ao_log.pp))
sbt = sbt_c(sv.ao_log.rr, sv.ao_log.pp)
print(dir(sbt))
