from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_siesta_xml_print import siesta_xml_print

label = 'siesta'

siesta_xml_print(label)

sv = system_vars_c(label)
#print(vars(sv))

