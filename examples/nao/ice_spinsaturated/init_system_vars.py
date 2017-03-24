from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_siesta_xml_print import siesta_xml_print

label = 'siesta'

#siesta_xml_print(label)

sv = system_vars_c(label)

print(dir(sv.wfsx))
print(sv.wfsx.ksn2e)

print(sv.xml_dict.keys())
print(sv.xml_dict['fermi_energy'])
print(sv.xml_dict['ksn2e'].shape)
print(sv.xml_dict['k2xyzw'].shape)
print(sv.xml_dict['atom2coord'])
print(sv.xml_dict['sp2elem'])
print(sv.xml_dict['ucell'])
