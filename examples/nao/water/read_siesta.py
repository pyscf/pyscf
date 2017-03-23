from pyscf.nao.m_siesta_eig import siesta_eig
from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
from pyscf.nao.m_siesta_hsx import siesta_hsx_c
from pyscf.nao.m_siesta_ion import siesta_ion_c
from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
from pyscf.nao.m_siesta_xml import siesta_xml
label = 'siesta'
print(siesta_eig(label)*27.2116, 'eV')
hsx = siesta_hsx_c(label, force_type=-1)

wfsx = siesta_wfsx_c(label)
print(wfsx.X[0,:,0,0,0])

ion_dict = siesta_ion_xml('O.ion.xml')
print( ion_dict.keys() )

siesta_dict = siesta_xml('siesta.xml')
