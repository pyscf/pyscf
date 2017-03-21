from pyscf.nao.m_siesta_eig import siesta_eig
from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
from pyscf.nao.m_siesta_hsx import siesta_hsx_c

label = 'siesta'
print(siesta_eig(label)*27.2116, 'eV')
hsx = siesta_hsx_c(label, force_type=-1)

wfsx = siesta_wfsx_c(label)
print(wfsx.X[0,22,21,0,0])


#print(hsx.H4.shape)
#print(hsx.S4.shape)
#print(hsx.X4.shape)
#print(hsx.row_ptr.shape)
#print(vars(hsx))
