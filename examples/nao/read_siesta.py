from pyscf.nao.m_siesta_eig import siesta_eig
from pyscf.nao.m_siesta_hsx import siesta_hsx_c

print(siesta_eig('siesta.EIG')*27.2116, 'eV')
hsx = siesta_hsx_c(fname='siesta.HSX', force_type=-1)
#print(hsx.H4.shape)
#print(hsx.S4.shape)
#print(hsx.X4.shape)
#print(hsx.row_ptr.shape)
#print(vars(hsx))
