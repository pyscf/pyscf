import pyscf.nao.m_siesta_eig as eig
import pyscf.nao.m_siesta_hsx as hsx

print(eig.siesta_eig('siesta.EIG'))
print(hsx.siesta_hsx(fname='siesta.HSX', force_type=-1))

