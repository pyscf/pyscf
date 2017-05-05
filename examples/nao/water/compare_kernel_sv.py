from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_coulomb_am import coulomb_am
from pyscf.nao.m_pack2den import pack2den
from pyscf.nao.m_comp_coulomb_den import comp_coulomb_den
import numpy as np

sv = system_vars_c('siesta')

cref = pack2den(np.loadtxt("kernel_sv_savetxt.txt") )

cnao = comp_coulomb_den(sv, funct=coulomb_am)

print(cnao[0,0], cref[0,0], cref[0,0]/cnao[0,0])
print(cnao[1,1], cref[1,1], cref[1,1]/cnao[1,1])
print(cnao[-1,-1], cref[-1,-1], cref[-1,-1]/cnao[-1,-1])

