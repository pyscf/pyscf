from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
from pyscf.nao.m_overlap_am import overlap_am
from pyscf.nao.m_prod_log import prod_log_c
from pyscf.nao.m_prod_log import overlap_check as overlap_check_prod_log
from pyscf.nao.m_prod_log import dipole_check
from conv_yzx2xyz import conv_yzx2xyz
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvtz') # coordinates in Angstrom!
oref = mol.intor_symmetric('cint1e_ovlp_sph')
sv = system_vars_c(gto=mol)

start1 = timer()
over = comp_overlap_coo(sv, funct=overlap_am).todense()
oxyz = conv_yzx2xyz.conv_yzx2xyz_2d(mol, over)
end1 = timer()

print(abs(oref-oxyz).max())
print(abs(oref-oxyz).argmax())
print('overlap runtime ', end1-start1)

prod_log = prod_log_c(sv.ao_log, 1e-2)

print( overlap_check_prod_log(prod_log, level=3) )
print( dipole_check(sv, prod_log, level=3) )

print(prod_log.sp2norbs, 'sp2norbs')
print(prod_log.sp2nmult, 'sp2nmult')
for sp,mu2j in enumerate(prod_log.sp_mu2j):
  for mu,j in enumerate(mu2j):
    print(sp, mu, j)
