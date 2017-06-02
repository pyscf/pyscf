from __future__ import print_function, division

from pyscf.nao import system_vars_c, ao_matelem_c, conv_yzx2xyz_c, comp_overlap_coo
from pyscf.nao.m_overlap_am import overlap_am
from pyscf.nao.m_overlap_ni import overlap_ni

from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvqz') # coordinates in Angstrom!
oref = mol.intor_symmetric('cint1e_ovlp_sph')
sv = system_vars_c(gto=mol)

start1 = timer()
over = comp_overlap_coo(sv, funct=overlap_ni, level=4).todense()
oxyz = conv_yzx2xyz_c(mol).conv_yzx2xyz_2d(over)
end1 = timer()

print(abs(oref-oxyz).max())
print(abs(oref-oxyz).argmax())
print('overlap runtime ', end1-start1)
