from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
from pyscf.nao.m_overlap_am import overlap_am
from conv_yzx2xyz import conv_yzx2xyz
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
oref = mol.intor_symmetric('cint1e_ovlp_sph')

sv = system_vars_c(gto=mol)

start1 = timer()
over = comp_overlap_coo(sv, overlap_funct=overlap_am).tocsr()
oxyz = conv_yzx2xyz.conv_yzx2xyz_2d(mol, over)
end1 = timer()

print(abs(oref-oxyz).max())
print(abs(oref-oxyz).argmax())

print(end1-start1)
