from __future__ import print_function, division
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_comp_overlap_coo import comp_overlap_coo
from pyscf.nao.m_comp_coulomb_den import comp_coulomb_den
from pyscf.nao.m_overlap_am import overlap_am
from pyscf.nao.m_coulomb_am import coulomb_am
from conv_yzx2xyz import conv_yzx2xyz
from pyscf import gto
import numpy as np
from timeit import default_timer as timer

mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvqz') # coordinates in Angstrom!
oref = mol.intor_symmetric('cint1e_ovlp_sph')
kin = mol.intor_symmetric('cint1e_kin_sph')
cref = 2*np.pi*np.linalg.inv(kin)
cref = np.einsum('ij,jk->ik', oref, np.einsum('ij,jk->ik', cref, oref))

sv = system_vars_c(gto=mol)

start1 = timer()
over = comp_overlap_coo(sv, funct=overlap_am).todense()
oxyz = conv_yzx2xyz.conv_yzx2xyz_2d(mol, over)
end1 = timer()

print(abs(oref-oxyz).max())
print(abs(oref-oxyz).argmax())
print('overlap runtime ', end1-start1)

start1 = timer()
cnao = comp_coulomb_den(sv, funct=coulomb_am)
cxyz = conv_yzx2xyz.conv_yzx2xyz_2d(mol, cnao)
end1 = timer()

print(cref[0,0], cxyz[0,0])
print(cref[1,1], cxyz[1,1])
print(cref[-1,-1], cxyz[-1,-1])

print(cref.shape, cxyz.shape)

print('coulomb runtime ', end1-start1)

