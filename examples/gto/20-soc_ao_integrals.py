#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto

'''
Integrals for spin-orbit coupling
'''

mol = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = 'ccpvdz'
)

# J Chem Phys, 122, 034107, Eq (2)
mat = 0
for atm_id in range(mol.natm):
    mol.set_rinv_orig(mol.coord_of_atm(atm_id))
    chg = mol.charge_of_atm(atm_id)
    mat += chg * mol.intor('cint1e_prinvxp_sph', 3)

# J Chem Phys, 122, 034107, Eq (3)
mat = mol.intor('cint2e_p1vxp1_sph', comp=3)

# spin-spin dipole-dipole coupling integrals
# Chem Phys 279, 133, Eq (1)
def ss(mol):
    n = mol.nao_nr()
    mat1 = mol.intor('cint2e_ip1ip2_sph', comp=9).reshape(3,3,n,n,n,n) # <nabla1 nabla2 | 1 2>
    mat2 =-mat1.transpose(0,1,2,3,5,4) # <nabla1 2 | 1 nabla2>
    mat3 =-mat2.transpose(1,0,3,2,4,5) # <1 nabla2 | nabla1 2>
    mat4 = mat1.transpose(0,1,3,2,5,4) # <1 2 | nabla1 nabla2>
    mat = mat1 - mat2 - mat3 + mat4
    s = numpy.array((((0, 1),
                      (1, 0)),
                     ((0, -1j),
                      (1j, 0)),
                     ((1, 0),
                      (0, -1)))) * .5
# wxyz are the spin indices, ijkl are the AO indicies
    mat = numpy.einsum('swx,tyz,stijkl->wxyzijkl', s[:,0,0], s[:,0,0], mat)
    return mat

