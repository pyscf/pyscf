#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Integrals for spin-orbit coupling

See also functions  make_h1_soc, direct_spin_spin in  pyscf/prop/zfs/uhf.py
'''

import numpy
from pyscf import gto
from pyscf import lib

mol = gto.M(
    verbose = 0,
    atom = 'C 0 0 0; O 0 0 1.5',
    basis = 'ccpvdz'
)

# J Chem Phys, 122, 034107, Eq (2)
mat = 0
for atm_id in range(mol.natm):
    mol.set_rinv_orig(mol.atom_coord(atm_id))
    chg = mol.atom_charge(atm_id)
    mat -= chg * mol.intor('int1e_prinvxp_sph')

# J Chem Phys, 122, 034107, Eq (3)
mat = mol.intor('int2e_p1vxp1_sph')  # (3,n*n,n*n) array, 3 for x,y,z components

# spin-spin dipole-dipole coupling integrals
# Chem Phys 279, 133, Eq (1)
def ss(mol):
    n = mol.nao_nr()
    mat1 = mol.intor('int2e_ip1ip2_sph').reshape(3,3,n,n,n,n) # <nabla1 nabla2 | 1 2>
    mat2 =-mat1.transpose(0,1,2,3,5,4) # <nabla1 2 | 1 nabla2>
    mat3 =-mat2.transpose(1,0,3,2,4,5) # <1 nabla2 | nabla1 2>
    mat4 = mat1.transpose(0,1,3,2,5,4) # <1 2 | nabla1 nabla2>
    mat = mat1 - mat2 - mat3 + mat4
# Fermi contact term
    h_fc = mol.intor('int4c1e').reshape(nao,nao,nao,nao) * (4*numpy.pi/3)
    mat[0,0] -= h_fc
    mat[1,1] -= h_fc
    mat[2,2] -= h_fc

    s = lib.PauliMatrices * .5
# wxyz are the spin indices, ijkl are the AO indicies
    alpha = 137.036
    fac = alpha ** 2 / 2
    mat = numpy.einsum('swx,tyz,stijkl->wxyzijkl', s, s, mat) * fac
    return mat

