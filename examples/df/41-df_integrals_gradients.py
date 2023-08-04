#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to generate the analytical gradients for DF integrals.
'''

import numpy
from pyscf import gto, df, lib
mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')

# Generate auxmol to hold auxiliary basis.
# One g shell is added in the auxiliary basis to improve the accuracy of
# integral partial derivatives
auxbasis = {'H': 'ccpvdz-jk-fit',
            'F':('ccpvdz-jk-fit', [[4, (3., 1)]])}
auxmol = df.addons.make_auxmol(mol, auxbasis)
#auxbasis = df.aug_etb(mol, beta=1.7)
#auxmol = df.addons.make_auxmol(mol, auxbasis)

nao = mol.nao_nr()
naux = auxmol.nao_nr()
print('number of AOs', nao)
print('number of auxiliary basis functions', naux)

ao_offset = mol.offset_nr_by_atom()
aux_offset = auxmol.offset_nr_by_atom()
for i in range(mol.natm):
    sh0, sh1, ao0, ao1 = ao_offset[i]
    aux0, aux1 = aux_offset[i][2:]
    print('atom %d %s, shell range %s:%s, AO range %s:%s, aux-AO range %s:%s' %
          (i, mol.atom_symbol(i), sh0, sh1, ao0, ao1, aux0, aux1))

# (d/dX i,j|P)
int3c_e1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)

# (i,j|d/dX P)
int3c_e2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)

# (d/dX P|Q)
int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)

# (ij|P)
int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)

# (P|Q)
int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
int2c_inv = numpy.linalg.inv(int2c)

# Errors in partial derivatives
eri1_direct = mol.intor('int2e_ip1', aosym='s1', comp=3).reshape(3,nao,nao,nao,nao)
eri1_df = lib.einsum('xijp,pq,klq->xijkl', int3c_e1, int2c_inv, int3c)
print('max partial derivative error', abs(eri1_direct - eri1_df).max())

# Errors in full derivatives
eri1_direct = eri1_direct + eri1_direct.transpose(0,2,1,3,4)
eri1_direct = eri1_direct + eri1_direct.transpose(0,3,4,1,2)

eri1_df = eri1_df + eri1_df.transpose(0,2,1,3,4)
eri1_df += lib.einsum('xijp,pq,klq->xijkl', int3c_e2, int2c_inv, int3c)
eri1_df = eri1_df + eri1_df.transpose(0,3,4,1,2)
eri1_df -= lib.einsum('ijp,pq,xqr,rs,kls->xijkl', int3c, int2c_inv,
                      (int2c_e1+int2c_e1.transpose(0,2,1)), int2c_inv, int3c)
print('max full derivative error', abs(eri1_direct - eri1_df).max())
