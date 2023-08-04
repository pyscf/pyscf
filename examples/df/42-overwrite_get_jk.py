#!/usr/bin/env python

'''
This example shows a customized get_jk function that mixes different
treatments for J matrix (using DF integrals) and K matrix (using 4-index
ERIs).
'''

import numpy
import scipy.linalg
from pyscf import gto, scf
from pyscf import df

mol = gto.M(atom='Ne', basis='ccpvdz')
# HF reference with analytical two-electron integrals
mf = scf.RHF(mol).run()
print('reference HF total energy =', mf.e_tot)

# Use density fitting to compute Coulomb matrix
# Build auxmol to hold auxiliary fitting basis
auxbasis = 'ccpvdz-jk-fit'
auxmol = df.addons.make_auxmol(mol, auxbasis)
# (ij|P)
int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s1', comp=1)
# (P|Q)
int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)

# Mimic the exchange part
nao = mol.nao
eri = mol.intor('int2e').reshape(nao,nao,nao,nao)
def get_k(mol, dm, *args, **kwargs):
    return numpy.einsum('ijkl,jk->il', eri, dm)

# Redefine get_jk function. This function is used by HF/DFT object, to compute
# the effective potential.
def get_jk(mol, dm, *args, **kwargs):
    rho = numpy.einsum('ijP,ji->P', int3c, dm)
    vj_P = scipy.linalg.solve(int2c, rho)
    vj = numpy.einsum('ijP,P->ij', int3c, vj_P)

    vk = get_k(mol, dm)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
mf.get_jk = get_jk
mf.kernel()
print('Approximate HF total energy =', mf.e_tot)
