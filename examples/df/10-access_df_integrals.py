#!/usr/bin/env python

'''
This example shows how to generate the 3-center integrals and the 2-center
integral metric for the density-fitting method.
'''

import scipy
from pyscf import gto, df, lib

mol = gto.Mole()
mol.atom = '''
  C   0.    0.    0.
  O   0.    0.    1.3
'''
mol.basis = 'ccpvdz'
mol.build()

# Define the auxiliary fitting basis for 3-center integrals. Use the function
# make_auxmol to construct the auxiliary Mole object (auxmol) which will be
# used to generate integrals.
auxbasis = 'ccpvdz-jk-fit'
auxmol = df.addons.make_auxmol(mol, auxbasis)

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
ints_3c = df.incore.aux_e2(mol, auxmol)
ints_2c = auxmol.intor('int2c2e')

nao = mol.nao
naux = auxmol.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
df_coef = scipy.linalg.solve(ints_2c, ints_3c.reshape(nao*nao, naux).T)
df_coef = df_coef.reshape(naux, nao, nao)

df_eri = lib.einsum('ijP,Pkl->ijkl', ints_3c, df_coef)

# Now check the error of DF integrals wrt the normal ERIs
print(abs(mol.intor('int2e') - df_eri).max())
