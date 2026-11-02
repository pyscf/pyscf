#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK builder of SCF method can be customized. This example shows how to
overwrite the default JK builder (function get_jk).
'''

import numpy
from pyscf import gto, scf, df

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
    vj_P = numpy.linalg.solve(int2c, rho)
    vj = numpy.einsum('ijP,P->ij', int3c, vj_P)

    vk = get_k(mol, dm)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
mf.get_jk = get_jk
mf.kernel()
print('Approximate HF total energy =', mf.e_tot)

# The direct JK builder accepts einsum-like contraction descriptions and
# avoids materializing four-index integrals. Multiple contractions can share
# one integral evaluation; derivative integrals return three Cartesian blocks.
from pyscf.scf import jk
dm = mf.make_rdm1()
vj, vk = jk.get_jk(mol, (dm, dm), ('ijkl,ji->kl', 'ijkl,jk->il'))
j_deriv = jk.get_jk(mol, dm, 'ijkl,lk->ij', intor='int2e_ip1',
                    aosym='s2kl', comp=3)
print('Direct J/K and derivative shapes', vj.shape, vk.shape, j_deriv.shape)
# Different molecules can supply the four integral slots, for example the
# Coulomb potential of mol's density evaluated in a second molecule's basis.
probe = gto.M(atom='He 3 0 0', basis='sto-3g')
j_cross = jk.get_jk((probe, probe, mol, mol), dm, 'ijkl,lk->ij')
print('Intermolecular Coulomb matrix', j_cross)
