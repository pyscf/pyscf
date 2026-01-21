#!/usr/bin/env python

'''
This example shows how to use pseudo spectral integrals in SCF calculation.
Computes the gradients with density fitting and SGX and compares them.
'''

from pyscf import gto
from pyscf import scf
from pyscf import sgx


mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.83    0.68
    ''',
    basis = 'ccpvdz',
)
#bl = 1.34765 # equil for exact HF
auxbasis = 'def2-universal-jkfit'
bl = 1.6
coord = bl / 2
mol = gto.M(atom='F 0. 0. -{coord}; F 0. 0. {coord}'.format(coord=coord), basis='ccpvdz')
# Direct K matrix for comparison
DFJ = True
OPTK = True
LEVELF = 7
DEBUG = False

# RHF
print('\nRHF GRADIENTS')
mf = scf.RHF(mol)
if DFJ:
    mf = mf.density_fit(auxbasis=auxbasis, only_dfj=True)
    mf.with_df.auxbasis = auxbasis
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

# Using SGX for J-matrix and K-matrix
mf = sgx.sgx_fit(scf.RHF(mol), pjs=False, auxbasis=auxbasis)
mf.with_df.dfj = DFJ
mf.with_df.optk = OPTK
mf.with_df.grids_level_f = LEVELF
mf.with_df.debug = DEBUG
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

# RKS
print('\nRKS GRADIENTS')
mf = scf.RKS(mol)
mf.xc = 'PBE0'
if DFJ:
    mf = mf.density_fit(auxbasis=auxbasis, only_dfj=True)
    mf.with_df.auxbasis = auxbasis
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

mf = sgx.sgx_fit(scf.RKS(mol).set(xc='PBE0'), pjs=False, auxbasis=auxbasis)
mf.with_df.dfj = DFJ
mf.with_df.optk = OPTK
mf.with_df.grids_level_f = LEVELF
mf.with_df.debug = DEBUG
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

# UHF
print('\nUHF GRADIENTS')
mol = gto.M(atom='O 0.0 0.0 0.0; O 0.0 0.0 1.18', spin=2, basis='def2-svp')
mf = scf.UHF(mol)
if DFJ:
    mf = mf.density_fit(auxbasis=auxbasis, only_dfj=True)
    mf.with_df.auxbasis = auxbasis
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()
dm = mf.make_rdm1()

mf = sgx.sgx_fit(scf.UHF(mol), pjs=False, auxbasis=auxbasis)
mf.with_df.dfj = DFJ
mf.with_df.optk = OPTK
mf.with_df.grids_level_f = LEVELF
mf.with_df.debug = DEBUG
mf.kernel(dm0=dm)
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

# UKS
print('\nUKS GRADIENTS')
mf = scf.UKS(mol)
mf.xc = 'PBE0'
if DFJ:
    mf = mf.density_fit(auxbasis=auxbasis, only_dfj=True)
    mf.with_df.auxbasis = auxbasis
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()
dm = mf.make_rdm1()

mf = sgx.sgx_fit(scf.UKS(mol).set(xc='PBE0'), pjs=False, auxbasis=auxbasis)
mf.with_df.dfj = DFJ
mf.with_df.optk = OPTK
mf.with_df.grids_level_f = LEVELF
mf.with_df.debug = DEBUG
mf.kernel(dm0=dm)
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()

