#!/usr/bin/env python

'''
This example shows how to use P-junction screening in the SGX module.
P-junction screening (pjs in the module) allows the SGX method to
achieve linear scaling for large systems.
'''

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import sgx
import time
import os


# Stack some water molecules on top of each other.
# The density matrix is well-localized on the individual waters
num_water = 12
mol = gto.Mole()
atom0 = [["O" , (0. , 0.     , 0.)],
         [1   , (0. , -0.757 , 0.587)],
         [1   , (0. , 0.757  , 0.587)],]
atom = []
for i in range(num_water):
    atom = atom + [[z, (c[0] + i * 5, c[1], c[2])] for z, c in atom0]
mol.build(
    atom=atom,
    basis='def2-svp',
)

mf = dft.RKS(mol)
mf.xc = 'PBE'
mf = mf.density_fit()
mf.kernel()
dm = mf.make_rdm1()

mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.with_df.dfj = True
mf.build()
ts = time.monotonic()
en0 = mf.energy_tot(dm=dm)
mf.kernel()
t0 = time.monotonic() - ts
print('Without P-junction screening:', t0, 's')
print('Energy:', en0)

# Turn on P-junction screening. dfj must also be true.
mf = sgx.sgx_fit(scf.RHF(mol), pjs=True)
mf.with_df.sgx_tol_energy = 1e-9
mf.direct_scf_tol = 1e-13
mf.build()
ts = time.monotonic()
en1 = mf.energy_tot(dm=dm)
mf.kernel()
t1 = time.monotonic() - ts
print('With P-junction screening:', t1, 's')
print('Energy:', en1)
print('P-junction screening error:', abs(en1-en0))
print('P-junction screening speedup:', t0/t1)
