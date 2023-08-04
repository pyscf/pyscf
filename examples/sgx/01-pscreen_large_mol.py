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

mol = gto.M(atom='a12.xyz', basis='sto-3g')

mf = dft.RKS(mol)
mf.xc = 'PBE'
mf.kernel()
dm = mf.make_rdm1()

mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.with_df.dfj = True
mf.build()
ts = time.monotonic()
en0 = mf.energy_tot(dm=dm)
t0 = time.monotonic() - ts
print('Without P-junction screening:', t0, 's')
print('Energy:', en0)

# Turn on P-junction screening. dfj must also be true.
mf.with_df.pjs = True
# Set larger screening tolerance to demonstrate speedup.
mf.direct_scf_tol = 1e-10
mf.build()
ts = time.monotonic()
en1 = mf.energy_tot(dm=dm)
t1 = time.monotonic() - ts
print('With P-junction screening:', t1, 's')
print('Energy:', en1)
print('P-junction screening error:', abs(en1-en0))
print('P-junction screening speedup:', t0/t1)
