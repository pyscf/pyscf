#!/usr/bin/env python

'''
This example shows how to use pseudo spectral integrals in SCF calculation.
'''

from pyscf import gto
from pyscf import scf
from pyscf import sgx
mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = 'ccpvdz',
)
# Using SGX for J-matrix and K-matrix
mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.kernel()

# Using RI for Coulomb matrix while K-matrix is constructed with COS-X method
mf.with_df.dfj = True
mf.kernel()

# Turn on P-junction screening
mf.with_df.pjs = True
mf.kernel()

# Use direct J-matrix evaluation (slow, primarily for testing purposes)
mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.with_df.direct_j = True
mf.with_df.dfj = False
mf.kernel()

