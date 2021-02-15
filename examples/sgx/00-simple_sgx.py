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
mf = sgx.sgx_fit(scf.RHF(mol))
mf.kernel()

# Using RI for Coulomb matrix while K-matrix is constructed with COS-X method
mf.with_df.dfj = True
mf.kernel()
