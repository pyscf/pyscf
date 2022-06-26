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
            H    0.   0.83    0.68
    ''',
    basis = 'ccpvdz',
)
#bl = 1.34765 # equil for exact HF
bl = 1.2
coord = bl / 2
mol = gto.M(atom='F 0. 0. -{coord}; F 0. 0. {coord}'.format(coord=coord), basis='ccpvdz')
# Direct K matrix for comparison
mf = scf.RHF(mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()
print(forces)

# Using SGX for J-matrix and K-matrix
mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.with_df.dfj = True
#mf.with_df.grids_level_f = 3
mf.kernel()
mf_grad = mf.nuc_grad_method()
forces = mf_grad.kernel()
print(forces)

