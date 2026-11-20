#!/usr/bin/env python

'''
TDA and TDDFT linear response on an SCF object attached with SGX
(seminumerical exchange) integrals.

The response operator reuses the SCF J/K constructor, so the
seminumerical K is applied consistently in the ground state and in the
response. Excitation energies of the SGX ground state agree with the
exact-exchange results within the screening tolerance.
'''

from pyscf import gto
from pyscf import dft
from pyscf import sgx

mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = 'ccpvdz',
    verbose = 4,
)

# Exact HF exchange, for comparison
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()
mf.TDA().run()
mf.TDDFT().run()

# SGX-attached RKS object. Set xc before sgx_fit so that the auxiliary
# basis is chosen for the functional.
mf = sgx.sgx_fit(dft.RKS(mol).set(xc='b3lyp'))
mf.kernel()

# TDA with singlet and triplet excitations
mf.TDA().run()
mf.TDA().set(singlet=False).run()

# Full TDDFT (RPA) equations
mf.TDDFT().run()
