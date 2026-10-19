#!/usr/bin/env python

'''
GKS for general spin-orbital Kohn-Sham method.

See also pyscf/examples/scf/02-ghf.py
'''

import numpy as np
from pyscf import gto, dft

#
# 1. real GKS
#
mol = gto.M(
    atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1,  # = 2S = spin_up - spin_down
    verbose = 4
)

mf = dft.GKS(mol)
mf.xc = 'camb3lyp'
mf.kernel()

#
# 2. complex GKS
#
mf = dft.GKS(mol)
mf.xc = 'camb3lyp'
dm = mf.get_init_guess() + 0j
dm[0,:] += .1j
dm[:,0] -= .1j
mf.kernel(dm0=dm)

# A symmetry-adapted GKS calculation also accepts complex spin-mixing
# densities.  Symmetrize the initial guess implicitly through SCF, then
# canonicalize the converged orbitals and inspect their spin character.
mol = gto.M(atom='H 0 0 0; H 0 0 1; O .5 .6 .2',
            basis='ccpvdz', symmetry=True)
mf = dft.GKS(mol)
mf.xc = 'b3lyp'
dm = mf.get_init_guess().astype(complex)
nao = mol.nao_nr()
rng = np.random.default_rng(12)
dm[:nao, nao:] = .1j * rng.random((nao, nao))
dm[nao:, :nao] = dm[:nao, nao:].T.conj()
mf.kernel(dm0=dm)
mo_energy, _ = mf.canonicalize(mf.mo_coeff, mf.mo_occ)
mf.analyze()
print('Canonical GKS HOMO/LUMO:', mo_energy[mol.nelectron-1:mol.nelectron+1])
print('Symmetry-adapted GKS <S^2>:', mf.spin_square())
