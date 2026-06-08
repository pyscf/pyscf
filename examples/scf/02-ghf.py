#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Examples of generalized Hartree–Fock (GHF) calculations.

Each molecular orbital in GHF is represented in a two-component basis (alpha
beta components). Typically, the GHF orbital coefficient matrix (mo_coeff) has
dimension 2N x 2N, where N is the number of AOs (mol.nao). The alpha
components are stored in the upper block (mo_coeff[:N]) and the beta components
are stored in the lower block (mo_coeff[N:]).

This example demonstrates

1. Real-valued GHF calculations.
2. Complex-valued GHF calculations.
3. Breaking the Sz spin symmetry in GHF.
'''

from pyscf import gto, scf

mol = gto.M(
    atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1  # = 2S = spin_up - spin_down
)

#
# 1. Real-valued GHF
#
# For a non-relativistic Hamiltonian with only real-valued integrals, the GHF
# solution is normally real. In this case, the converged GHF solution is usually
# equivalent to the corresponding UHF solution. Although the Hamiltonian itself
# does not couple the alpha and beta spin channels, degeneracy can lead to the
# rotation within the alpha and beta orbitals, leading to spin mixed spin
# components in the GHF orbitals.
#
mf = mol.GHF()
mf.kernel()

#
# 2. Complex-valued GHF
#
# GHF can also optimize complex-valued orbitals. One way to obtain such a
# solution is to start the SCF procedure from a complex density matrix.
#
mf = mol.GHF()
dm = mf.get_init_guess() + 0j
dm[0,0] += .05j
dm[1,1] -= .05j
mf.kernel(dm0=dm)

#
# 3. Breaking the Sz spin symmetry
#
# Spin-orbit coupling (SOC) operator can mix alpha and beta components. The SOC
# term can be enabled by the X2C relativistic calculations with GHF (see also
# examples/x2c/03-x2c_ghf.py) or the configuration mf.with_soc in the case of
# ECP-SOC calculations (see also examples/scf/44-soc_ecp.py).
#
mf = mol.GHF().x2c()
mf.run()

#
# A non-zero alpha-beta block in the density matrix explicitly couples the two
# spin sectors. Such initial guesses can drive the SCF procedure toward a
# solution that breaks the Sz symmetry, even without an explicit SOC term in the
# Hamiltonian.
#
mf = mol.GHF()
dm = mf.get_init_guess() + 0j
nao = mol.nao
dm[:nao,nao:] = 0.05j
dm[nao:,:nao] = -0.05j
mf.kernel(dm0=dm)
