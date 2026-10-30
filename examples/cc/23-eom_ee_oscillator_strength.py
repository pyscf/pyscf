#!/usr/bin/env python

'''
Transition dipole moments and oscillator strengths of EOM-EE-CCSD singlet
states.

EOM-CC is not Hermitian. The transition moment from the ground state to state
n is evaluated with the right eigenvector R_n and the CCSD Lambda amplitudes,
the moment from state n to the ground state with the left eigenvector L_n.
The oscillator strength is f = 2/3 w <0|r|n>.<n|r|0>. The Lambda equations
and the left eigenvectors are solved on the first call.
'''

import pyscf

mol = pyscf.M(
    atom = '''
    O    0.   0.       0.
    H    0.   -0.757   0.587
    H    0.   0.757    0.587''',
    basis = 'aug-cc-pvdz',
    symmetry = True)
mf = mol.RHF().run()
mycc = mf.CCSD().run()

eom = mycc.EOMEESinglet()
e, v = eom.kernel(nroots=4)
f = eom.oscillator_strength()
for i, (ei, fi) in enumerate(zip(e, f)):
    print('Excited state %d  %10.6f eV  f = %.6f' % (i+1, ei * 27.211386, fi))

# Excitation energies, symmetries, oscillator strengths and leading single
# excitations. The leading excitations and transition dipoles are printed
# with verbose >= 4.
eom.verbose = 4
eom.analyze()

# The two transition dipole moments, <0|r|n> and <n|r|0>
dip_0n, dip_n0 = eom.transition_dipole()
print(dip_0n)
print(dip_n0)

# Transition density matrices of the first excited state, in the MO basis
dm_0n, dm_n0 = eom.eeccsd_trans_rdm1(eom.v[0], eom.v_left[0])
