#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF wavefunction stability analysis
'''

from pyscf import gto, scf

mol = gto.M(atom='C 0 0 0; O 0 0 1.2', basis='631g*')
mf = scf.RHF(mol).run()
# Run stability analysis for the SCF wave function
mf.stability()

#
# This instability is associated to a symmetry broken answer.  When point
# group symmetry is used, the same wave function is stable.
#
mol = gto.M(atom='C 0 0 0; O 0 0 1.2', basis='631g*', symmetry=1)
mf = scf.RHF(mol).run()
mf.stability()

#
# If the SCF wavefunction is unstable, the stability analysis program will
# transform the SCF wavefunction and generate a set of initial guess (orbitals).
# The initial guess can be used to make density matrix and fed into a new SCF
# iteration (see 15-initial_guess.py).
#
mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')
mf = scf.UHF(mol).run()
mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

#
# Also the initial guess orbitals from stability analysis can be used as with
# second order SCF solver
#
mf = scf.newton(mf).run(mo1, mf.mo_occ)
mf.stability()
