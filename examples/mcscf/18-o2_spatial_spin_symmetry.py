#!/usr/bin/env python

'''
A small O2 CASSCF calculation, in which spin and spatial symmetry should be
carefully chosen.
'''

from pyscf import gto, scf, mcscf, fci

#
# Starting from a triplet SCF calculation
#
mol = gto.M(atom='O; O 1 1.2', basis='ccpvdz', spin=2, symmetry=1)
mf = scf.RHF(mol).run()
norb = 4
nelec = 6
mc = mcscf.CASSCF(mf, norb, nelec)
try:
    mc.kernel()
except IndexError as e:
    print('When symmetry is enabled, FCI solver optimize the wfn of A1g symmetry by default.')
    print('In this CAS space, wfn of A1g symmetry does not exist.  This leads to error')
    print(e)

# Ground state is A2g
mc.fcisolver.wfnsym = 'A2g'
mc.kernel()

#
# Starting from a singlet SCF calculation
#
# SCF calculation for singlet state does not converge if mol.symmetry is
# enabled.  Starting from a symmetry-broken SCF initial guess, the CASSCF
# solver can converge to the solution with correct symmetry, with more
# iterations.
mol = gto.M(atom='O; O 1 1.2', basis='ccpvdz', spin=0)
mf = scf.RHF(mol).run()
nalpha = 4
nbeta = 2
mc = mcscf.CASSCF(mf, norb, (nalpha,nbeta))
mc.kernel()

#
# Print out the largest CI coefficients
#
for c, deta, detb in fci.addons.large_ci(mc.ci, norb, (nalpha, nbeta),
                                         tol=.01, return_strs=False):
    print(deta.tolist(), detb.tolist(), c)
