#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, dft, mcscf

'''
CASSCF can be called with UHF/UKS objects.

The UHF/UKS orbitals will be used as initial guess for CASSCF.  But the CASSCF
solver is the RHF-based CASSCF which assumes the degeneracy between alpha
orbitals and beta orbitals for the core orbitals.
'''

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz', symmetry=1)

mf = scf.UHF(mol)
mf.kernel()
#mf.analyze()
# 6 orbitals, 6 electrons
mc = mcscf.CASSCF(mf, 6, 6)
e = mc.kernel()[0]
print('CASSCF based on UHF, E = %.12f, ref = -109.075063732553' % e)

mf = scf.UKS(mol)
mf.kernel()
#mf.analyze()
# 6 orbitals, 6 electrons
mc = mcscf.CASSCF(mf, 6, 6)
e = mc.kernel()[0]
print('CASSCF based on UKS, E = %.12f, ref = -109.075063732553' % e)

