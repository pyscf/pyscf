#!/usr/bin/env python

"""CCSD(T) triples energies and lambda amplitudes for H2O/cc-pVDZ.

Compare triples corrections for distorted and symmetric geometries with
reference values, then solve the CCSD(T) lambda equations using the optimized
ccsd_t_lambda implementation and compare the amplitude norms.
"""

import numpy
from pyscf import gto, scf, cc
from pyscf.cc import ccsd_t, ccsd_t_lambda

# Distorted H2O without symmetry adaptation.
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.957 , .587)],
    [1 , (0.2,  .757 , .487)]]

mol.basis = 'ccpvdz'
mol.build()
rhf = scf.RHF(mol)
rhf.conv_tol = 1e-14
rhf.scf()
mcc = cc.CCSD(rhf)
mcc.conv_tol = 1e-14
mcc.ccsd()
e3a = ccsd_t.kernel(mcc, mcc.ao2mo())
print('Distorted H2O triples correction difference:', e3a - -0.0033300722704016289)

# Symmetric H2O with molecular symmetry enabled.
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -.757 , .587)],
    [1 , (0. ,  .757 , .587)]]
mol.symmetry = True

mol.basis = 'ccpvdz'
mol.build()
rhf = scf.RHF(mol)
rhf.conv_tol = 1e-14
rhf.scf()
mcc = cc.CCSD(rhf)
mcc.conv_tol = 1e-14
mcc.ccsd()
e3a = ccsd_t.kernel(mcc, mcc.ao2mo())
print('Symmetry-adapted H2O triples correction difference:', e3a - -0.003060022611584471)

# CCSD(T) lambda equations for symmetric H2O.
mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = 'cc-pvdz'
mol.build()
rhf = scf.RHF(mol)
rhf.conv_tol = 1e-16
rhf.scf()

mcc = cc.CCSD(rhf)
mcc.conv_tol = 1e-12
ecc, t1, t2 = mcc.kernel()

conv, l1, l2 = ccsd_t_lambda.kernel(mcc, mcc.ao2mo(), t1, t2, tol=1e-8)
print('CCSD(T) lambda converged:', conv)
print('Lambda singles norm difference:', numpy.linalg.norm(l1)-0.013575484203926739)
print('Lambda doubles norm difference:', numpy.linalg.norm(l2)-0.22029981372536928)
