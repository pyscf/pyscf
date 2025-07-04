#!/usr/bin/env python

'''
A few examples to run TDDFT calculations with frozen orbitals.
'''

from pyscf import gto, dft, tddft

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g',
    symmetry = True,
)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mytd = tddft.TDDFT(mf)
mytd.kernel()
mytd.analyze()

# Freeze the core
mytd = tddft.TDDFT(mf, frozen=1)
mytd.kernel()
mytd.analyze()

# Freeze high energy virtuals
mytd = tddft.TDDFT(mf, frozen=[9, 10])
mytd.kernel()
mytd.analyze()

# Freeze valence (CVS) to access core spectra
mytd = tddft.TDDFT(mf, frozen=[1, 2, 3, 4])
mytd.kernel()
mytd.analyze()

# UKS

mol.spin = 2
mol.symmetry = False
mol.build()
mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

# Freeze the core
mytd = tddft.TDDFT(mf, frozen=1)
mytd.kernel()
mytd.analyze()
