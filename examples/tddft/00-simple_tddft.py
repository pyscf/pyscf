#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft, tddft

'''
A simple example to run TDDFT calculation.
'''

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
#mytd.nstates = 10
mytd.kernel()
print('Excitation energy (eV)', mytd.e * 27.2114)
