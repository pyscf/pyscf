#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run TDDFT calculation.
'''

from pyscf import gto, scf, dft, tddft

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
mytd.analyze()

# PySCF-1.6.1 and newer supports the .TDDFT method to create a TDDFT
# object after importing tdscf module.
from pyscf import tddft
mytd = mf.TDDFT().run()
mytd = mol.RHF().run().TDHF().run()
