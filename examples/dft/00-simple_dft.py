#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, dft

'''
A simple example to run DFT calculation.

See pyscf/dft/vxc.py for the complete list of available XC functional
'''

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g',
    symmetry = True,
)

mydft = dft.RKS(mol)
#mydft.xc = 'lda,vwn'
#mydft.xc = 'lda,vwn_rpa'
#mydft.xc = 'b86,p86'
#mydft.xc = 'b88,lyp'
#mydft.xc = 'b97,pw91'
#mydft.xc = 'b3p86'
#mydft.xc = 'o3lyp'
mydft.xc = 'b3lyp'
mydft.kernel()

# Orbital energies, Mulliken population etc.
mydft.analyze()
