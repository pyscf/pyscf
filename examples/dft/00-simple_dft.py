#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import pyscf

'''
A simple example to run DFT calculation.

See also pyscf/dft/libxc.py and pyscf/dft/xcfun.py for the complete list of
available XC functionals.
'''

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g',
    symmetry = True,
)

mf = mol.KS()
#mf.xc = 'svwn' # shorthand for slater,vwn
#mf.xc = 'bp86' # shorthand for b88,p86
#mf.xc = 'blyp' # shorthand for b88,lyp
#mf.xc = 'pbe' # shorthand for pbe,pbe
#mf.xc = 'lda,vwn_rpa'
#mf.xc = 'b97,pw91'
#mf.xc = 'pbe0'
#mf.xc = 'b3p86'
#mf.xc = 'wb97x'
#mf.xc = '' or mf.xc = None # Hartree term only, without exchange
mf.xc = 'b3lyp'
mf.kernel()

# Orbital energies, Mulliken population etc.
mf.analyze()
