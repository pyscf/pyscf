#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
The TDDFT calculations by default use the same XC functional, grids, _numint
schemes as the ground state DFT calculations.  Different XC, grids, _numint
can be set in TDDFT.
'''

import copy
from pyscf import gto, dft, tddft

mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='6-31g*')
mf = dft.RKS(mol).run(xc='pbe0')

#
# A common change for TDDFT is to use different XC functional library.  For
# example, PBE0 is not supported by the default XC library (libxc) in the TDDFT
# calculation.  Changing to xcfun library for TDDFT can solve this problem
#
mf._numint.libxc = dft.xcfun
# PySCF-1.6.1 and newer supports the .TDDFT method to create a TDDFT
# object after importing tdscf module.
td = mf.TDDFT()
print(td.kernel()[0] * 27.2114)

#
# Overwriting the relevant attributes of the ground state mf object,
# the TDDFT calculations can be run with different XC, grids.
#
mf.xc = 'lda,vwn'
mf.grids.set(level=2).kernel(with_non0tab=True)
td = mf.TDDFT()
print(td.kernel()[0] * 27.2114)

#
# Overwriting the ground state SCF object is unsafe.  A better solution is to
# create a new fake SCF object to hold different XC, grids parameters.
#
from pyscf.dft import numint
mf = dft.RKS(mol).run(xc='pbe0')
mf1 = copy.copy(mf)
mf1.xc = 'lda,vwn'
mf1.grids = dft.Grids(mol)
mf1.grids.level = 2
mf1._numint = numint.NumInt()
mf1._numint.libxc = dft.xcfun
td = mf1.TDDFT()
print(td.kernel()[0] * 27.2114)
