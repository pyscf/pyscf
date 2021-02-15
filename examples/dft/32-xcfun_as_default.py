#!/usr/bin/env python

'''
Set the default XC functional library to XCFun (https://github.com/dftlibs/xcfun)
'''

from pyscf import gto, dft, grad

mol = gto.M(
    atom = '''
    F  0.   0.       0.
    H  0.   0.       1.587 ''',
    basis = 'ccpvdz')

#
# Scheme 1: Change ._numint of MF object for a single calculation.
#
mf = dft.RKS(mol)
mf._numint.libxc = dft.xcfun
mf.xc = 'b88,lyp'
mf.kernel()
mf.nuc_grad_method().run()

mf.xc = 'scan'
mf.kernel()

#
# Scheme 2: Change the default XC library globally.  All DFT calculations will
# call xcfun for XC functional values.
#
dft.numint.NumInt.libxc = dft.xcfun
mf = dft.RKS(mol)
mf.xc = 'b88,lyp'
mf.kernel()
mf.nuc_grad_method().run()

mf.xc = 'scan'
mf.kernel()
