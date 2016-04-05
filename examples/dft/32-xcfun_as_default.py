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
# Modifying _NumInt.libxc changes the default XC library globally.  All DFT
# calculations will call the xcfun to evaluate XC values.
#
dft.numint._NumInt.libxc = dft.xcfun
mf = dft.RKS(mol)
mf.xc = 'b88,lyp'
mf.kernel()
grad.RKS(mf).run()
