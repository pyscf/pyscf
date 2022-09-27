#!/usr/bin/env python

'''
A simple example to run CAM-B3LYP TDDFT calculation.

You need to switch to xcfun library if you found an error like
NotImplementedError: libxc library does not support derivative order 2 for  camb3lyp
    This functional derivative is supported in the xcfun library.
    The following code can be used to change the libxc library to xcfun library:

        from pyscf.dft import xcfun
        mf._numint.libxc = xcfun
'''

import pyscf

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = '6-31g(d,p)',
    symmetry = True,
)

mf = mol.RKS()
mf.xc= 'camb3lyp'
mf.run()

# Note you need to switch to xcfun library for cam-b3lyp tddft
mf._numint.libxc = pyscf.dft.xcfun
mytd = mf.TDDFT()
mytd.kernel()
