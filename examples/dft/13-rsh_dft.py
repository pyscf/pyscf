#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Range-seperated functionals customization
'''

from pyscf import gto, dft

#
# The RSH functionals predefined in XC functional library
#
mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf.xc = 'CAMB3LYP'
mf.kernel()


#
# Customize the range seperation parameter omega
#
mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf.xc = 'CAMB3LYP'
mf.omega = 0.9
mf.kernel()


#
# Multiple ways to customize the entire RSH functional
# See also 24-custom_xc_functional.py 24-define_xc_functional.py
#
mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf.xc = 'RSH(0.33,0.65,-0.46) + 0.46*ITYH + .35*B88, VWN5*0.19 + LYP*0.81'
mf.kernel()

mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf.xc = '0.19*SR_HF(0.33) + 0.65*LR_HF(0.33) + 0.46*ITYH + .35*B88, VWN5*0.19 + LYP*0.81'
mf.kernel()

mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf.xc = 'RSH(0.9,0.65,-0.46) + 0.46*ITYH + .35*B88 + VWN5*0.19, LYP*0.81'
mf.kernel()

#
# If xcfun library was used CAM-B3LYP can be defined the following way
#
mol = gto.M(atom="H; F 1 1.", basis='631g')
mf = dft.RKS(mol)
mf._numint.libxc = dft.xcfun
mf.xc = 'RSH(0.33,0.65,-0.46) + 0.46*BECKESRX + 0.35*B88 + VWN5C*0.19 + LYPC*0.81'
mf.kernel()
