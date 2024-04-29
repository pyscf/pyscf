#!/usr/bin/env python

import pyscf
from pyscf.dft import KS

'''
D3 and D4 Dispersion.

This is a simplified dispersion interface to
d3 (https://github.com/dftd3/simple-dftd3) and
d4 (https://github.com/dftd4/dftd4) libraries.
This interface can automatically configure the necessary settings including
dispersion, xc, and nlc attributes of PySCF mean-field objects. However,
advanced features of d3 and d4 program are not available.

If you need to access more features d3 and d4 libraries, such as overwriting the
dispersion parameters, you can use the wrapper provided by the simple-dftd3 and
dftd4 libraries. When using these libraries, please disable the .disp attribute
of the underlying mean-field object, and properly set the .xc and .nlc attributes
following this example.
'''

mol = pyscf.M(
    atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
    basis = 'def2-tzvp',
)

#
# It is recommended to enable D3, D4 dispersion corrections through the KS class
# instantiation. The values of attributes nlc, disp, and xc of KS object are
# automatically configured in this way. Both the mol.KS method or pyscf.dft.RKS
# function can be used.
mf = mol.KS(xc='wb97x-d4')
#mf = mol.KS(xc='wb97m-d3bj)
#mf = mol.KS(xc='wb97x-d3bj)
#mf = mol.KS(xc='b3lyp-d4')
#mf = mol.KS(xc='b3lyp-d3bj')
#mf = mol.KS(xc='b3lyp-d3zero')
#mf = mol.KS(xc='b3lyp-d3bj2b')
#mf = mol.KS(xc='b3lyp-d3bjatm')
mf.kernel()

mf = KS(xc='wb97x-d3bj')
mf.kernel()

#
# Assigning the D3, D4 keywords directly to the xc attribute will lead to an
# error in XC functional parser.
#
mf.xc = 'wb97x-d3'
mf.kernel()

# Alternatively, you can configure the dispersion correction manually, through
# the xc, nlc, disp attributes.
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.nlc = False  # this will disable NLC correction.
mf.disp = 'd4'
mf.kernel()

# To disable the dispersion correction, you can simply set disp = None
mf.disp = None
mf.kernel()

# DFTD3 and DFTD4 libraries require two parameters to control the dispersion
# computation, including which dispersion version to use (like d3, d4, d3bj,
# d3zero), and which XC type of dispersion to target at (like b3lyp, wb97, hf).
# The two parameters can be configured in the disp attribute, separated by ",".
# If the combination of XC and dispersion version is not found, DFTD3 and DFTD4
# will employ the default parameters of dispersion corrections. Please refer the
# the database of DFTD3 and DFTD4 for the proper xc names
# DFTD3: https://github.com/dftd3/simple-dftd3/blob/main/assets/parameters.toml
# DFTD4: https://github.com/dftd4/dftd4/blob/main/assets/parameters.toml

mf.disp = 'd3,b3lyp'
mf.disp = 'd4,wb97m'
mf.disp = 'd3bj,hf'

# If the xc parameter is not specified in the disp, the DFT code will automatically
# employ the .xc attribute as the xc-type parameter.
mf.xc = 'b3lyp'
mf.disp = 'd3bj' # == 'd3bj,b3lyp'

# Please note that the second parameter of mf.disp can be different to mf.xc.
# This parameter is meant to be used by dftd3 and dftd4 program only.
# You can combine DFT calculation with any kinds of dispersion corrections via the
# disp attribute.
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.nlc = False
mf.disp = 'd3bj,b3lyp'
mf.kernel()

mf = mol.HF()
mf.disp = 'd3bj,b3lyp'
mf.kernel()
