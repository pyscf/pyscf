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
# We can assign D3, D4 keywords directly to the xc attribute
#
mf.xc = 'wb97x-d3bj'
mf.kernel()

# Alternatively, you can configure the dispersion correction manually, through
# the xc, nlc, disp attributes. The previous xc keyword 'wb97x-d3bj' is
# equivalent to the following settings
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.nlc = 0  # this will disable NLC correction.
mf.disp = 'd3bj'
mf.kernel()

# To disable the dispersion correction, you can simply reset disp
mf.disp = None
mf.kernel()

# DFTD3 and DFTD4 libraries require two parameters to control the dispersion
# computation, including which dispersion version to use (like d3, d4, d3bj,
# d3zero), and which XC type of dispersion to target at (like b3lyp, wb97, hf).
# DFT code will automatically employ the .xc attribute as the xc-type parameter.
#
mf.xc = 'b3lyp'
mf.disp = 'd3bj'

# You can combine DFT calculation with any kinds of dispersion corrections via the
# disp attribute.
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.nlc = False
mf.disp = 'd3bj'
mf.kernel()

mf = mol.HF()
mf.disp = 'd3bj'
mf.kernel()

# The combination of (xc, nlc, disp) typically falls into the following categories:
# 1. mf.xc, mf.nlc, mf.disp = 'xc-keyword-d3', '', None
#   nlc and disp are default values. NLC as well as dispersion computation is
#   based on the 'xc-keyword-d3'
mf = mol.KS()
mf.xc = 'wb97x-d3bj'
mf.kernel()

# 2. mf.xc, mf.nlc, mf.disp = 'xc-keyword', 0, 'd3'
#   Manually control the dispersion calculations, where nlc is muted, and disp
#   is set to a particular version of DFT-D3.
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.nlc = 0  # this will disable NLC correction.
mf.disp = 'd3'
mf.kernel() # equivalent to mol.KS(xc='wb97x-d3')

# 3. mf.xc, mf.nlc, mf.disp = 'xc-keyword', '', 'd3bj'
#   nlc is the default value. NLC computation is based on the xc value 'xc-keyword-d3'.
#   disp is computed with the specified version (d3bj)
mf = mol.KS()
mf.xc = 'wb97x-v'
mf.disp = 'd3bj'
mf.kernel() # Do both NLC and disp d3bj. You will receive a warning for the double counting of NLC.

# 4. mf.xc, mf.nlc, mf.disp = 'xc-keyword-d3', '', 'd3bj'
#   nlc is the default value. NLC computation is based on the xc value.
#   disp is computed with the specified version (d3bj). However, the specified
#   disp version is conflicted with the xc setting. You will receive
#   an error due to this conflict.
mf = mol.KS()
mf.xc = 'wb97x-d3'
mf.disp = 'd3bj'
mf.kernel() # Crash

