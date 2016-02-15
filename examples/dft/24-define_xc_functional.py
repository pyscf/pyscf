#!/usr/bin/env python

'''
User defined XC functional

See also `pyscf.dft.vxc.define_xc_` function
'''

from pyscf import gto
from pyscf import dft

mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587 ''',
    basis = 'ccpvdz')

#
# Function define_xc can parse the user defined XC functional, following the
# rules:
# * The given functional description must be a one-line string.
# * The functional description is case-insensitive.
# * The functional description string has two parts, separated by ",".  The
#   first part describes the exchange functional, the second is the correlation
#   functional.  If "," not appeared in string, entire string is considered as
#   X functional.  There is no way to neglect X functional (just apply C
#   functional)
# * The functional name can be placed in arbitrary order.  Two name needs to
#   be separated by operations + or -.  Blank spaces are ignored.
#   NOTE the parser only reads operators + - *.  / is not in support.
# * A functional name is associated with one factor.  If the factor is not
#   given, it is assumed equaling 1.
# * String "HF" stands for exact exchange (HF K matrix).  It is allowed to
#   put in C functional part.
# * Be careful with the libxc convention on GGA functional, in which the LDA
#   contribution is included.
#

mf = dft.RKS(mol)
b3lyp = '.2*HF + .08*LDA + .72*B88, .81*LYP + .19*VWN'
mf.define_xc_(b3lyp)
e1 = mf.kernel()
print('E = %.15g  ref = -76.3832244350081' % e1)

#
# No correlation functional
#
mf.define_xc_('.2*HF + .08*LDA + .72*B88')
e1 = mf.kernel()
print('E = %.15g  ref = -75.9807850596666' % e1)

#
# If not given, the factor for each functional equals 1 by default.
#
mf = dft.RKS(mol)
mf.define_xc_('b88,lyp')
e1 = mf.kernel()

mf = dft.RKS(mol)
mf.xc = 'b88,lyp'
eref = mf.kernel()
print('%.15g == %.15g' % (e1, eref))

