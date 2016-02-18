#!/usr/bin/env python

'''
User defined XC functional

See also `pyscf.dft.libxc.parse_xc` function
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
#   functional.
#   - If "," not appeared in string, the entire string is considered as
#     X functional.
#   - To neglect X functional (just apply C functional), leave blank in the
#     first part, eg description=',vwn' for pure VWN functional
# * The functional name can be placed in arbitrary order.  Two name needs to
#   be separated by operators + or -.  Blank spaces are ignored.
#   NOTE the parser only reads operators + - *.  / is not in support.
# * A functional name is associated with one factor.  If the factor is not
#   given, it is assumed equaling 1.
# * String "HF" stands for exact exchange (HF K matrix).  It is allowed to
#   put in C functional part.
# * Be careful with the libxc convention on GGA functional, in which the LDA
#   contribution is included.
#

mf = dft.RKS(mol)
mf.xc = 'HF*0.2 + .08*LDA + .72*B88, .81*LYP + .19*VWN'
e1 = mf.kernel()
print('E = %.15g  ref = -76.3832244350081' % e1)

#
# No correlation functional
#
mf.xc = '.2*HF + .08*LDA + .72*B88'
e1 = mf.kernel()
print('E = %.15g  ref = -75.9807850596666' % e1)

#
# If not given, the factor for each functional equals 1 by default.
#
mf = dft.RKS(mol)
mf.xc = 'b88,lyp'
e1 = mf.kernel()

mf = dft.RKS(mol)
mf.xc = 'b88*1,lyp*1'
eref = mf.kernel()
print('%.15g == %.15g' % (e1, eref))

