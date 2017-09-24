#!/usr/bin/env python

'''
Input a XC functional which is not defined in the Libxc or XcFun library.

See also
* dft.libxc for API of function eval_xc;
* dft.numint._NumInt class for its methods eval_xc, hybrid_coeff and _xc_type.
  These methods controls the XC functional evaluation;
* Example 24-custom_xc_functional.py to customize XC functionals using the
  functionals provided by Libxc or XcFun library.
'''

from pyscf import gto
from pyscf import dft

mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587 ''',
    basis = 'ccpvdz')

# half-half exact exchange and GGA functional
def hybrid_coeff(xc_code, spin=1):
    return 0.5
def rsh_coeff(xc_code, spin=1):
    return 0, 0, 0
def _xc_type(xc_code):
    return 'GGA'
def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    # A fictitious XC functional to demonstrate the usage
    rho0, dx, dy, dz = rho[:4]
    gamma = (dx**2 + dy**2 + dz**2)
    exc = .1 * rho0**2 + .02 * (gamma+.001)**.5
    vrho = .1 * 2 * rho0
    vgamma = .02 * .5 * (gamma+.001)**(-.5)
    vlapl = None
    vtau = None
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative
    return exc, vxc, fxc, kxc

mf = dft.RKS(mol)
mf._numint.hybrid_coeff = hybrid_coeff
mf._numint.rsh_coeff = rsh_coeff
mf._numint.eval_xc = eval_xc
mf._numint._xc_type = _xc_type
mf.xc = 'My XC'  # optional, only affect the output message
mf.verbose = 4
mf.kernel()
