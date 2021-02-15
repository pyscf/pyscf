#!/usr/bin/env python

'''
Input a XC functional which was not implemented in pyscf.

See also
* The definition of define_xc_ function in pyscf/dft/libxc.py
* pyscf/dft/libxc.py for API of function eval_xc;
* dft.numint.NumInt class for its methods eval_xc, hybrid_coeff and _xc_type.
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
hybrid_coeff = 0.5

def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
    # A fictitious XC functional to demonstrate the usage
    rho0, dx, dy, dz = rho[:4]
    gamma = (dx**2 + dy**2 + dz**2)
    exc = .01 * rho0**2 + .02 * (gamma+.001)**.5
    vrho = .01 * 2 * rho0
    vgamma = .02 * .5 * (gamma+.001)**(-.5)
    vlapl = None
    vtau = None
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative

    # Mix with existing functionals
    pbe_xc = dft.libxc.eval_xc('pbe,pbe', rho, spin, relativity, deriv,
                               verbose)
    exc += pbe_xc[0] * 0.5
    vrho += pbe_xc[1][0] * 0.5
    vgamma += pbe_xc[1][1] * 0.5
    return exc, vxc, fxc, kxc

mf = dft.RKS(mol)
mf = mf.define_xc_(eval_xc, 'GGA', hyb=hybrid_coeff)
mf.verbose = 4
mf.kernel()

# half exact exchange in which 40% of the exchange is computed with short
# range part of the range-separation Coulomb operator (omega = 0.8)
beta = 0.2
rsh_coeff = (0.8, hybrid_coeff-beta, beta)
mf = dft.RKS(mol)
mf = mf.define_xc_(eval_xc, 'GGA', rsh=rsh_coeff)
mf.verbose = 4
mf.kernel()

