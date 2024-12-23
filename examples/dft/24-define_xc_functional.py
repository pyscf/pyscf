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

import numpy as np
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

def eval_gga_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    # A fictitious XC functional to demonstrate the usage
    rho0, dx, dy, dz = rho
    gamma = (dx**2 + dy**2 + dz**2)
    exc = .01 * rho0**2 + .02 * (gamma+.001)**.5
    vrho = .01 * 3 * rho0**2 + .02 * (gamma+.001)**.5
    vgamma = .02 * .5 * (gamma+.001)**(-.5)
    vxc = (vrho, vgamma)
    v2rho2 = 0.01 * 6 * rho0
    v2rhosigma = np.zeros(gamma.shape)
    v2sigma2 = 0.02 * .5 * -.5 * (gamma+.001)**(-1.5)
    # 2nd order functional derivative
    fxc = (v2rho2, v2rhosigma, v2sigma2)
    kxc = None  # 3rd order functional derivative

    # Mix with existing functionals
    pbe_xc = dft.libxc.eval_xc('pbe,pbe', rho, spin, relativity, deriv, verbose)
    exc += pbe_xc[0] * 0.5
    vrho += pbe_xc[1][0] * 0.5
    vgamma += pbe_xc[1][1] * 0.5
    # The output follows the libxc.eval_xc API convention
    return exc, vxc, fxc, kxc

mf = dft.RKS(mol)
mf = mf.define_xc_(eval_gga_xc, 'GGA', hyb=hybrid_coeff)
mf.verbose = 4
mf.kernel()

def eval_mgga_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    # A fictitious XC functional to demonstrate the usage
    rho0, dx, dy, dz, tau = rho
    gamma = (dx**2 + dy**2 + dz**2)
    exc = .01 * rho0**2 + .02 * (gamma+.001)**.5 +  0.01 * tau**2
    vrho = .01 * 3 * rho0**2 + .02 * (gamma+.001)**.5
    vgamma = .02 * .5 * (gamma+.001)**(-.5)
    vtau = 0.02 * tau
    vxc = (vrho, vgamma, vtau)
    v2rho2 = 0.01 * 6 * rho0
    v2rhosigma = np.zeros(gamma.shape)
    v2sigma2 = 0.02 * .5 * -.5 * (gamma+.001)**(-1.5)
    v2tau2 = np.full(tau.shape, 0.02)
    v2rhotau = np.zeros(tau.shape)
    v2sigmatau = np.zeros(tau.shape)
    # 2nd order functional derivative
    fxc = (v2rho2, v2rhosigma, v2sigma2, v2tau2, v2rhotau, v2sigmatau)
    kxc = None  # 3rd order functional derivative

    # Mix with existing functionals
    pbe_xc = dft.libxc.eval_xc('pbe,pbe', rho[:4], spin, relativity, deriv, verbose)
    exc += pbe_xc[0] * 0.5
    vrho += pbe_xc[1][0] * 0.5
    vgamma += pbe_xc[1][1] * 0.5
    # The output follows the libxc.eval_xc API convention
    return exc, vxc, fxc, kxc

# half exact exchange in which 40% of the exchange is computed with short
# range part of the range-separation Coulomb operator (omega = 0.8)
beta = 0.2
rsh_coeff = (0.8, hybrid_coeff-beta, beta)
mf = dft.RKS(mol)
mf = mf.define_xc_(eval_mgga_xc, 'MGGA', rsh=rsh_coeff)
mf.verbose = 4
mf.kernel()
