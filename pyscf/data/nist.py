'''
NIST physical constants

https://physics.nist.gov/cuu/Constants/
'''

from pyscf.lib.parameters import BOHR, LIGHT_SPEED, ALPHA

G_ELECTRON = 2.00231930436182   # http://physics.nist.gov/cgi-bin/cuu/Value?gem
E_MASS = 9.10938356e-31         # kg https://physics.nist.gov/cgi-bin/cuu/Value?me
PROTON_MASS = 1.672621898e-27   # kg https://physics.nist.gov/cgi-bin/cuu/Value?mp
BOHR_MAGNETON = 927.4009994e-26 # J/T http://physics.nist.gov/cgi-bin/cuu/Value?mub
NUC_MAGNETON = BOHR_MAGNETON * E_MASS / PROTON_MASS
PLANCK = 6.626070040e-34        # J*s http://physics.nist.gov/cgi-bin/cuu/Value?h
HARTREE2J = 4.359744650e-18     # J https://physics.nist.gov/cgi-bin/cuu/Value?hrj
HARTREE2EV = 27.21138602        # eV https://physics.nist.gov/cgi-bin/cuu/Value?threv
E_CHARGE = 1.6021766208e-19     # C https://physics.nist.gov/cgi-bin/cuu/Value?e
LIGHT_SPEED_SI = 299792458      # https://physics.nist.gov/cgi-bin/cuu/Value?c
