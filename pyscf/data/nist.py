# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
NIST physical constants

https://physics.nist.gov/cuu/Constants/
https://physics.nist.gov/cuu/Constants/Table/allascii.txt
'''

LIGHT_SPEED = 137.03599967994   #http://physics.nist.gov/cgi-bin/cuu/Value?alph
# BOHR = .529 177 210 92(17) e-10m  #http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
BOHR = 0.52917721092  # Angstroms

ALPHA = 7.2973525664e-3         #http://physics.nist.gov/cgi-bin/cuu/Value?alph
G_ELECTRON = 2.00231930436182   # http://physics.nist.gov/cgi-bin/cuu/Value?gem
E_MASS = 9.10938356e-31         # kg https://physics.nist.gov/cgi-bin/cuu/Value?me
PROTON_MASS = 1.672621898e-27   # kg https://physics.nist.gov/cgi-bin/cuu/Value?mp
BOHR_MAGNETON = 927.4009994e-26 # J/T http://physics.nist.gov/cgi-bin/cuu/Value?mub
NUC_MAGNETON = BOHR_MAGNETON * E_MASS / PROTON_MASS
PLANCK = 6.626070040e-34        # J*s http://physics.nist.gov/cgi-bin/cuu/Value?h
HARTREE2J = 4.359744650e-18     # J https://physics.nist.gov/cgi-bin/cuu/Value?hrj
HARTREE2EV = 27.21138602        # eV https://physics.nist.gov/cgi-bin/cuu/Value?threv
HARTREE2WAVENUMBER = 2.194746313702e9
E_CHARGE = 1.6021766208e-19     # C https://physics.nist.gov/cgi-bin/cuu/Value?e
LIGHT_SPEED_SI = 299792458      # https://physics.nist.gov/cgi-bin/cuu/Value?c
AVOGADRO = 6.022140857e23       # https://physics.nist.gov/cgi-bin/cuu/Value?na
DEBYE = 3.335641e-30            # C*m = 1e-18/LIGHT_SPEED_SI https://cccbdb.nist.gov/debye.asp
AU2DEBYE = E_CHARGE * BOHR*1e-10 / DEBYE # 2.541746
