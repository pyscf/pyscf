#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Thermochemistry analysis based on nuclear Hessian.
'''

from pyscf import gto
from pyscf.hessian import thermo

# First compute nuclear Hessian.
mol = gto.M(
    atom = '''O    0.   0.       0
              H    0.   -0.757   0.587
              H    0.    0.757   0.587''',
    basis = '631g')

mf = mol.RHF().run()
hessian = mf.Hessian().kernel()

# Frequency analysis
freq_info = thermo.harmonic_analysis(mf.mol, hessian)
# Thermochemistry analysis at 298.15 K and 1 atmospheric pressure
thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)

print('Rotation constant')
print(thermo_info['rot_const'])

print('Zero-point energy')
print(thermo_info['ZPE'   ])

print('Internal energy at 0 K')
print(thermo_info['E_0K'  ])

print('Internal energy at 298.15 K')
print(thermo_info['E_tot' ])

print('Enthalpy energy at 298.15 K')
print(thermo_info['H_tot' ])

print('Gibbs free energy at 298.15 K')
print(thermo_info['G_tot' ])

print('Heat capacity at 298.15 K')
print(thermo_info['Cv_tot'])

