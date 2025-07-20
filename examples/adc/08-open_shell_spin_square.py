#!/usr/bin/env python

'''
IP/EA-UADC calculations for open-shell OH
'''

from pyscf import gto, scf, adc
from pyscf.adc.uadc_ee import get_spin_square

mol = gto.Mole()
r = 0.969286393
mol.atom = [
    ['O', ( 0., 0.    , -r/2   )],
    ['H', ( 0., 0.    ,  r/2)],]
mol.basis = {'O':'aug-cc-pvdz',
             'H':'aug-cc-pvdz'}
mol.verbose = 4
mol.symmetry = False
mol.spin  = 1
mol.build()

#Start with using the UHF reference
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#EE-UADC(2)/UHF for 4 roots with properties and spin square expectation values
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = True
myadc.compute_spin_square = True
eee,vee,pee,xee = myadc.kernel(nroots=4)
myadc.analyze()

#Saving spin expectation values into an array
e,v,p,x = myadc.kernel(nroots=4)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = False
myadc.compute_spin_square = False
eee,vee,pee,xee = myadc.kernel(nroots=4)
spin = get_spin_square(myadc._adc_es)[0]
print("ADC(2)/UHF spin expectation values:")
print(spin)

#Repeat calculation using the ROHF reference
mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#EE-UADC(2)/ROHF for 4 roots with properties and spin square expectation values
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = True
myadc.compute_spin_square = True
eee,vee,pee,xee = myadc.kernel(nroots=4)
myadc.analyze()

#Saving spin expectation values into an array
e,v,p,x = myadc.kernel(nroots=4)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = False
myadc.compute_spin_square = False
eee,vee,pee,xee = myadc.kernel(nroots=4)
spin = get_spin_square(myadc._adc_es)[0]
print("ADC(2)/ROHF spin expectation values:")
print(spin)
