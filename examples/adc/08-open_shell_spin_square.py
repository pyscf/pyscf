#!/usr/bin/env python

'''
IP/EA/EE-UADC calculations with spin-square expectation values
for the open-shell OH radical
'''

from pyscf import gto, scf, adc
from pyscf.adc.uadc_ee import get_spin_square as spin_square_ee
from pyscf.adc.uadc_ip import get_spin_square as spin_square_ip
from pyscf.adc.uadc_ea import get_spin_square as spin_square_ea

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

#1. UHF reference

mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#1.1 EE-UADC(2)/UHF for 4 roots with properties and spin square expectation values
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = True
myadc.compute_spin_square = True
eee,vee,pee,xee = myadc.kernel(nroots=4)
myadc.analyze()

#The spin expectation values can also be recovered after a plain kernel call
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
myadc.compute_properties = False
myadc.compute_spin_square = False
e,v,p,x = myadc.kernel(nroots=4)
spin = spin_square_ee(myadc._adc_es)[0]
print("EE-UADC(2)/UHF spin expectation values:")
print(spin)

#1.2 IP-UADC(3)/UHF for 4 roots
myadc = adc.ADC(mf)
myadc.method = "adc(3)"
myadc.method_type = "ip"
e_ip,v_ip,p_ip,x_ip = myadc.kernel(nroots=4)
spin = spin_square_ip(myadc._adc_es)[0]
print("IP-UADC(3)/UHF spin expectation values:")
print(spin)

#1.3 EA-UADC(2)/UHF for 4 roots
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ea"
e_ea,v_ea,p_ea,x_ea = myadc.kernel(nroots=4)
spin = spin_square_ea(myadc._adc_es)[0]
print("EA-UADC(2)/UHF spin expectation values:")
print(spin)

#2. ROHF reference

mf = scf.ROHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

#2.1 EE-UADC(2)/ROHF for 4 roots
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ee"
e,v,p,x = myadc.kernel(nroots=4)
spin = spin_square_ee(myadc._adc_es)[0]
print("EE-UADC(2)/ROHF spin expectation values:")
print(spin)

#2.2 IP-UADC(2)/ROHF for 4 roots
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ip"
e,v,p,x = myadc.kernel(nroots=4)
spin = spin_square_ip(myadc._adc_es)[0]
print("IP-UADC(2)/ROHF spin expectation values:")
print(spin)

#2.3 EA-UADC(2)/ROHF for 4 roots
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.method_type = "ea"
e,v,p,x = myadc.kernel(nroots=4)
spin = spin_square_ea(myadc._adc_es)[0]
print("EA-UADC(2)/ROHF spin expectation values:")
print(spin)
