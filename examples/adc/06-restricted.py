#!/usr/bin/env python

'''
IP/EA-RADC calculations for N2 
'''

from pyscf import gto, scf, adc

mol = gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2   )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.build()

mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)

#IP/EA-ADC(2) for 1 root
myadc.verbose = 4
myadc.kernel_gs()
eip,vip,pip = myadc.ip_adc()
eea,vea,pea = myadc.ea_adc()

#IP for 3 roots
myadc.method = "adc(2)-x"
eip,vip,pip = myadc.kernel(nroots = 3)

#EA-ADC(3) for 4 roots
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea = myadc.kernel(nroots = 4)
