#!/usr/bin/env python

'''
IP/EA-RADC outcore calculations for closed-shell N2 
'''

from pyscf import gto, scf, adc

mol = gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2   )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)

#IP-RADC(2) for 1 root
myadc.max_memory = 1
myadc.verbose = 6
eip,vip,pip,xip = myadc.kernel()

#EA-RADC(3) for 3 roots
myadc.max_memory = 20
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots = 3)

#EE-RADC(3) for 3 roots
myadc.max_memory = 1
myadc.method = "adc(3)"
myadc.method_type = "ee"
eea,vea,pea,xea = myadc.kernel(nroots = 3)
