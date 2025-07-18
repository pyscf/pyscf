#!/usr/bin/env python

'''
IP/EA-RADC calculations for closed-shell N2 
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
myadc.verbose = 6
eip,vip,pip,xip = myadc.kernel()

#EA-RADC(2)-x for 1 root
myadc.method = "adc(2)-x"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel()

#Get EA-RADC(2)-x eigenevector analysis only
myadc.compute_properties = False
myadc.analyze()

#EA-RADC(3) for 3 roots and properties
myadc.compute_properties = True
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots = 3)
myadc.analyze()

#EE-RADC(2) for 3 roots
myadc.compute_properties = False
myadc.method = "adc(2)"
myadc.method_type = "ee"
eea,vea,pea,xea = myadc.kernel(nroots = 3)

