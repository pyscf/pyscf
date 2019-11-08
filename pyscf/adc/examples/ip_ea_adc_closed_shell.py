#!/usr/bin/env python

'''
IP/EA-ADC calculations for closed-shell N2
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
myadc.kernel()

#IP/EA-ADC(2)
myadc.verbose = 4
eip,vip,pip = myadc.ip_adc(nroots=1)
eea,vea,pea = myadc.ea_adc(nroots=1)

#IP/EA-ADC(2)-x
myadc.method = "adc(2)-x"
myadc.kernel()
eip,vip,pip = myadc.ip_adc(nroots=1)
eea,vea,pea = myadc.ea_adc(nroots=1)

#IP/EA-ADC(3)
myadc.method = "adc(3)"
myadc.kernel()
eip,vip,pip = myadc.ip_adc(nroots=1)
eea,vea,pea = myadc.ea_adc(nroots=1)
