#!/usr/bin/env python

'''
IP/EA-ADC calculations for open-shell OH
'''

from pyscf import gto, scf, adc

mol = gto.Mole()
r = 0.969286393
mol.atom = [
    ['O', ( 0., 0.    , -r/2   )],
    ['H', ( 0., 0.    ,  r/2)],]
mol.basis = {'O':'aug-cc-pvdz',
             'H':'aug-cc-pvdz'}
mol.verbose = 0
mol.symmetry = False
mol.spin  = 1
mol.build()

mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

myadc = adc.ADC(mf)
myadc.kernel()

#IP/EA-ADC(2)
myadc.verbose = 4
eip,vip,pip = myadc.ip_adc(nroots=4)
eea,vea,pea = myadc.ea_adc(nroots=4)

#IP/EA-ADC(2)-x
myadc.method = "adc(2)-x"
myadc.kernel()
eip,vip,pip = myadc.ip_adc(nroots=4)
eea,vea,pea = myadc.ea_adc(nroots=4)

#IP/EA-ADC(3)
myadc.method = "adc(3)"
myadc.kernel()
eip,vip,pip = myadc.ip_adc(nroots=4)
eea,vea,pea = myadc.ea_adc(nroots=4)
