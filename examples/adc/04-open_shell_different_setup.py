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
myadc.kernel_gs()

#IP/EA-UADC(2) for 4 roots
myadc.verbose = 4
eip,vip,pip,xip,ip_es = myadc.ip_adc(nroots=4)
eea,vea,pea,xea,ea_es = myadc.ea_adc(nroots=4)

#IP/EA-UADC(2)-x for 4 roots
myadc.method = "adc(2)-x"
myadc.kernel_gs()
eip,vip,pip,xip,ip_es = myadc.ip_adc(nroots=4)
eea,vea,pea,xea,ea_es = myadc.ea_adc(nroots=4)

#IP/EA-UADC(3) for 4 roots
myadc.method = "adc(3)"
myadc.kernel_gs()
eip,vip,pip,xip,ip_es= myadc.ip_adc(nroots=4)
eea,vea,pea,xea,ea_es= myadc.ea_adc(nroots=4)

#Compute EA properties
ea_es.analyze()

#EE-UADC(3) for 4 roots
myadc.method = "adc(3)"
myadc.kernel_gs()
myadc.ee_adc(nroots=4)
