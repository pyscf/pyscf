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
myadc.kernel_gs()

#IP-RADC(2) for 3 roots
myadc.verbose = 4
myadcip = adc.radc_ip.RADCIP(myadc)
eip,vip,pip,xip = myadcip.kernel(nroots=3)

#EA-RADC(3) for 3 roots
myadc.method = "adc(3)"
myadc.kernel_gs()
myadcea = adc.radc_ea.RADCEA(myadc)
eea,vea,pea,xea = myadcea.kernel(nroots=3)

#Analyze eigenvectors only
myadcea.compute_properties = False
myadcea.analyze()

#IP/EA-RADC(3) for 1 root
eip,vip,pip,xip,adc_es = myadc.ip_adc()
eea,vea,pea,xea,adc_es = myadc.ea_adc()

