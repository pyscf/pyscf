#!/usr/bin/env python

'''
DF IP/EA-RADC calculations for closed-shell N2 
'''

from pyscf import gto, scf, adc, df

mol = gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2   )],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.build()

# Running conventional SCF followed by DF-ADC

mf = scf.RHF(mol).run()
myadc = adc.ADC(mf).density_fit('aug-cc-pvdz-ri')
myadc.kernel_gs()

# Running DF-SCF followed by DF-ADC

mf = scf.RHF(mol).density_fit().run()
myadc = adc.ADC(mf)
myadc.kernel_gs()

# Using different auxiliary basis for DF-SCF and DF-ADC 

mf = scf.RHF(mol).density_fit('aug-cc-pvdz-jkfit').run()
myadc = adc.ADC(mf).density_fit('aug-cc-pvdz-ri')
myadc.verbose = 6
eip,vip,pip,xip = myadc.kernel()

# Alternate way to compute DF-ADC 

mf = scf.RHF(mol).density_fit('aug-cc-pvdz-jkfit').run()
myadc = adc.ADC(mf)
myadc.with_df = df.DF(mol, auxbasis='aug-cc-pvdz-ri')
myadc.verbose = 6
myadc.method = "adc(3)"
myadc.method_type = "ea"
eea,vea,pea,xea = myadc.kernel(nroots = 3)

# Compute properties
myadc.compute_properties = True
myadc.analyze()

# Using different auxiliary basis for EE-ADC 
mf = scf.RHF(mol).density_fit('aug-cc-pvdz-jkfit').run()
myadc = adc.ADC(mf).density_fit('aug-cc-pvdz-ri')
myadc.method = "adc(3)"
myadc.method_type = "ee"
myadc.verbose = 6
eip,vip,pip,xip = myadc.kernel(nroots = 3)
