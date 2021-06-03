#!/usr/bin/env python

'''
FNO-CCSD for RHF and UHF
'''

from pyscf import gto, scf, cc

mol = gto.Mole()
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]]

mol.basis = 'cc-pvqz'
mol.build()

# test code correctness
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()
print(mycc.e_corr - -0.3170511898840137)
nvir = mycc.nmo-mycc.nocc

mycc = cc.FNOCCSD(mf).run()
print(mycc.e_corr - -0.3170419015445936)

# print FNO error
print("Default FNO error (using %s/%s vir orbs) ="%(mycc.nmo-mycc.nocc,nvir), mycc.e_corr - -0.3170511898840137)

# lower NO occupation threshold
mycc = cc.FNOCCSD(mf, thresh=1e-5).run()
print("Lower threshold FNO error (using %s/%s vir orbs) ="%(mycc.nmo-mycc.nocc,nvir), mycc.e_corr - -0.3170511898840137)
# use delta-MP2 as correction
print("With delta-MP2, error =", mycc.e_corr+mycc.delta_emp2 - -0.3170511898840137)

# specify number of NOs to keep
mycc = cc.FNOCCSD(mf, nvir_act=55).run()
print("Lower threshold FNO error (using %s/%s vir orbs) ="%(mycc.nmo-mycc.nocc,nvir), mycc.e_corr - -0.3170511898840137)
# use delta-MP2 as correction
print("With delta-MP2, error =", mycc.e_corr+mycc.delta_emp2 - -0.3170511898840137)

# add (T) from NOs
mycc.ccsd_t()
