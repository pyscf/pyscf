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

mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()
print(mycc.e_corr - -0.3170511898840137)

mycc = cc.FNOCCSD(mf).run()
print(mycc.e_corr - -0.3170419015445936)

# lower NO occupation threshold
mycc = cc.FNOCCSD(mf, thresh=1e-5).run()
print("error from canonical =", mycc.e_corr - -0.3170511898840137)

# use delta-MP2 as correction
print("error from canonical =", mycc.e_corr+mycc.delta_emp2 - -0.3170511898840137)
