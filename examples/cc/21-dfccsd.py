#!/usr/bin/env python

'''
DF-CCSD and DF-IP/EA-EOM-CCSD for RHF is available

To use DF-RCCSD object, density fitting should be enabled at SCF level.
DFCCSD uses the same auxiliary basis as the DF-SCF method.  It does not
support a separated auxiliary basis.
'''

from pyscf import gto, scf, cc

mol = gto.Mole()
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]]

mol.basis = 'cc-pvdz'
mol.build()

mf = scf.RHF(mol).density_fit().run()
mycc = cc.RCCSD(mf).run()
print(mycc.e_corr - -0.21337100025961622)

print("IP energies... (right eigenvector)")
part = None
e,v = mycc.ipccsd(nroots=3,partition=part)
print(e)
print(e[0] - 0.43364287418576897)
print(e[1] - 0.5188001071775572 )
print(e[2] - 0.67851590275796392)

print("IP energies... (left eigenvector)")
e,lv = mycc.ipccsd(nroots=3,left=True,partition=part)
print(e)
print(e[0] - 0.43364286531878882)
print(e[1] - 0.51879999865136994)
print(e[2] - 0.67851587320495355)

mycc.ipccsd_star(e,v,lv)

print("EA energies... (right eigenvector)")
e,v = mycc.eaccsd(nroots=3,partition=part)
print(e)
print(e[0] - 0.16730125785810035)
print(e[1] - 0.23999823045518162)
print(e[2] - 0.50960183439619933)

print("EA energies... (left eigenvector)")
e,lv = mycc.eaccsd(nroots=3,left=True,partition=part)
print(e)
print(e[0] - 0.16730137808538076)
print(e[1] - 0.23999845448276602)
print(e[2] - 0.50960182130968001)


