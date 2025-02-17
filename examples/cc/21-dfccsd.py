#!/usr/bin/env python

'''
DF-CCSD and DF-IP/EA-EOM-CCSD for RHF/UHF is available

To use DF-RCCSD object, density fitting should be enabled at SCF level.
DFCCSD uses the same auxiliary basis as the DF-SCF method.  It does not
support a separated auxiliary basis.
'''

from pyscf import gto, scf, cc, df

mol = gto.Mole()
mol.atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]]

mol.basis = 'cc-pvdz'
mol.build()

mf = scf.RHF(mol).density_fit().run()
mycc = cc.CCSD(mf).run()
print(mycc.e_corr - -0.2134130020105784)

#
# Using different auxiliary basis for correlation part
#
mycc.with_df = df.DF(mol, auxbasis='ccpvdz-ri')
mycc.run()
print(mycc.e_corr - -0.2134650622862191)


print("IP energies... (right eigenvector)")
part = None
e,v = mycc.ipccsd(nroots=3,partition=part)
print(e)
print(e[0] - 0.43359796846314946)
print(e[1] - 0.51880158734392556)
print(e[2] - 0.67828839618227565)

print("IP energies... (left eigenvector)")
e,lv = mycc.ipccsd(nroots=3,left=True,partition=part)
print(e)
print(e[0] - 0.43359795868636808)
print(e[1] - 0.51880156554336332)
print(e[2] - 0.67828799083077551)

print("EA energies... (right eigenvector)")
e,v = mycc.eaccsd(nroots=3,partition=part)
print(e)
print(e[0] - 0.16737521666993965)
print(e[1] - 0.24027719217022464)
print(e[2] - 0.50917083110896155)

print("EA energies... (left eigenvector)")
e,lv = mycc.eaccsd(nroots=3,left=True,partition=part)
print(e)
print(e[0] - 0.16737533537691762)
print(e[1] - 0.24027732497703608)
print(e[2] - 0.50917096186773469)
