#!/usr/bin/env python

# Author: Junzi Liu <latrix1247@gmail.com>

'''
 Delta-SCF with maximium occupation method for calculating specific excited state.

 Here two SCF procedures are carried out with different MO occupation principle.
 Firstly the energy and molecular orbital information of ground state is
 obtained by a regular SCF calculation. The excited state is calculated by an
 independent SCF with a prescribed occupation pattern based on ground state. The
 energy difference between this two states can be seen as the excitation erergy
 originated from previous assigned orbitals.

 Instead of Afubau principle, maximium occupation method is used to avoid the
 excited state collpsing to ground state. Original get_occ will be replace by
 mom_occ with an addon and you will see message as below:

 overwite keys get_occ of <class 'pyscf.scf.uks.UKS'>
'''

from pyscf import gto, scf, dft

mol = gto.Mole()
mol.verbose = 1
#mol.output ='mom_DeltaSCF.out'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    ["H" , (0. , -0.757 , 0.587)],
    ["H" , (0. , 0.757  , 0.587)] ]
mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()

a = dft.UKS(mol)
a.xc = 'b3lyp'
# Use chkfile to store ground state information and start excited state
# caculation from these information directly 
#mf.chkfile='ground.chkfile'
a.scf()

# Read MO coefficients and occpuation number from chkfile
#mo0 = scf.chkfile.load('ground.chkfile', 'scf/mo_coeff')
#occ = scf.chkfile.load('ground.chkfile', 'scf/mo_occ')
mo0 = a.mo_coeff
occ = a.mo_occ

# Assigned initial occupation pattern
occ[0][4]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][5]=1      # it is still a singlet state

# New SCF caculation 
b = dft.UKS(mol)
b.xc = 'b3lyp'

# construct new dnesity matrix with new occpuation pattern
dm = b.make_rdm1(mo0, occ)
# Use mom occupation principle overwirte original one
b.get_occ = scf.addons.mom_occ(b, mo0, occ)
# Start new SCF with new density matrix
b.scf(dm)

print('Excited state energy: %.3g eV' % ((b.e_tot - a.e_tot)*27.211))
print('Alpha electron occpation pattern of excited state : %s' %(b.mo_occ[0]))
print(' Beta electron occpation pattern of excited state : %s' %(b.mo_occ[1]))
