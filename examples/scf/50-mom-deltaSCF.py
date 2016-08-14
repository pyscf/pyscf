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
 mom_occ with an addon. And it can be applied to unrestricted HF/KS and
 restricted open-shell HF/KS and you will see message as below:

 overwite keys get_occ of <class 'pyscf.scf.uks.UKS'> or <class 'pyscf.scf.roks.ROKS'>
'''
import numpy
from pyscf import gto, scf, dft

mol = gto.Mole()
mol.verbose = 5
#mol.output ='mom_DeltaSCF.out'
mol.atom = [
    ["O" , (0. , 0.     , 0.)],
    ["H" , (0. , -0.757 , 0.587)],
    ["H" , (0. , 0.757  , 0.587)] ]
mol.basis = {"H": '6-31g',
             "O": '6-31g',}
mol.build()

# 1. mom-Delta-SCF based on unrestricted HF/KS 
a = dft.UKS(mol)
a.xc = 'b3lyp'
# Use chkfile to store ground state information and start excited state
# caculation from these information directly 
#a.chkfile='ground_u.chkfile'
a.scf()

# Read MO coefficients and occpuation number from chkfile
#mo0 = scf.chkfile.load('ground_u.chkfile', 'scf/mo_coeff')
#occ = scf.chkfile.load('ground_u.chkfile', 'scf/mo_occ')
mo0 = a.mo_coeff
occ = a.mo_occ

# Assign initial occupation pattern
occ[0][4]=0      # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][5]=1      # it is still a singlet state

# New SCF caculation 
b = dft.UKS(mol)
b.xc = 'b3lyp'

# Construct new dnesity matrix with new occpuation pattern
dm_u = b.make_rdm1(mo0, occ)
# Apply mom occupation principle
b = scf.addons.mom_occ(b, mo0, occ)
# Start new SCF with new density matrix
b.scf(dm_u)


# 2. mom-Delta-SCF based on restricted open-shell HF/KS 
c = dft.ROKS(mol)
c.xc = 'b3lyp'
# Use chkfile to store ground state information and start excited state
# caculation from these information directly 
#c.chkfile='ground_ro.chkfile'
c.scf()

# Read MO coefficients and occpuation number from chkfile
#mo0 = scf.chkfile.load('ground_ro.chkfile', 'scf/mo_coeff')
#occ = scf.chkfile.load('ground_ro.chkfile', 'scf/mo_occ')
# Change 1-dimension occupation number list into 2-dimension occupation number
# list like the pattern in unrestircted calculation
mo0 = c.mo_coeff
occ = c.mo_occ
setocc = numpy.zeros((2, occ.size))
setocc[:, occ==2] = 1

# Assigned initial occupation pattern
setocc[0][4] = 0    # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
setocc[0][5] = 1    # it is still a singlet state
ro_occ = setocc[0][:] + setocc[1][:]    # excited occupation pattern within RO style

# New SCF caculation 
d = dft.ROKS(mol)
d.xc = 'b3lyp'

# Construct new dnesity matrix with new occpuation pattern
dm_ro = d.make_rdm1(mo0, ro_occ)
# Apply mom occupation principle
d = scf.addons.mom_occ(d, mo0, setocc)
# Start new SCF with new density matrix
d.scf(dm_ro)

# Summary
print('----------------UKS calculation----------------')
print('Excitation energy(UKS): %.3g eV' % ((b.e_tot - a.e_tot)*27.211))
print('Alpha electron occpation pattern of excited state(UKS) : %s' %(b.mo_occ[0]))
print(' Beta electron occpation pattern of excited state(UKS) : %s' %(b.mo_occ[1]))
print('----------------ROKS calculation----------------')
print('Excitation energy(ROKS): %.3g eV' % ((d.e_tot - c.e_tot)*27.211))
print('Electron occpation pattern of excited state(ROKS) : %s' %(d.mo_occ))
