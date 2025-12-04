#!/usr/bin/env python

'''
Example to run NEVPT2 with density fitting.
'''

from pyscf import gto, scf, mcscf, mrpt

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvqz',
    spin = 2)

myhf = scf.RHF(mol).density_fit(auxbasis='ccpvqz-jkfit')
myhf.kernel()

from pyscf.mcscf import avas
mo_coeff = avas.kernel(myhf, ['O 2p'], minao=mol.basis)[2]

mycas = mcscf.CASCI(myhf, 6, 8)
mycas.kernel(mo_coeff)

# For DF-CAS object by default the density fitting will be used
# for NEVPT2 calculation as well.
mp = mrpt.nevpt2.NEVPT(mycas)
mp.kernel()
e_tot1 = mycas.e_tot + mp.e_corr

# Even though density fitting is used for reference MCSCF wavefunction,
# it can be turned off for NEVPT2 calculation as follows:
mp = mrpt.nevpt2.NEVPT(mycas, density_fit=False)
mp.kernel()
e_tot2 = mycas.e_tot + mp.e_corr

print("Total energy with DF-NEVPT2: ", e_tot1)
print("Total energy with NEVPT2: ", e_tot2)
