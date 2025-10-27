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

# NEVPT2 calculation by default with CAS or DFCAS reference wavefunction does
# not uses the density fitting. That can be turned on by setting density_fit=True like below.
mp = mrpt.nevpt2.NEVPT(mycas, density_fit=True)
mp.kernel()

print("NEVPT2 correlation energy with density fitting: ", mp.e_corr)
print("Total energy: ", mycas.e_tot + mp.e_corr)
