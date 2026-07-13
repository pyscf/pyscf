#!/usr/bin/env python

'''
GW calculation with exact frequency integration
'''

from pyscf import gto, dft, gw, scf
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

gw = gw.GW(mf, freq_int='exact')
gw.kernel()
print(gw.mo_energy)


# density-fitting GW with exact frequency integration
from pyscf.gw.gw_exact_df import GWExactDF
gw = GWExactDF(mf)
# test charge-test charge vertex correction
# gw.RPAE = True
gw.kernel()
print(gw.mo_energy)
# Galitskii-Migdal total energy
e_tot, e_hf, e_c = gw.energy_tot()
print("GW total energy:", e_tot)
print("HF energy:", e_hf)
print("Correlation energy:", e_c)

# spin-unrestricted
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz',
    spin = 1,
    charge = 1)
mf = scf.UHF(mol)
mf.kernel()

# density-fitting GW with exact frequency integration
from pyscf.gw.ugw_exact_df import UGWExactDF
gw = UGWExactDF(mf)
# test charge-test charge vertex correction
# gw.RPAE = True
gw.kernel()
print(gw.mo_energy)
# Galitskii-Migdal total energy
e_tot, e_hf, e_c = gw.energy_tot()
print("GW total energy:", e_tot)
print("HF energy:", e_hf)
print("Correlation energy:", e_c)
