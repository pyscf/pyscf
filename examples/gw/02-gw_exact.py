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
gw.kernel()
# Galitskii-Migdal total energy
gw.energy_tot()

# test charge-test charge vertex correction
mf = scf.RHF(mol)
mf.kernel()
gw = GWExactDF(mf)
gw.RPAE = True
gw.kernel()

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
mygw = UGWExactDF(mf)
mygw.kernel()
# Galitskii-Migdal total energy
mygw.energy_tot()

# test charge-test charge vertex correction
mf = scf.UHF(mol)
mf.kernel()
mygw = UGWExactDF(mf)
mygw.RPAE = True
mygw.kernel()
