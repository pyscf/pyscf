#!/usr/bin/env python

'''
GW calculation with exact frequency integration 
'''

from pyscf import gto, dft, gw
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

nocc = mol.nelectron//2

gw = gw.GW(mf, freq_int='exact')
gw.kernel()
print(gw.mo_energy)
