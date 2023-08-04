#!/usr/bin/env python

'''
A simple example to run a GW calculation 
'''

from pyscf import gto, dft, gw
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

nocc = mol.nelectron//2

# By default, GW is done with analytic continuation
gw = gw.GW(mf)
# same as gw = gw.GW(mf, freq_int='ac')
gw.kernel(orbs=range(nocc-3,nocc+3))
print(gw.mo_energy)
