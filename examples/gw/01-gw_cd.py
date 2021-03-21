#!/usr/bin/env python

'''
GW calculation with contour deformation
'''

from pyscf import gto, dft, gw
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

nocc = mol.nelectron//2

gw = gw.GW(mf, freq_int='cd')
gw.kernel(orbs=range(nocc-3,nocc+3))
print(gw.mo_energy)
