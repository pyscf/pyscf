#!/usr/bin/env python

'''
A simple example to run a GW calculation
'''

from pyscf import gto, dft, gw

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

nocc = mol.nelectron // 2
nmo = len(mf.mo_energy)

# By default, GW is done with analytic continuation
frozen = list(range(0, nocc - 3)) + list(range(nocc + 3, nmo))
gw = gw.GW(mf, frozen=frozen)
# same as
# gw = gw.gw_ac.GWAC(mf, frozen=frozen)
gw.kernel()
print(gw.mo_energy)
