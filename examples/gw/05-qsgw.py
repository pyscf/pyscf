#!/usr/bin/env python

'''
Quasiparticle self-consistent GW calculation.

Method                                                     GW class
qsGW (analytical continuation, N4 scaling)                 QSGW
qsGW (fully analytic, N6 scaling)                          QSGWExact
unrestricted qsGW (analytical continuation, N4 scaling)    UQSGW
unrestricted qsGW (fully analytic, N6 scaling)             UQSGWExact
'''

from pyscf import gto, dft
from pyscf.gw.qsgw import QSGW
from pyscf.gw.uqsgw import UQSGW


# spin-restricted case
mol = gto.M(atom='H 0 0 0; F 0 0 0.91', basis='ccpvdz')

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

gw = QSGW(mf)
gw.kernel()
print(gw.mo_energy)

# spin-unrestricted case
mol = gto.M(atom='H 0 0 0; F 0 0 0.91', basis='ccpvdz', charge=1, spin=1)

mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

gw = UQSGW(mf)
gw.kernel()
# alpha channel
print(gw.mo_energy[0])
# beta channel
print(gw.mo_energy[1])
