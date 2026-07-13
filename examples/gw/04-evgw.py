#!/usr/bin/env python

'''
Eigenvalue self-consistent GW calculation.

Method                                                   GW class    Option
evGW (analytical continuation, N4 scaling)               EVGW        -
evGW0 (analytical continuation, N4 scaling)              EVGW        W0 = True
evGW (fully analytic, N6 scaling)                        EVGWExact   -
evGW0 (fully analytic, N6 scaling)                       EVGWExact   W0 = True
unrestricted evGW (analytical continuation, N4 scaling)  UEVGW       -
unrestricted evGW0 (analytical continuation, N4 scaling) UEVGW       W0 = True
unrestricted evGW exact (fully analytic, N6 scaling)     UEVGWExact  -
unrestricted evGW0 exact (fully analytic, N6 scaling)    UEVGWExact  W0 = True
'''

from pyscf import gto, dft
from pyscf.gw.evgw import EVGW
from pyscf.gw.uevgw import UEVGW


# spin-restricted case
mol = gto.M(atom='H 0 0 0; F 0 0 0.91', basis='ccpvdz')

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

gw = EVGW(mf)
# use W0 option to enable evGW0 calculation
# gw.W0 = True
gw.kernel()
print(gw.mo_energy)

# spin-unrestricted case
mol = gto.M(atom='H 0 0 0; F 0 0 0.91', basis='ccpvdz', charge=1, spin=1)

mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

gw = UEVGW(mf)
# use W0 option to enable evGW0 calculation
# gw.W0 = True
gw.kernel()
# alpha channel
print(gw.mo_energy[0])
# beta channel
print(gw.mo_energy[1])
