#!/usr/bin/env python

'''
eigenvalue self-energy GW calculation
'''

from pyscf import gto, dft

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

# self-energy evaluated using analytical continuation
# N4 scaling, inaccurate for core and high-lying states
from pyscf.gw.evgw import EVGW
gw = EVGW(mf)
gw.kernel()

# evGW0
gw = EVGW(mf)
gw.W0 = True
gw.kernel()

# self-energy evaluated using exact frequency integration
# N6 scaling, accurate for all states
from pyscf.gw.evgw_exact import EVGWExact
gw = EVGWExact(mf)
gw.kernel()

# evGW0 with exact frequency integration
gw = EVGWExact(mf)
gw.W0 = True
gw.kernel()

# spin-unrestricted
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz',
    spin = 1,
    charge = 1)
mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

# self-energy evaluated using analytical continuation
from pyscf.gw.uevgw import UEVGW
gw = UEVGW(mf)
gw.kernel()

# evGW0
gw = UEVGW(mf)
gw.W0 = True
gw.kernel()

# self-energy evaluated using exact frequency integration
from pyscf.gw.uevgw_exact import UEVGWExact
gw = UEVGWExact(mf)
gw.kernel()

# evGW0 with exact frequency integration
gw = UEVGWExact(mf)
gw.W0 = True
gw.kernel()
