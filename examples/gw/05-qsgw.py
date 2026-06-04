#!/usr/bin/env python

'''
quasiparticle self-energy GW calculation
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
from pyscf.gw.qsgw import QSGW
gw = QSGW(mf)
gw.kernel()

# self-energy evaluated using exact frequency integration
# N6 scaling, accurate for all states
from pyscf.gw.qsgw_exact import QSGWExact
gw = QSGWExact(mf)
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
from pyscf.gw.uqsgw import UQSGW
gw = UQSGW(mf)
gw.kernel()

# self-energy evaluated using exact frequency integration
from pyscf.gw.uqsgw_exact import UQSGWExact
gw = UQSGWExact(mf)
gw.kernel()
