#!/usr/bin/env python
#
# Author: Igor S. Gerasimov <foxtranigor@gmail.com>
#

"""
A simple example to run MP2 calculation with CABS correction.
"""

import pyscf

mol = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz')

mf = mol.RHF().run()

mf.MP2().run()

pyscf.mp.cabs.energy_singles(mf, auxbasis='ccpvdzri')
pyscf.dbbsc.energy(mf)
