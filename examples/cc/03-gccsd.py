#!/usr/bin/env python

'''
GCCSD: CCSD based on the GHF reference.

The cluster amplitudes of GCCSD are represented in the spin-orbital basis and
are solved without assuming spin symmetry. GCCSD can be applied to spin-orbit
coupled systems.

For non-relativistic calculations, GCCSD is typically equivalent to the
corresponding UHF-CCSD calculation. When spin-orbit coupling (SOC) is included,
for example, the X2C Hamiltonian (see examples/x2c/03-x2c_ghf.py) or SOC-ECP
(see examples/scf/44-soc_ecp.py), the GHF orbitals become complex-valued, and
the resulting GCCSD amplitude are also complex-valued.
'''

import pyscf

mol = pyscf.M(atom='''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
basis='cc-pvdz')
#
# Non-relativistic calculation. The CCSD object returned by mf.CCSD() is an
# instance of the GCCSD class. The cluster amplitudes are represented in the
# spin-orbital basis.
#
mf = mol.GHF().run()
mycc = mf.CCSD().run()

#
# Enable SOC via the X2C Hamiltonian. GCCSD amplitudes are complex-valued.
#
mf = mol.GHF().x2c().run()
mycc = mf.CCSD().run()

#
# For calculations using ECPs, SOC can be enabled with the setting
#     mf.with_soc = True
#
# Running mf.CCSD() on such a reference will performs a GCCSD calculation
# with complex-valued amplitudes.
#
