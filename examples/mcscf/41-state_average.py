#!/usr/bin/env python

'''
State average over states of different spins and/or spatial symmetry

The mcscf.state_average_ function maybe not generate the right spin or spatial
symmetry as one needs.  This example shows how to put states with different
spins and spatial symmetry in a state-average solver using the method
state_average_mix_.
'''

import numpy as np
from pyscf import gto, scf, mcscf, fci

r = 1.8
mol = gto.Mole()
mol.atom = [
    ['C', ( 0., 0.    , -r/2   )],
    ['C', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pvdz'
mol.unit = 'B'
mol.symmetry = True
mol.build()
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g': 4, 'E1gx': 0, 'E1gy': 0, 'A1u': 4,
                  'E1uy': 2, 'E1ux': 2, 'E2gx': 0, 'E2gy': 0, 'E2uy': 0, 'E2ux': 0}
ehf = mf.kernel()
#mf.analyze()

#
# state-average over 1 triplet + 2 singlets
# Note direct_spin1 solver is called here because the CI solver will take
# spin-mix solution as initial guess which may break the spin symmetry
# required by direct_spin0 solver
#
weights = np.ones(3)/3
solver1 = fci.direct_spin1_symm.FCI(mol)
solver1.spin = 2
solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
solver1.nroots = 1
solver2 = fci.direct_spin0_symm.FCI(mol)
solver2.spin = 0
solver2.nroots = 2

mc = mcscf.CASSCF(mf, 8, 8)
mcscf.state_average_mix_(mc, [solver1, solver2], weights)

# Mute warning msgs
mc.check_sanity = lambda *args: None

mc.verbose = 4
mc.kernel()


#
# Example 2: Mix FCI wavefunctions with different symmetry irreps
#
mol = gto.Mole()
mol.build(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
''', symmetry=True, basis='631g')

mf = scf.RHF(mol).run()

weights = [.5, .5]
solver1 = fci.direct_spin1_symm.FCI(mol)
solver1.wfnsym= 'A1'
solver1.spin = 0
solver2 = fci.direct_spin1_symm.FCI(mol)
solver2.wfnsym= 'A2'
solver2.spin = 0

mc = mcscf.CASSCF(mf, 4, 4)
mcscf.state_average_mix_(mc, [solver1, solver2], weights)
mc.kernel()
