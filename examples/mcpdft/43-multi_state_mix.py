#!/usr/bin/env python


'''
Perform multi-state PDFT averaging over states of different spins and/or
spatial symmetry

The mcpdft.multi_state function maybe not generate the right spin or spatial
symmetry as one needs.  This example shows how to put states with different
spins and spatial symmetry in a state-average solver using the method
multi_state_mix.
'''

import numpy as np
from pyscf import gto, scf, mcpdft, fci

mol = gto.M(atom='''
   O     0.    0.000    0.1174
   H     0.    0.757   -0.4696
   H     0.   -0.757   -0.4696
      ''', symmetry=True, basis="6-31G", verbose=3)

mf = scf.RHF(mol).run()

#
# state-average over 1 triplet + 2 singlets
# Note direct_spin1 solver is called here because the CI solver will take
# spin-mix solution as initial guess which may break the spin symmetry
# required by direct_spin0 solver
#
weights = np.ones(3) / 3
solver1 = fci.direct_spin1_symm.FCI(mol)
solver1.spin = 2
solver1 = fci.addons.fix_spin(solver1, shift=.2, ss=2)
solver1.nroots = 1
solver2 = fci.direct_spin0_symm.FCI(mol)
solver2.spin = 0
solver2.nroots = 2

mc = mcpdft.CASSCF(mf, "tPBE", 4, 4, grids_level=1)
# Currently, only the Linearized PDFT method is available for
# multi_state_mix
mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
mc.run()


#
# Example 2: Mix FCI wavefunctions with different symmetry irreps
#
mol = gto.Mole()
mol.build(atom='''
 O     0.    0.000    0.1174
 H     0.    0.757   -0.4696
 H     0.   -0.757   -0.4696
''', symmetry=True, basis='631g', verbose=3)

mf = scf.RHF(mol).run()

# Also possible to construct 2 solvers of different wfnsym, but
# of the same spin symmetry
weights = [.5, .5]
solver1 = fci.direct_spin1_symm.FCI(mol)
solver1.wfnsym= 'A1'
solver1.spin = 0
solver2 = fci.direct_spin1_symm.FCI(mol)
solver2.wfnsym= 'A2'
solver2.spin = 2

mc = mcpdft.CASSCF(mf, "tPBE", 4, 4, grids_level=1)
mc = mc.multi_state_mix([solver1, solver2], weights, "lin")
mc.kernel()
