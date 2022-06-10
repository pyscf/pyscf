#!/usr/bin/env python

'''
X2C can be solved in different AO basis: the (j-adapted) spinor basis and the
spin-orbital real-spherical GTO basis. Ideally, they should converge to the
same SCF solutions. However, different solutions may be found in many
open-shell systems. It often caused by symmetry (the z-component of total
angular momentum) broken in GHF calculations.
'''

import numpy as np
from pyscf import gto

#
# For simple systems, X2C with spinor basis and X2C with spin-orbital basis
# under GHF framework give the same results.
#
mol = gto.M(atom='Ne', basis='ccpvdz-dk')
print('X2C-GHF')
mol.GHF().x2c1e().run()
print('j-adapted X2C-UHF')
mol.X2C().run()

#
# Different results for X2C with spinor basis and X2C with spin-orbital basis.
#
mol = gto.M(atom='C', basis='ccpvdz-dk')
print('X2C-GHF')
mol.GHF().x2c1e().run()
print('j-adapted X2C-UHF')
mf = mol.X2C().run()

#
# Using the j-adapted results to construct initial guess for X2C-GHF, SCF can
# be converged to the correct result in one iteration.
#
# The transformation from spin orbital basis to spinor basis
c = np.vstack(mol.sph2spinor_coeff())
# Construct new initial guess from the spinor basis solution
mo1 = c.dot(mf.mo_coeff)
dm = mf.make_rdm1(mo1, mf.mo_occ)

x2c_ghf = mol.GHF().x2c1e()
x2c_ghf.verbose = 4
x2c_ghf.max_cycle = 1
x2c_ghf.kernel(dm0=dm)
