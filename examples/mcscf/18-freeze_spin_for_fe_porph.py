#!/usr/bin/env python

from pyscf import scf
from pyscf import gto
from pyscf import mcscf, fci
from pyscf.mcscf import dmet_cas

'''
Force the FCI solver of CASSCF follow the spin state.

The default FCI solver cannot stick on the triplet state.  Using
fci.addons.force_spin_ function to decorate mc.fcisolver, a constraint of
the spin eigenfunction is imposed on the FCI solver.

Slow convergence is observed in this test.  In this system, the CI
coefficients and orbital rotation are strongly coupled.  Small orbital
rotation leads to significant change of CI eigenfunction.  The micro iteration
is not able to predict the right orbital rotations since the first order
approximation for orbital gradients and CI hamiltonian are just too far to the
exact value.
'''

mol = gto.Mole()
mol.atom = [
    ['Fe', (0.      , 0.0000  , 0.0000)],
    ['N' , (1.9764  , 0.0000  , 0.0000)],
    ['N' , (0.0000  , 1.9884  , 0.0000)],
    ['N' , (-1.9764 , 0.0000  , 0.0000)],
    ['N' , (0.0000  , -1.9884 , 0.0000)],
    ['C' , (2.8182  , -1.0903 , 0.0000)],
    ['C' , (2.8182  , 1.0903  , 0.0000)],
    ['C' , (1.0918  , 2.8249  , 0.0000)],
    ['C' , (-1.0918 , 2.8249  , 0.0000)],
    ['C' , (-2.8182 , 1.0903  , 0.0000)],
    ['C' , (-2.8182 , -1.0903 , 0.0000)],
    ['C' , (-1.0918 , -2.8249 , 0.0000)],
    ['C' , (1.0918  , -2.8249 , 0.0000)],
    ['C' , (4.1961  , -0.6773 , 0.0000)],
    ['C' , (4.1961  , 0.6773  , 0.0000)],
    ['C' , (0.6825  , 4.1912  , 0.0000)],
    ['C' , (-0.6825 , 4.1912  , 0.0000)],
    ['C' , (-4.1961 , 0.6773  , 0.0000)],
    ['C' , (-4.1961 , -0.6773 , 0.0000)],
    ['C' , (-0.6825 , -4.1912 , 0.0000)],
    ['C' , (0.6825  , -4.1912 , 0.0000)],
    ['H' , (5.0441  , -1.3538 , 0.0000)],
    ['H' , (5.0441  , 1.3538  , 0.0000)],
    ['H' , (1.3558  , 5.0416  , 0.0000)],
    ['H' , (-1.3558 , 5.0416  , 0.0000)],
    ['H' , (-5.0441 , 1.3538  , 0.0000)],
    ['H' , (-5.0441 , -1.3538 , 0.0000)],
    ['H' , (-1.3558 , -5.0416 , 0.0000)],
    ['H' , (1.3558  , -5.0416 , 0.0000)],
    ['C' , (2.4150  , 2.4083  , 0.0000)],
    ['C' , (-2.4150 , 2.4083  , 0.0000)],
    ['C' , (-2.4150 , -2.4083 , 0.0000)],
    ['C' , (2.4150  , -2.4083 , 0.0000)],
    ['H' , (3.1855  , 3.1752  , 0.0000)],
    ['H' , (-3.1855 , 3.1752  , 0.0000)],
    ['H' , (-3.1855 , -3.1752 , 0.0000)],
    ['H' , (3.1855  , -3.1752 , 0.0000)],
]

mol.basis = 'ccpvdz'
mol.verbose = 5
mol.output = 'fepor3.out'
mol.spin = 2
mol.build()

m = scf.ROHF(mol)
m.level_shift_factor = 1.5
scf.fast_newton(m)

mc = mcscf.CASSCF(m, 10, 10)
idx3d = [i for i,s in enumerate(mol.spheric_labels(1)) if 'Fe 3d' in s]
mo = dmet_cas.dmet_cas(mc, m.make_rdm1(), idx3d, base=0)
fci.addons.force_spin_(mc.fcisolver)
mc.kernel(mo)

