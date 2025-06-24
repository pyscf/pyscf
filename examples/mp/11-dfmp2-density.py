#!/usr/bin/env python
#

'''
Calculating densities with DF-MP2, demonstrated for the dipole moment of CH3Cl.
'''

from numpy.linalg import norm
from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.mp.dfmp2_native import DFMP2

mol = Mole()
mol.atom = '''
C   0.000000   0.000000   0.000000
Cl  0.000000   0.000000   1.785000
H   1.019297   0.000000  -0.386177
H  -0.509649   0.882737  -0.386177
H  -0.509649  -0.882737  -0.386177
'''
mol.basis = 'aug-cc-pVTZ'
mol.build()

mf = RHF(mol).run()
pt = DFMP2(mf).run()

# The unrelaxed density always has got natural occupation numbers between 2 and 0.
# However, it is inaccurate for properties.
dm_ur = pt.make_rdm1_unrelaxed(ao_repr=True)

# The relaxed density is more accurate for properties when MP2 is well-behaved,
# whereas the natural occupation numbers can be above 2 or below 0 for ill-behaved systems.
dm_re = pt.make_rdm1_relaxed(ao_repr=True)

print('')
print('HF dipole moment:')
dip = mf.dip_moment()   # 2.10
print('Absolute value: {0:.3f} Debye'.format(norm(dip)))

print('')
print('Unrelaxed MP2 dipole moment:')
dip = mf.dip_moment(dm=dm_ur)   # 2.07
print('Absolute value: {0:.3f} Debye'.format(norm(dip)))

print('')
print('Relaxed MP2 dipole moment:')
dip = mf.dip_moment(dm=dm_re)   # 1.90
print('Absolute value: {0:.3f} Debye'.format(norm(dip)))

print('')
print('Experimental reference: 1.870 Debye')
