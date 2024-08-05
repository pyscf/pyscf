#!/usr/bin/env python

'''
Mean field with k-points sampling when Brillouin Zone symmetry is considered
'''

import numpy
from pyscf.pbc import gto, scf, dft

cell = gto.M(
    a = numpy.asarray([[0.0, 2.6935121974, 2.6935121974], 
                       [2.6935121974, 0.0, 2.6935121974], 
                       [2.6935121974, 2.6935121974, 0.0]]),
    atom = '''Si  0.0000000000 0.0000000000 0.0000000000
              Si  1.3467560987 1.3467560987 1.3467560987''', 
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 5,
    space_group_symmetry = True,
    #symmorphic: if True, only the symmorphic subgroup is considered
    symmorphic = True,
)

nk = [2,2,2]
#The Brillouin Zone symmetry info is contained in the kpts object
kpts = cell.make_kpts(nk, 
                      space_group_symmetry=True, 
                      time_reversal_symmetry=True)
print(kpts)

kmf = scf.KRHF(cell, kpts)
kmf.kernel()

kmf = dft.KRKS(cell, kpts)
kmf.xc = 'camb3lyp'
kmf.kernel()

# By default, the mean-field calculation will use symmetry-adapted
# crystalline orbitals whenever possible. This can be turned off manually
# when instantiating the mean-field object.
kmf = scf.KRHF(cell, kpts, use_ao_symmetry=False)
kmf.kernel()

#
# The mean-field object with k-point symmetry can be converted back to
# the corresponding non-symmetric mean-field object
#

kmf = kmf.to_khf()
kmf.kernel(kmf.make_rdm1())

#
# Second order SCF solver can be used in the PBC SCF code the same way in the
# molecular calculation
#
kmf = scf.KRHF(cell, kpts).newton()
kmf.kernel()

#
#KUHF
#
kumf = scf.KUHF(cell, kpts)
kumf.kernel()

#
# The mean-field object with k-point symmetry can be converted back to
# the corresponding non-symmetric mean-field object
#
kumf = kumf.to_khf()
kumf.kernel(kumf.make_rdm1())

#
#KUHF with smearing
#
cell.spin = 2 * 2**3 #assume S=1 in each cell
kumf = scf.KUHF(cell, kpts)
#fix_spin: 
#   if True:
#       the final solution will have the same spin as input
#   if False: 
#       alpha and beta orbitals are sorted together based on energies,
#       and the final solution can have different spin from input
#
#Note: when gap is small, symmetry broken solution is usually the case,
#      which should be computed by turning off the symmstry options
kumf = scf.addons.smearing_(kumf, sigma=0.001, method='fermi',fix_spin=True)
kumf.kernel()
