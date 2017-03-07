#!/usr/bin/env python
#
# Author: koval.peter@gmail.com
#

'''
Loop over atom pairs
'''

from pyscf import gto, scf
import sys
import numpy as np

mol = gto.Mole_pure()
mol.build(
    atom = 'O 0.0000 0.0000 0.1132; H 0.0000 0.7512 -0.4528; H 0.0000 -0.7512 -0.4528',  # in Angstrom
    basis = 'ccpvtz')

natoms = len(mol._atom)
nprod  = 0
nprod_dir = 0
for ia in range(natoms):
  for ja in range(ia+1):
    #print(ia+1, ja+1, mol._atom[ia], mol._atom[ja])
    m2 = gto.Mole_pure()
    m2.build(atom=[mol._atom[ia], mol._atom[ja]], basis=mol.basis)
    eri = m2.intor('cint2e_sph')
    print(ia+1,ja+1, m2._atom, eri.shape)
    eigval = np.linalg.eigvalsh(eri)
    nprd_pp = np.sum(eigval > 1e-14)
    nprod = nprod + nprd_pp
    nprod_dir = nprod_dir + eigval.shape[0]
    print(eigval.shape[0], nprd_pp)

print nprod_dir, nprod, 1.0*nprod_dir/nprod

#m1 = gto.Mole()
#m1.build()
#m1.dump_input()





