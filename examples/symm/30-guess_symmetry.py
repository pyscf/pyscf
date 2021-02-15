#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, symm, scf

'''
Pseudo symmetry for molecules which are slightly distorted from the symmetric
geometry.

Tuning the error tolerence for symmetry detection:

symm.geom.GEOM_THRESHOLD = .1
symm.geom.PLACE = 1

It can help guess the possible point group symmetry.  Using the "symmetry
adapted orbitals" mol.symm_orb, we can test how likely a MO belongs to a
particular irrep.
'''

mol = gto.M(
    atom = '''
      C    -.697620857  -1.20802476  -0.00800148
      C     .697539143  -1.20802476  -0.00800148
      C     1.39507714   0.          -0.00800148
      C     .697423143   1.20823524  -0.00920048
      C    -.697401857   1.20815724  -0.00967948
      C    -1.39500286   0.          -0.00868348
      H    -1.24737986  -2.16034176  -0.00755148
      H     1.24704714  -2.16053776  -0.00668648
      H     2.49475714   0.          -0.00736748
      H     1.24762314   2.16037824  -0.00925948
      H    -1.24752386   2.16043824  -0.01063248
      H    -2.49460686   0.          -0.00886348
    ''',
    basis = 'ccpvdz')
mf = scf.RHF(mol)
mf.kernel()

# Change error tolerence (to 0.1 Bohr, default is 1e-5 Bohr) for symmetry
# detection, so that the program can find the raw symmetry.
symm.geom.TOLERANCE = .1
mol.symmetry = True
mol.build(False, False)
print('Pseudo symmetry %s' % mol.groupname)

# Call mol.build again to avoid the atom coordinates changed
mol.symmetry = mol.groupname
mol.build(False, False)

irname = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff, check=False)

s = mol.intor('cint1e_ovlp_sph')
print('MO  id  irreps  likely')
for i, ir in enumerate(irname):
    k = mol.irrep_name.index(ir)
    s1 = reduce(numpy.dot, (mol.symm_orb[k].T, s, mf.mo_coeff[:,i]))
    s0 = reduce(numpy.dot, (mol.symm_orb[k].T, s, mol.symm_orb[k]))
    print('MO %3d  %-3s    %8.5f' % (i, ir, numpy.dot(s1, numpy.linalg.solve(s0, s1))))
