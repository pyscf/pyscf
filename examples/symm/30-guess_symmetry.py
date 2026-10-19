#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
from pyscf import gto, symm, scf

# The low-level detection API reports the point group and orientation before
# Mole.build rotates coordinates into its symmetry-adapted frame.
water_atoms = [['O', (1., 0., 0.)],
               ['H', (0., -.757, .587)], ['H', (0., .757, .587)]]
group, origin, axes = symm.geom.detect_symm(water_atoms)
aligned_atoms = gto.format_atom(water_atoms, origin, axes)
print('Water point group and equivalent atoms:', group,
      symm.geom.symm_identical_atoms(group, aligned_atoms))
water = gto.M(atom=aligned_atoms, basis='ccpvdz')
_, irrep_ids = symm.basis.symm_adapted_basis(water, group)
print('Symmetry-adapted basis irreps:', irrep_ids)

# Linear molecules have their own symmetry-adapted basis construction.
linear = gto.M(atom='H 0 0 -1; H 0 0 0; H 0 0 1',
               basis='ccpvtz', charge=1)
linear_group, linear_origin, linear_axes = symm.geom.detect_symm(linear._atom)
_, linear_irreps = symm.basis.symm_adapted_basis(
    linear, linear_group, linear_origin, linear_axes)
print('Linear point group and basis irreps:', linear_group, linear_irreps)

# For a high-symmetry structure, subgroup maps the detected group to the
# supported Abelian subgroup used for orbital symmetry labels.
octahedron = [['H', xyz] for xyz in [(0, 0, 0), (1, 0, 0), (-1, 0, 0),
                                     (0, 1, 0), (0, -1, 0), (0, 0, 1),
                                     (0, 0, -1)]]
high_group, _, high_axes = symm.geom.detect_symm(octahedron)
print('Detected group and Abelian subgroup:', high_group,
      symm.geom.subgroup(high_group, high_axes)[0])

# route selects orbital indices whose irrep product matches a target symmetry.
# The integer irrep IDs use bitwise XOR for the direct product in D2h.
orbsym = [0, 3, 0, 2, 5, 6]
print('Three-orbital route to irrep 7:', symm.route(7, 3, orbsym))

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
