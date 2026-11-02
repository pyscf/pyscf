#!/usr/bin/env python

"""Detect molecular symmetry and construct symmetry-adapted AO bases.
"""

from pyscf import gto, symm

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
