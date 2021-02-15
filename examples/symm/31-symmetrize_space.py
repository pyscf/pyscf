#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import gto, symm, scf

'''
Symmetrize orbital space
'''

mol = gto.M(atom = '''C  0  0  0
                      H  1  1  1
                      H -1 -1  1
                      H  1 -1 -1
                      H -1  1 -1''',
            basis = 'sto3g',
            verbose = 0)
mf = scf.RHF(mol).run()
mo = mf.mo_coeff

#
# call mol.build to construct symmetry adapted basis.
#
# NOTE the molecule orientation.  Simply assigning mol.build(symmetry=True)
# may change the orientation of the molecule.  If the orientation is different
# to the orientation of the orbital space, the orbitals cannot be symmetrized.
# To keep the molecule orientation fixed, we need specify the molecular
# symmetry.
#
mol.build(0, 0, symmetry='D2')

#
# HOMO-2 HOMO-1 HOMO are degenerated.  They may not be properly symmetrized.
#
try:
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
except ValueError:
    print('The orbital symmetry cannot be labelled because some '
          'degenerated orbitals are not symmetrized.\n'
          'Degenerated HOMO energy: %s' % mf.mo_energy[2:5])

nocc = mol.nelectron // 2
occ_orb = symm.symmetrize_space(mol, mo[:,:nocc])
irrep_ids = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, occ_orb)
print('Occupied orbital symmetry: %s' % irrep_ids)

virt_orb = symm.symmetrize_space(mol, mo[:,nocc:])
irrep_ids = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, virt_orb)
print('Virtual orbital symmetry: %s' % irrep_ids)

