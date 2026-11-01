#!/usr/bin/env python
#
# Author: Nikolay Bogdanov <n.bogdanov@inbox.lv>
#

'''
An example to run HF with background charges and ecp's.
'''

import numpy
from pyscf import gto, scf, qmmm
from pyscf.data.nist import BOHR

mol = gto.M(atom='''
H 0.0 0.0 0.0
F 0.0 0.0 1.0
            ''',
            basis='3-21g',
            verbose=4)

# For the example generate a random array of point charges
numpy.random.seed(1)
pc_n = 10
coords_pc = numpy.random.random((pc_n, 3))
charges_pc = (numpy.arange(pc_n) + 1.) * -0.1

# For the example generate a random array of embedidng ecp's coords
ecp_n = 5
coords_ecp = numpy.random.random((ecp_n, 3))
# We use Sr ecp with 36 core electrons, therefore the charge is +2
ecp_atom = "Sr"
ecp_dict = {ecp_atom: gto.load_ecp('stuttgart_dz', 'Sr')}
ecp_charge = 2.0
charges_ecp = numpy.ones(ecp_n) * ecp_charge

# Concatenate ecp and pc arrays
coords = numpy.vstack((coords_ecp, coords_pc))
charges = numpy.hstack((charges_ecp, charges_pc))

# Generate a (label, coords) list with atom for ecps and ghost "X" for PC's
atoms = [(ecp_atom, coord) if i < ecp_n else ("X", coord)
         for i, coord in enumerate(coords)]
print(charges)
print(coords)
print(atoms)

print("\n\nAdd point charges using (coords, charges):")
mf = qmmm.mm_charge(scf.HF(mol), coords, charges)
mf.kernel()

print("\n\nSame result when adding point charges using (atoms, charges):")
mf = qmmm.mm_charge(scf.HF(mol), atoms, charges)
mf.kernel()

# This is the actual example
print("\n\nNow add both PC's and ECP's with (atoms, charges, ecp=ecp_dict):")
mf = qmmm.mm_charge(scf.HF(mol), atoms, charges, ecp=ecp_dict)
mf.kernel()
E_emb = mf.e_tot

print("\n\nTest by including ECP's in mol explicitly:")
# Bring mol.atom to standard form (convert mol._atom fro Bohr back to Angstrom)
mol_atom_std = [(lab, [x*BOHR for x in coord]) for (lab, coord) in mol._atom]

mol_explicit = gto.M(atom=mol_atom_std+atoms[:ecp_n],basis=mol._basis,
                     dump_input=False, verbose=4, charge=ecp_charge*ecp_n,
                     ecp=ecp_dict)
mf_explicit = qmmm.mm_charge(scf.HF(mol_explicit), coords[ecp_n:], charges[ecp_n:])
mf_explicit.init_guess = "1e"
mf_explicit.kernel()
E_expl = mf_explicit.e_tot

print("\n\nThis way we add extra ineraction of embedding ECP's with PC's")
mol_nuc = gto.M(atom=atoms[:ecp_n],basis=mol._basis, dump_input=False,
                verbose=4, charge=ecp_charge*ecp_n, ecp=ecp_dict)
mf_nuc = qmmm.mm_charge(scf.HF(mol_nuc), coords[ecp_n:], charges[ecp_n:])
nuc_corr = mf_nuc.energy_nuc()
print("Nuclear energy of ECP with point charges:", nuc_corr)

print("\nDifference of SCF total energies of ECP's in embedding and in mole:")
print(f"E_expl - E_emb: {E_expl - E_emb}")
print("\nAfter correction:")
print(f"E_expl - E_emb - nuc_corr: {E_expl - E_emb - nuc_corr}")

print("\n\nTry to enable X2C with ECP's in mol:")
mf_explicit = qmmm.mm_charge(scf.HF(mol_explicit).x2c(), coords[ecp_n:], charges[ecp_n:])
mf_explicit.init_guess = "1e"
try:
    mf_explicit.kernel()
except NotImplementedError:
    print("Error: X2C does not work with ECP's in mol")

# This is the actual example
print("\n\nWith ECP's in embedding, it is possible to enable X2C:")
mf = qmmm.mm_charge(scf.HF(mol).x2c(), atoms, charges, ecp=ecp_dict)
mf.kernel()
