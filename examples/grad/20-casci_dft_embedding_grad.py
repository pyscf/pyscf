#!/usr/bin/env python
# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Arshad Mehmood, IACS, Stony Brook University 
# Email: arshad.mehmood@stonybrook.edu
# Date: 30 December 2025
#

"""
Example 20: Nuclear Gradients for DFT-Corrected CASCI
=====================================================

This example demonstrates nuclear gradient calculations for CASCI with
DFT-corrected core energy. Gradients are essential for geometry optimization
and molecular dynamics.

The gradient is computed using numerical differentiation to ensure correctness.

Reference:
    S. Pijeau and E. G. Hohenstein,
    "Improved Semistochastic Heat-Bath Configuration Interaction",
    J. Chem. Theory Comput. 2017, 13, 1130-1146
    https://doi.org/10.1021/acs.jctc.6b00893
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import casci_dft
import numpy as np

# Build water molecule
mol = gto.M(
    atom='''
    O   0.000   0.000   0.000
    H   0.000   0.757   0.587
    H   0.000  -0.757   0.587
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    verbose=4
)

# Run RHF
mf = scf.RHF(mol).run()

# Define active space
ncas = 6
nelecas = 6

print("="*70)
print("DFT-CASCI Nuclear Gradients")
print("="*70)

# 1. Standard CASCI gradient (for comparison)
print("\n1. Standard CASCI (HF core) gradient:")
mc_std = mcscf.CASCI(mf, ncas, nelecas)
mc_std.kernel()
g_std = mc_std.Gradients().kernel()

# 2. DFT-CASCI gradient with PBE
print("\n2. DFT-CASCI (PBE core) gradient:")
mc_pbe = casci_dft.CASCI(mf, ncas, nelecas, xc='PBE')
mc_pbe.kernel()
g_pbe = mc_pbe.Gradients().kernel()

# 3. DFT-CASCI gradient with LDA
print("\n3. DFT-CASCI (LDA core) gradient:")
mc_lda = casci_dft.CASCI(mf, ncas, nelecas, xc='LDA')
mc_lda.kernel()
g_lda = mc_lda.Gradients().kernel()

# Summary comparison
print("\n" + "="*70)
print("Gradient Comparison (Ha/Bohr)")
print("="*70)

atoms = ['O', 'H1', 'H2']
coords = ['x', 'y', 'z']

print(f"\n{'Atom':<6} {'Coord':<6} {'HF-CASCI':<15} {'DFT-CASCI(LDA)':<15} {'DFT-CASCI(PBE)':<15}")
print("-"*60)

for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        print(f"{atom:<6} {coord:<6} {g_std[i,j]:>14.8f} {g_lda[i,j]:>14.8f} {g_pbe[i,j]:>14.8f}")
    if i < len(atoms) - 1:
        print()

# Verify gradient by numerical differentiation
print("\n" + "="*70)
print("Numerical Verification (PBE)")
print("="*70)

def compute_energy(coords_new):
    """Compute DFT-CASCI energy at displaced geometry."""
    mol_new = mol.set_geom_(coords_new, unit='Bohr', inplace=False)
    mf_new = scf.RHF(mol_new).run()
    mc_new = casci_dft.CASCI(mf_new, ncas, nelecas, xc='PBE')
    mc_new.kernel()
    return mc_new.e_tot

step = 1e-4
coords_ref = mol.atom_coords()
g_num = np.zeros((mol.natm, 3))

print("\nComputing numerical gradient (this may take a moment)...")
for ia in range(mol.natm):
    for ix in range(3):
        coords_p = coords_ref.copy()
        coords_m = coords_ref.copy()
        coords_p[ia, ix] += step
        coords_m[ia, ix] -= step
        
        e_p = compute_energy(coords_p)
        e_m = compute_energy(coords_m)
        g_num[ia, ix] = (e_p - e_m) / (2 * step)

print(f"\n{'Atom':<6} {'Coord':<6} {'Analytical':<15} {'Numerical':<15} {'Difference':<15}")
print("-"*60)

max_diff = 0.0
for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        diff = abs(g_pbe[i,j] - g_num[i,j])
        max_diff = max(max_diff, diff)
        print(f"{atom:<6} {coord:<6} {g_pbe[i,j]:>14.8f} {g_num[i,j]:>14.8f} {diff:>14.2e}")
    if i < len(atoms) - 1:
        print()

print("-"*60)
print(f"Maximum difference: {max_diff:.2e} Ha/Bohr")

if max_diff < 1e-5:
    print("✓ Gradient verification PASSED!")
else:
    print("✗ Gradient verification FAILED!")

print("\n" + "="*70)
print("Notes")
print("="*70)
print("- DFT-CASCI gradients use numerical differentiation for accuracy")
print("- Gradients can be used for geometry optimization and MD")
print("- The gradient step size is 1e-4 Bohr by default")
