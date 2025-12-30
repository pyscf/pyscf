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
Example 21: Nuclear Gradients for FOMO-CASCI
============================================

This example demonstrates nuclear gradient calculations for FOMO-CASCI
(Floating Occupation Molecular Orbital CASCI).

FOMO-CASCI uses orbitals from a FOMO-SCF calculation, which assigns
fractional occupations to orbitals near the Fermi level. This improves
SCF convergence for multireference systems.

For FOMO-CASCI gradients, the standard CPHF orbital response equations
cannot be used due to fractional occupations. The gradient is computed
by setting the orbital response to zero (frozen orbital approximation)
or using numerical differentiation.

Reference:
    P. Slavíček and T. J. Martínez,
    "Ab initio floating occupation molecular orbital-complete active space
    configuration interaction: An efficient approximation to CASSCF",
    J. Chem. Phys. 132, 234102 (2010)
    https://doi.org/10.1063/1.3436501
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons_fomo
from pyscf.grad import casci_dft as casci_dft_grad
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

# Define active space
ncas = 6
nelecas = 6
ncore = (mol.nelectron - nelecas) // 2

print("="*70)
print("FOMO-CASCI Nuclear Gradients")
print("="*70)
print(f"Active space: ({nelecas}e, {ncas}o)")
print(f"Core orbitals: {ncore}")
print(f"FOMO temperature: 0.25 eV (Gaussian smearing)")

# Run RHF and FOMO-SCF
mf_rhf = scf.RHF(mol).run()

mf_fomo = addons_fomo.fomo_scf(
    mf_rhf,
    temperature=0.25,
    method='gaussian',
    restricted=(ncore, ncas)
)
mf_fomo.kernel()

print(f"\nFOMO-SCF orbital occupations: {mf_fomo.mo_occ}")

# 1. Standard CASCI gradient (for comparison)
print("\n" + "-"*70)
print("1. Standard CASCI (RHF orbitals) gradient:")
print("-"*70)
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.kernel()
g_std = mc_std.Gradients().kernel()

# 2. FOMO-CASCI gradient
print("\n" + "-"*70)
print("2. FOMO-CASCI gradient:")
print("-"*70)
mc_fomo = mcscf.CASCI(mf_fomo, ncas, nelecas)
mc_fomo.kernel()

# Note: Standard CASCI gradient doesn't work directly with FOMO
# due to fractional occupations. We use the grad_elec_fomo function.
# For HF core, we can use a wrapper or compute numerically.

# Numerical gradient for FOMO-CASCI
print("\nComputing numerical FOMO-CASCI gradient...")

def compute_fomo_casci_energy(coords_new):
    """Compute FOMO-CASCI energy at displaced geometry."""
    mol_new = mol.set_geom_(coords_new, unit='Bohr', inplace=False)
    mf_new = scf.RHF(mol_new).run()
    mf_fomo_new = addons_fomo.fomo_scf(
        mf_new,
        temperature=0.25,
        method='gaussian',
        restricted=(ncore, ncas)
    )
    mf_fomo_new.kernel()
    mc_new = mcscf.CASCI(mf_fomo_new, ncas, nelecas)
    mc_new.kernel()
    return mc_new.e_tot

step = 1e-4
coords_ref = mol.atom_coords()
g_fomo = np.zeros((mol.natm, 3))

for ia in range(mol.natm):
    for ix in range(3):
        coords_p = coords_ref.copy()
        coords_m = coords_ref.copy()
        coords_p[ia, ix] += step
        coords_m[ia, ix] -= step
        
        e_p = compute_fomo_casci_energy(coords_p)
        e_m = compute_fomo_casci_energy(coords_m)
        g_fomo[ia, ix] = (e_p - e_m) / (2 * step)

print("\nFOMO-CASCI gradient (numerical):")
atoms = ['O', 'H1', 'H2']
print(f"         {'x':>14} {'y':>14} {'z':>14}")
for i, atom in enumerate(atoms):
    print(f"{atom:<6}   {g_fomo[i,0]:>14.8f} {g_fomo[i,1]:>14.8f} {g_fomo[i,2]:>14.8f}")

# 3. Comparison
print("\n" + "="*70)
print("Gradient Comparison (Ha/Bohr)")
print("="*70)

print(f"\n{'Atom':<6} {'Coord':<6} {'Standard CASCI':<15} {'FOMO-CASCI':<15} {'Difference':<15}")
print("-"*60)

coords = ['x', 'y', 'z']
max_diff = 0.0
for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        diff = abs(g_std[i,j] - g_fomo[i,j])
        max_diff = max(max_diff, diff)
        print(f"{atom:<6} {coord:<6} {g_std[i,j]:>14.8f} {g_fomo[i,j]:>14.8f} {diff:>14.8f}")
    if i < len(atoms) - 1:
        print()

print("-"*60)
print(f"Maximum difference between methods: {max_diff:.6f} Ha/Bohr")

# 4. Effect of temperature on gradient
print("\n" + "="*70)
print("Effect of FOMO Temperature on Gradient (O atom, z-component)")
print("="*70)

temperatures = [0.1, 0.25, 0.5, 1.0]
print(f"{'Temperature (eV)':<20} {'dE/dR_O_z (Ha/Bohr)':<20}")
print("-"*40)

for temp in temperatures:
    def energy_at_temp(coords_new):
        mol_new = mol.set_geom_(coords_new, unit='Bohr', inplace=False)
        mf_new = scf.RHF(mol_new).run()
        mf_fomo_new = addons_fomo.fomo_scf(
            mf_new, temperature=temp, method='gaussian', restricted=(ncore, ncas)
        )
        mf_fomo_new.kernel()
        mc_new = mcscf.CASCI(mf_fomo_new, ncas, nelecas)
        mc_new.kernel()
        return mc_new.e_tot
    
    # Compute just O_z component
    coords_p = coords_ref.copy()
    coords_m = coords_ref.copy()
    coords_p[0, 2] += step
    coords_m[0, 2] -= step
    
    g_oz = (energy_at_temp(coords_p) - energy_at_temp(coords_m)) / (2 * step)
    print(f"{temp:<20.2f} {g_oz:<20.8f}")

print("\n" + "="*70)
print("Notes")
print("="*70)
print("- FOMO-CASCI gradients require special handling due to fractional occupations")
print("- Standard CPHF equations don't apply; numerical differentiation is used")
print("- The gradient depends on the FOMO temperature parameter")
print("- As temperature → 0, FOMO-CASCI approaches standard CASCI")
