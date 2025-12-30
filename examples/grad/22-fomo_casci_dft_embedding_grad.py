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
Example 22: Nuclear Gradients for FOMO-CASCI with DFT Core Embedding
====================================================================

This example demonstrates nuclear gradient calculations for FOMO-CASCI
with DFT-corrected core energy. This combines:

1. FOMO orbitals: Handle near-degenerate systems
2. DFT core: Include dynamical correlation in core treatment
3. Analytical gradients: Enable geometry optimization and dynamics

The gradient is computed using numerical differentiation to ensure
correctness for both the FOMO and DFT contributions.

References:
    FOMO-CASCI:
    P. Slavíček and T. J. Martínez,
    "Ab initio floating occupation molecular orbital-complete active space
    configuration interaction: An efficient approximation to CASSCF",
    J. Chem. Phys. 132, 234102 (2010)
    https://doi.org/10.1063/1.3436501
    
    DFT Core Embedding:
    S. Pijeau and E. G. Hohenstein,
    J. Chem. Theory Comput. 2017, 13, 1130-1146
    https://doi.org/10.1021/acs.jctc.6b00893
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons_fomo, casci_dft
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
print("FOMO-CASCI-DFT Nuclear Gradients")
print("="*70)
print(f"Active space: ({nelecas}e, {ncas}o)")
print(f"Core orbitals: {ncore}")
print(f"FOMO temperature: 0.25 eV (Gaussian smearing)")
print(f"XC functional: PBE")

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

# Compute FOMO-CASCI-DFT energy and gradient
print("\n" + "-"*70)
print("Computing FOMO-CASCI-DFT energy and gradient")
print("-"*70)

mc = casci_dft.CASCI(mf_fomo, ncas, nelecas, xc='PBE')
mc.kernel()
print(f"Energy: {mc.e_tot:.10f} Ha")

# Compute gradient
g = mc.Gradients().kernel()

# Compare all four methods
print("\n" + "="*70)
print("Comparison: All Four Methods")
print("="*70)

methods = {}

# 1. Standard CASCI (RHF + HF core)
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.kernel()
g_std = mc_std.Gradients().kernel()
methods['Standard CASCI'] = (mc_std.e_tot, g_std)

# 2. DFT-CASCI (RHF + DFT core)
mc_dft = casci_dft.CASCI(mf_rhf, ncas, nelecas, xc='PBE')
mc_dft.kernel()
g_dft = mc_dft.Gradients().kernel()
methods['DFT-CASCI (PBE)'] = (mc_dft.e_tot, g_dft)

# 3. FOMO-CASCI (FOMO + HF core) - numerical gradient
print("\nComputing FOMO-CASCI gradient numerically...")

def compute_fomo_casci_energy(coords_new):
    mol_new = mol.set_geom_(coords_new, unit='Bohr', inplace=False)
    mf_new = scf.RHF(mol_new).run()
    mf_fomo_new = addons_fomo.fomo_scf(
        mf_new, temperature=0.25, method='gaussian', restricted=(ncore, ncas)
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

mc_fomo = mcscf.CASCI(mf_fomo, ncas, nelecas)
mc_fomo.kernel()
methods['FOMO-CASCI'] = (mc_fomo.e_tot, g_fomo)

# 4. FOMO-CASCI-DFT (FOMO + DFT core) - already computed
methods['FOMO-CASCI-DFT (PBE)'] = (mc.e_tot, g)

# Print energy comparison
print("\n" + "-"*70)
print("Energy Comparison")
print("-"*70)
print(f"{'Method':<25} {'Energy (Ha)':<20} {'ΔE from Std (mHa)':<20}")
print("-"*65)
e_ref = methods['Standard CASCI'][0]
for name, (e, _) in methods.items():
    delta = (e - e_ref) * 1000
    print(f"{name:<25} {e:<20.10f} {delta:<+20.4f}")

# Print gradient comparison
print("\n" + "-"*70)
print("Gradient Comparison (Ha/Bohr)")
print("-"*70)

atoms = ['O', 'H1', 'H2']
coords = ['x', 'y', 'z']

for name, (_, grad) in methods.items():
    print(f"\n{name}:")
    print(f"         {'x':>14} {'y':>14} {'z':>14}")
    for i, atom in enumerate(atoms):
        print(f"{atom:<6}   {grad[i,0]:>14.8f} {grad[i,1]:>14.8f} {grad[i,2]:>14.8f}")

# Verify FOMO-CASCI-DFT gradient
print("\n" + "="*70)
print("Numerical Verification of FOMO-CASCI-DFT Gradient")
print("="*70)

def compute_fomo_dft_energy(coords_new):
    mol_new = mol.set_geom_(coords_new, unit='Bohr', inplace=False)
    mf_new = scf.RHF(mol_new).run()
    mf_fomo_new = addons_fomo.fomo_scf(
        mf_new, temperature=0.25, method='gaussian', restricted=(ncore, ncas)
    )
    mf_fomo_new.kernel()
    mc_new = casci_dft.CASCI(mf_fomo_new, ncas, nelecas, xc='PBE')
    mc_new.kernel()
    return mc_new.e_tot

g_num = np.zeros((mol.natm, 3))
print("\nComputing numerical gradient...")

for ia in range(mol.natm):
    for ix in range(3):
        coords_p = coords_ref.copy()
        coords_m = coords_ref.copy()
        coords_p[ia, ix] += step
        coords_m[ia, ix] -= step
        e_p = compute_fomo_dft_energy(coords_p)
        e_m = compute_fomo_dft_energy(coords_m)
        g_num[ia, ix] = (e_p - e_m) / (2 * step)

print(f"\n{'Atom':<6} {'Coord':<6} {'Analytical':<15} {'Numerical':<15} {'Difference':<15}")
print("-"*60)

max_diff = 0.0
for i, atom in enumerate(atoms):
    for j, coord in enumerate(coords):
        diff = abs(g[i,j] - g_num[i,j])
        max_diff = max(max_diff, diff)
        print(f"{atom:<6} {coord:<6} {g[i,j]:>14.8f} {g_num[i,j]:>14.8f} {diff:>14.2e}")
    if i < len(atoms) - 1:
        print()

print("-"*60)
print(f"Maximum difference: {max_diff:.2e} Ha/Bohr")

if max_diff < 1e-5:
    print("✓ Gradient verification PASSED!")
else:
    print("✗ Gradient verification FAILED!")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)
print("FOMO-CASCI-DFT combines:")
print("  - FOMO orbitals for handling near-degenerate systems")
print("  - DFT core energy for dynamical correlation")
print("  - Verified gradients for geometry optimization")
print("\nRecommended applications:")
print("  - Photochemistry and excited state dynamics")
print("  - Conical intersection optimization")
print("  - Multireference systems with significant core correlation")
