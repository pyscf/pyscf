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
Example 83: FOMO-CASCI with DFT Core Embedding
==============================================

This example combines FOMO (Floating Occupation Molecular Orbital) orbitals
with DFT-corrected core energy. This provides:

1. FOMO orbitals: Better handling of near-degenerate systems
2. DFT core: Improved treatment of dynamical correlation in core

Theory:
    - FOMO-SCF provides orbitals with fractional occupations near the
      Fermi level, improving convergence for multireference systems.
    - DFT core embedding replaces HF exchange with XC functional for
      core electrons, capturing dynamical correlation.
    - Active space uses HF-like embedding (J - 0.5*K) to preserve
      wavefunction topology.

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
print("FOMO-CASCI with DFT Core Embedding")
print("="*70)
print(f"Active space: ({nelecas}e, {ncas}o)")
print(f"Core orbitals: {ncore}")
print(f"FOMO temperature: 0.25 eV (Gaussian smearing)")

# Run base RHF for FOMO
mf_rhf = scf.RHF(mol).run()

# 1. Standard CASCI (HF orbitals, HF core)
print("\n" + "-"*70)
print("1. Standard CASCI (RHF orbitals, HF core)")
print("-"*70)
mc_std = mcscf.CASCI(mf_rhf, ncas, nelecas)
mc_std.kernel()
print(f"   Energy: {mc_std.e_tot:.10f} Ha")

# 2. FOMO-CASCI (FOMO orbitals, HF core)
print("\n" + "-"*70)
print("2. FOMO-CASCI (FOMO orbitals, HF core)")
print("-"*70)
mf_fomo = addons_fomo.fomo_scf(
    mf_rhf, 
    temperature=0.25, 
    method='gaussian',
    restricted=(ncore, ncas)
)
mf_fomo.kernel()
mc_fomo = mcscf.CASCI(mf_fomo, ncas, nelecas)
mc_fomo.kernel()
print(f"   Energy: {mc_fomo.e_tot:.10f} Ha")

# 3. DFT-CASCI (RHF orbitals, DFT core)
print("\n" + "-"*70)
print("3. DFT-CASCI (RHF orbitals, PBE core)")
print("-"*70)
mc_dft = casci_dft.CASCI(mf_rhf, ncas, nelecas, xc='PBE')
mc_dft.kernel()
print(f"   Energy: {mc_dft.e_tot:.10f} Ha")

# 4. FOMO-CASCI-DFT (FOMO orbitals, DFT core) - the full method
print("\n" + "-"*70)
print("4. FOMO-CASCI-DFT (FOMO orbitals, PBE core)")
print("-"*70)
mc_fomo_dft = casci_dft.CASCI(mf_fomo, ncas, nelecas, xc='PBE')
mc_fomo_dft.kernel()
print(f"   Energy: {mc_fomo_dft.e_tot:.10f} Ha")

# 5. Try different XC functionals with FOMO
print("\n" + "-"*70)
print("5. FOMO-CASCI-DFT with different XC functionals")
print("-"*70)

functionals = ['LDA', 'PBE', 'B3LYP', 'PBE0']
print(f"{'Functional':<15} {'Energy (Ha)':<20} {'ΔE from HF (mHa)':<20}")
print("-"*55)

for xc in functionals:
    mc_xc = casci_dft.CASCI(mf_fomo, ncas, nelecas, xc=xc)
    mc_xc.kernel()
    delta_e = (mc_xc.e_tot - mc_fomo.e_tot) * 1000
    print(f"{xc:<15} {mc_xc.e_tot:<20.10f} {delta_e:<20.4f}")

# Summary
print("\n" + "="*70)
print("Summary: Energy Comparison")
print("="*70)
print(f"{'Method':<40} {'Energy (Ha)':<20}")
print("-"*60)
print(f"{'Standard CASCI (RHF + HF core)':<40} {mc_std.e_tot:<20.10f}")
print(f"{'FOMO-CASCI (FOMO + HF core)':<40} {mc_fomo.e_tot:<20.10f}")
print(f"{'DFT-CASCI (RHF + PBE core)':<40} {mc_dft.e_tot:<20.10f}")
print(f"{'FOMO-CASCI-DFT (FOMO + PBE core)':<40} {mc_fomo_dft.e_tot:<20.10f}")
print("="*70)

# Analysis
print("\nEnergy contributions:")
print(f"  FOMO effect (HF core):    {(mc_fomo.e_tot - mc_std.e_tot)*1000:+.4f} mHa")
print(f"  DFT effect (RHF orbitals):{(mc_dft.e_tot - mc_std.e_tot)*1000:+.4f} mHa")
print(f"  Combined (FOMO + DFT):    {(mc_fomo_dft.e_tot - mc_std.e_tot)*1000:+.4f} mHa")

print("\nNote: FOMO-CASCI-DFT is recommended for:")
print("  - Multireference systems requiring accurate core treatment")
print("  - Photochemistry and excited state dynamics")
print("  - Systems where both static and dynamic correlation are important")
