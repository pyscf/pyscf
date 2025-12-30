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
Example 81: DFT-Corrected CASCI with DFT Core Embedding
========================================================

This example demonstrates CASCI calculations with DFT-corrected core energy.
The core electrons are treated with DFT (exchange-correlation functional),
while the active space uses HF-like embedding to preserve wavefunction topology.

Theory:
    E_core^DFT = E_nuc + Tr[D_core * h] + 0.5*Tr[D_core * J] + E_xc[core]
    
    The active space embedding still uses J - 0.5*K (HF-like) to maintain
    compatibility with standard CI solvers.

Reference:
    S. Pijeau and E. G. Hohenstein,
    "Improved Semistochastic Heat-Bath Configuration Interaction",
    J. Chem. Theory Comput. 2017, 13, 1130-1146
    https://doi.org/10.1021/acs.jctc.6b00893
"""

from pyscf import gto, scf, mcscf
from pyscf.mcscf import casci_dft

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

# Define active space: 6 electrons in 6 orbitals
ncas = 6
nelecas = 6

print("="*70)
print("Comparison: Standard CASCI vs DFT-Corrected CASCI")
print("="*70)

# 1. Standard CASCI (HF core)
print("\n1. Standard CASCI (HF core energy):")
mc_hf = mcscf.CASCI(mf, ncas, nelecas)
mc_hf.kernel()
print(f"   Total Energy: {mc_hf.e_tot:.10f} Ha")

# 2. DFT-CASCI with LDA
print("\n2. DFT-CASCI with LDA core:")
mc_lda = casci_dft.CASCI(mf, ncas, nelecas, xc='LDA')
mc_lda.kernel()
print(f"   Total Energy: {mc_lda.e_tot:.10f} Ha")

# 3. DFT-CASCI with PBE
print("\n3. DFT-CASCI with PBE core:")
mc_pbe = casci_dft.CASCI(mf, ncas, nelecas, xc='PBE')
mc_pbe.kernel()
print(f"   Total Energy: {mc_pbe.e_tot:.10f} Ha")

# 4. DFT-CASCI with B3LYP
print("\n4. DFT-CASCI with B3LYP core:")
mc_b3lyp = casci_dft.CASCI(mf, ncas, nelecas, xc='B3LYP')
mc_b3lyp.kernel()
print(f"   Total Energy: {mc_b3lyp.e_tot:.10f} Ha")

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"{'Method':<25} {'Energy (Ha)':<20} {'ΔE from HF-CASCI (mHa)':<20}")
print("-"*70)
print(f"{'Standard CASCI (HF)':<25} {mc_hf.e_tot:<20.10f} {0.0:<20.4f}")
print(f"{'DFT-CASCI (LDA)':<25} {mc_lda.e_tot:<20.10f} {(mc_lda.e_tot - mc_hf.e_tot)*1000:<20.4f}")
print(f"{'DFT-CASCI (PBE)':<25} {mc_pbe.e_tot:<20.10f} {(mc_pbe.e_tot - mc_hf.e_tot)*1000:<20.4f}")
print(f"{'DFT-CASCI (B3LYP)':<25} {mc_b3lyp.e_tot:<20.10f} {(mc_b3lyp.e_tot - mc_hf.e_tot)*1000:<20.4f}")
print("="*70)

# Note: The CI coefficients are identical for all methods because
# the active space embedding uses HF-like potential (J - 0.5*K)
print("\nNote: CI coefficients are identical across all methods because")
print("the active space embedding uses HF-like potential (J - 0.5*K).")
print("Only the core energy differs (DFT vs HF treatment).")
