# Filename: 55-momgfccsd_advanced_example.py
# Author: Xiexuan <xiexuan@kernel-dev.com>

"""
An advanced example demonstrating the use of PySCF for quantum chemistry calculations.
This script will:
1. Define a molecular system (N2).
2. Perform a Hartree-Fock calculation.
3. Execute a CCSD calculation.
4. Perform a CCSDT calculation to include triple excitations.
5. Extract and print molecular orbital energies, correlation energies, and excitation energies.
6. Calculate the one-body and two-body density matrices.
7. Save results to a text file.

References:
1. Sun, Q.; et al. "PySCF: The Python for Strongly Correlated Electron Systems." 
   Wiley Interdisciplinary Reviews: Computational Molecular Science 2018, 8(1), e1340. DOI: 10.1002/wcms.1340.
2. Neuscamman, E.; et al. "The Coupled-Cluster Method with Explicitly Correlated Wave Functions." 
   The Journal of Chemical Physics 2012, 136(4), 044113. DOI: 10.1063/1.3671783.
"""

import numpy as np
from pyscf import gto, scf, cc, lib

# Define the molecular system (Nitrogen molecule)
mol = gto.Mole()
mol.atom = "N 0 0 0; N 0 0 1.1"  # Geometry of the nitrogen molecule
mol.unit = "A"  # Set units to Angstroms
mol.basis = "cc-pvdz"  # Set the basis set (correlation consistent polarized valence double zeta)
mol.verbose = 4  # Set verbosity level for output
mol.build()  # Build the molecule

# Run mean-field calculation (Hartree-Fock)
mf = scf.RHF(mol)  # Use restricted Hartree-Fock method
mf.kernel()  # Perform the calculation
assert mf.converged  # Check for convergence

# Print the Hartree-Fock molecular orbital energies
print("Hartree-Fock Molecular Orbital Energies:")
print(mf.mo_energy)

# Run CCSD calculation
ccsd = cc.CCSD(mf)  # Create CCSD object
ccsd.kernel()  # Perform CCSD calculation
assert ccsd.converged  # Check for convergence

# Print CCSD correlation energy and molecular orbital energies
print("CCSD Correlation Energy:", ccsd.e_corr)
print("CCSD Molecular Orbital Energies:")
print(ccsd.mo_energy)

# Extract excitation energies using CCSD
nroots = 5  # Number of excited states to compute
excited_states_ip = ccsd.ipccsd(nroots=nroots)  # Ionization potentials
excited_states_ea = ccsd.eaccsd(nroots=nroots)  # Electron affinities
print("CCSD Excitation Energies (Ionization Potentials):")
print(excited_states_ip)
print("CCSD Excitation Energies (Electron Affinities):")
print(excited_states_ea)

# Run CCSDT calculation for triple excitations
ccsdt = cc.CCSDT(ccsd)  # Create CCSDT object
ccsdt.kernel()  # Perform CCSDT calculation
assert ccsdt.converged  # Check for convergence

# Print CCSDT correlation energy
print("CCSDT Correlation Energy:", ccsdt.e_corr)

# Compare CCSD and CCSDT correlation energies
print("Difference CCSDT - CCSD:", ccsdt.e_corr - ccsd.e_corr)

# Generate one-body and two-body density matrices from the CCSDT calculation
dm1 = ccsdt.make_rdm1()  # 1-RDM (one-body density matrix)
dm2 = ccsdt.make_rdm2()  # 2-RDM (two-body density matrix)

print("One-Body Density Matrix:")
print(dm1)

print("Two-Body Density Matrix:")
print(dm2)

# Calculate the total energy of the system
total_energy = mf.e_tot + ccsdt.e_corr  # Total energy is the sum of HF energy and correlation energy
print("Total Energy (HF + CCSDT):", total_energy)

# Save results to a text file
with open("pyscf_advanced_results.txt", "w") as f:
    f.write("Hartree-Fock Molecular Orbital Energies:\n")
    f.write(str(mf.mo_energy) + "\n")
    f.write("CCSD Correlation Energy: {}\n".format(ccsd.e_corr))
    f.write("CCSD Molecular Orbital Energies:\n")
    f.write(str(ccsd.mo_energy) + "\n")
    f.write("CCSD Excitation Energies (Ionization Potentials):\n")
    f.write(str(excited_states_ip) + "\n")
    f.write("CCSD Excitation Energies (Electron Affinities):\n")
    f.write(str(excited_states_ea) + "\n")
    f.write("CCSDT Correlation Energy: {}\n".format(ccsdt.e_corr))
    f.write("Difference CCSDT - CCSD: {}\n".format(ccsdt.e_corr - ccsd.e_corr))
    f.write("One-Body Density Matrix:\n")
    f.write(str(dm1) + "\n")
    f.write("Two-Body Density Matrix:\n")
    f.write(str(dm2) + "\n")
    f.write("Total Energy (HF + CCSDT): {}\n".format(total_energy))

print("Results saved to pyscf_advanced_results.txt")
