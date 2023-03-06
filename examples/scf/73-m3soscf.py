'''

This file details the basic usage of the M3SOSCF algorithm.

'''

from pyscf import gto, scf, dft



# Building the molecule, here CO is used as a simple example
mol_1 = gto.M(atom='C 0.0 0.0 0.0; O 1.27 0.0 0.0', basis='6-31g', verbose=4)
mol_2 = gto.M(atom='C 0.0 0.0 0.0; O 1.27 0.0 0.0', basis='6-31g', verbose=4, 
              charge=1, spin=1)


# First, a simple M3 iteration can be performed directly. The argument 'agents'
# is positional and required. In this example, 5 agents are used
m3_1 = scf.M3SOSCF(scf.RHF(mol_1), 5)

# The Iteration can be started both with the 'converge()' or 'kernel()'
# function.
conv, e_tot, mo_energy, mo_coeff, mo_occ = m3_1.kernel()

print(f"Converged? {conv}\nTotal Energy: {e_tot} ha\nMO Occupancies: {mo_occ}")

# This works identically for UHF
m3_2 = scf.M3SOSCF(scf.UHF(mol_2), 5)
conv, e_tot, mo_energy, mo_coeff, mo_occ = m3_2.kernel()

print(f"Converged? {conv}\nTotal Energy: {e_tot} ha")

# And it works identically for ROHF, RKS, UKS and ROKS
m3_1 = scf.M3SOSCF(dft.RKS(mol_1, xc='blyp'), 5)
conv, e_tot, mo_energy, mo_coeff, mo_occ = m3_1.kernel()
print(f"Converged? {conv}\nTotal Energy: {e_tot} ha\nMO Occupancies: {mo_occ}")

m3_2 = scf.M3SOSCF(dft.UKS(mol_2, xc='blyp'), 5)
conv, e_tot, mo_energy, mo_coeff, mo_occ = m3_2.kernel()
print(f"Converged? {conv}\nTotal Energy: {e_tot} ha")

# When constructing the M3SOSCF object, the following optional arguments can 
# be used:

# purge_solvers: fraction of solvers that are to be removed and reassigned
# every iteration
purge_solvers = 0.5 # Default
# convergence: 10^-convergence is the trust threshold for when an iteration 
# is perceived as converged.
convergence = 8 # Default
# init_scattering: Initial random scattering of the solvers.
init_scattering = 0.3 # Default
# trust_scale_range: Three numbers that define the influence of trust on the 
# scaling of the next iteration. The three numbers (gamma_1, gamma_2, gamma_3) 
# are used in the formula:
#
#   1 / scale = (gamma_2 - gamma_1) * (1 - trust)^gamma_3 + gamma_1
#
# Therefore, they carray the role of (min, max, power)
trust_scale_range = (0.5, 0.5, 0.05) # Default
# init_guess: Initial Guess. This can either be a string that is documented 
# in the SCF module (e. g. minao, 1e) or a numpy.ndarray of MO coeffs. In the 
# latter case, the matrix is not modified or controlled prior to the SCF 
# iteration, therefore incorrect symmetry will lead to incorrect results.
init_guess = 'minao' # Default
# stepsize: Stepsize of the parent CIAH solver
stepsize = 0.2 # Default

m3_1 = scf.M3SOSCF(scf.RHF(mol_1), 5, purge_solvers=purge_solvers, 
                   convergence=convergence, init_scattering=init_scattering, 
                   trust_scale_range=trust_scale_range, init_guess=init_guess, 
                   stepsize=stepsize)
conv, e_tot, mo_energy, mo_coeff, mo_occ = m3_1.kernel()
print(f"Converged? {conv}\nTotal Energy: {e_tot} ha")

