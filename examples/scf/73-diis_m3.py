'''

This file details the basic usage of the M3SOSCF algorithm.

'''

from pyscf import gto, scf, dft



# Building the molecule, here CO is used as a simple example
mol = gto.M(atom='C 0.0 0.0 0.0; O 1.27 0.0 0.0', basis='6-31g', verbose=4)

# Next, the DIIS/M3 object has to be built.
diis_m3 = scf.DIIS_M3(scf.RHF(mol), 15)
# The following optional arguments can be specified in the DIIS_M3 constructor:
#
#   purge_solvers: float
#           The percentage of solvers which are to be annihilated and reassigned in every step of M3.
#       convergence: float
#           10^-convergence is the convergence threshold for M3.
#       init_scattering: float
#           Initial Scattering value for the M3 calculation.
#       trust_scale_range: float[3]
#           Array of 3 floats consisting of min, max and gamma for the trust scale.
#       mem_size: int
#           Number of past values that should be considered in the M3 calculation. Default is strongly recommended.
#       mem_scale: float
#           Scaling used for past iterations in the M3 calculation. Default is strongly recommended.
#


# This can then be converged.

conv, e_tot, mo_energy, mo_coeff, mo_occ = diis_m3.kernel()
print(f"Converged? {conv}\nTotal Energy: {e_tot}")
# The following optional arguments can be specified in the kernel function:
#  
#   buffer_size: int
#       Minimum number of DIIS iterations. Strongly recommended to be at least the size of the DIIS buffer.
#   switch_thresh: float
#       Maximum difference of energy that is tolerated between two macro-iterations of DIIS before a switch to M3 is enforced.
#   hard_switch: int
#       Maximum number of DIIS iterations (not macro-iterations) that are allowed before a switch to
#       M3 is enforced.
#
