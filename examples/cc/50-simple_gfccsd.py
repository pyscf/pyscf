# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
GF-CCSD via moments of the Green's function.
"""

from pyscf import gto, scf, cc

# Define system
mol = gto.Mole()
mol.atom = "O 0 0 0; O 0 0 1"
mol.basis = "cc-pvdz"
mol.verbose = 5
mol.build()

# Run mean-field
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-10
mf.kernel()
assert mf.converged

# Run CCSD
ccsd = cc.CCSD(mf)
ccsd.kernel()
assert ccsd.converged

# Solve lambda equations
ccsd.solve_lambda()
assert ccsd.converged_lambda

# Run GF-CCSD
#
# Here we use 3 iterations in both the occupied (hole) and virtual
# (particle) sector, which ensures conservation of the first
# 2 * niter + 2 = 8 moments (0th through 7th) of the separate
# occupied (hole) and virtual (particle) Green's functions.
#
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc.kernel()
