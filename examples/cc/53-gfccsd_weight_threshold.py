# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
GF-CCSD physical weight threshold.
"""

from pyscf import gto, scf, cc
import numpy as np

# Define system
mol = gto.Mole()
mol.atom = "Li 0 0 0; Li 0 0 1.7"
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

# The parameter weight_tol controls which excitations are
# reported as IPs or EAs in the output. Due to the nature
# of the method, at some parameterisations the approximate
# GFCCSD solver may give non-physical excitations near the
# Fermi energy with non-zero weight.

# Run a GF-CCSD calculation with the default weight_tol
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc.weight_tol = 1e-1
gfcc.kernel()

# Run a GF-CCSD calculation with a much lower weight_tol -
# one observes additional low-weighted IPs in the output
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc.weight_tol = 1e-5
gfcc.kernel()
