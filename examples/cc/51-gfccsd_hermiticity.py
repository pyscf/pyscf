# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
GF-CCSD with different hermiticity options.
"""

from pyscf import gto, scf, cc

# Define system
mol = gto.Mole()
mol.atom = "C 0 0 0; O 0 0 1.13"
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
# We can force hermiticity in the moments, and additionally
# force the solver to be hermitian (positively defined).
#
gfcc1 = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc1.hermi_moments = False
gfcc1.hermi_solver = False
gfcc1.kernel()
ip1 = gfcc1.ipccsd(nroots=1)[0]

gfcc2 = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc2.hermi_moments = True
gfcc2.hermi_solver = False
gfcc2.kernel()
ip2 = gfcc2.ipccsd(nroots=1)[0]

gfcc3 = cc.gfccsd.GFCCSD(ccsd, niter=(3, 3))
gfcc3.hermi_moments = True
gfcc3.hermi_solver = True
gfcc3.kernel()
ip3 = gfcc3.ipccsd(nroots=1)[0]

print("Ionisation potentials:")
print("non-hermitian solver, non-hermitian moments", ip1)
print("non-hermitian solver, hermitian moments    ", ip2)
print("hermitian solver,     hermitian moments    ", ip3)
