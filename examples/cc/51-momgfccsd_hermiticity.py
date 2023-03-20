# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
Moment constrained GF-CCSD with different hermiticity options.

Ref: Backhouse, Booth, arXiv:2206.13198 (2022).
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

# Run moment-constrained GF-CCSD
#
# Default mode: GF Moments are non-hermitian, and 
# full Hamiltonian/Green's function is non-hermitian
gfcc1 = cc.MomGFCCSD(ccsd, niter=(4, 4))
gfcc1.hermi_moments = False
gfcc1.hermi_solver = False
gfcc1.kernel()
ip1 = gfcc1.ipgfccsd(nroots=1)[0]

# We can force the CCSD GF moments to be hermitian
gfcc2 = cc.MomGFCCSD(ccsd, niter=(4, 4))
gfcc2.hermi_moments = True
gfcc2.hermi_solver = False
gfcc2.kernel()
ip2 = gfcc2.ipgfccsd(nroots=1)[0]

# We can constrain the GF moments and full GF / 
# hamiltonian to be hermitian
gfcc3 = cc.MomGFCCSD(ccsd, niter=(4, 4))
gfcc3.hermi_moments = True
gfcc3.hermi_solver = True
gfcc3.kernel()
ip3 = gfcc3.ipgfccsd(nroots=1)[0]

# Compare to EOM-CCSD-IP first ionization potential
eip = ccsd.ipccsd(nroots=1)[0]

print("Ionisation potentials:")
print("non-hermitian solver, non-hermitian moments", ip1)
print("non-hermitian solver, hermitian moments    ", ip2)
print("hermitian solver,     hermitian moments    ", ip3)
print("IP-EOM-CCSD solver                         ", eip)
