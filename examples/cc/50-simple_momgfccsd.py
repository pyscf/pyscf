# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
Construct a Green's function at the CCSD level via a
number of spectral moment constraints

Ref: Backhouse, Booth, arXiv:2206.13198 (2022).
"""

import numpy
from pyscf import gto, scf, cc, lib

# Define system
mol = gto.Mole()
mol.atom = "O 0 0 0; O 0 0 1.2"
mol.unit = "A"
mol.basis = "cc-pvdz"
mol.verbose = 4
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
# Here we use 4 cycles in both the occupied (hole) and virtual
# (particle) sector, which ensures conservation of the first
# 2 * niter + 2 = 10 spectral moments (0th through 9th) of the separate
# occupied (hole) and virtual (particle) Green's functions.
# These can be increased for more accuracy but will eventually
# lose numerical precision.
#
# The gfcc object will store information on the resulting
# pole energies and residues of the Green's function.
gfcc = cc.MomGFCCSD(ccsd, niter=(4, 4))
gfcc.kernel()

# Compare IPs and EAs to IP/EA-EOM-CCSD
eip,cip = ccsd.ipccsd(nroots=6)
eea,cea = ccsd.eaccsd(nroots=6)

# The poles of the full-frequency Green's function can then be 
# accessed and very cheaply expressed on a real or Matsubara 
# axis to give access to the full Green's function and photoemission 
# spectrum at (an approximation to) the EOM-CCSD level of theory, 
# with broadening eta.
e = numpy.concatenate([gfcc.eh, gfcc.ep], axis=0)
v = numpy.concatenate([gfcc.vh[0], gfcc.vp[0]], axis=1)
u = numpy.concatenate([gfcc.vh[1], gfcc.vp[1]], axis=1)
grid = numpy.linspace(-5.0, 5.0, 100)
eta = 1e-2
denom = grid[:, None] - (e + numpy.sign(e) * eta * 1.0j)[None]
gf = lib.einsum("pk,qk,wk->wpq", v, u.conj(), 1.0/denom)
