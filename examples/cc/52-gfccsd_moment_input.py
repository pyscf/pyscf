# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
Moment-constrained GF-CCSD with reuse of moments or custom moment input.

Ref: Backhouse, Booth, arXiv:2206.13198 (2022).
"""

from pyscf import gto, scf, cc
import numpy as np

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

# Run a moment-constrained GF-CCSD calculation
# Note: 5 cycles of moment constraint in the EA
# sector compared to 3 in the IP sector.
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 5))
gfcc.kernel()
ip = gfcc.ipccsd(nroots=1)[0]

# We can also build the moments ahead of time, and
# pass them in as the moment constraints, with the
# subsequent GF construction agnostic to the 
# provenance of these moments.
th = gfcc.build_hole_moments()
tp = gfcc.build_part_moments()
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 5))
gfcc.kernel(hole_moments=th, part_moments=tp)
assert np.allclose(ip, gfcc.ipccsd(nroots=1)[0])

# Or use custom moments, which must enumerate at least the moments
# of order 0 through 2n+1 where n is the `niter` parameter in each
# sector (since these are not physical this example will typically
# spit out a bunch of warnings and complex pole positions!)
# Note that physical moments must be in an orthogonal basis.
th = np.random.random((3*2+2, ccsd.nmo, ccsd.nmo))
tp = np.random.random((5*2+2, ccsd.nmo, ccsd.nmo))
gfcc = cc.gfccsd.GFCCSD(ccsd, niter=(3, 5))
gfcc.kernel(hole_moments=th, part_moments=tp)
