# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
Moment-constrained GF-CCSD with reuse of moments or custom moment input
from other level of theory.

Ref: Backhouse, Booth, arXiv:2206.13198 (2022).
"""

from pyscf import gto, scf, cc
import numpy as np

# Define system
mol = gto.Mole()
mol.atom = "Li 0 0 0; H 0 0 1.64"
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
gfcc = cc.MomGFCCSD(ccsd, niter=(3, 5))
gfcc.kernel()
ip = gfcc.ipgfccsd(nroots=1)[0]

# We can also build the moments ahead of time, and
# pass them in as the moment constraints, with the
# subsequent GF construction agnostic to the 
# provenance of these moments.
th = gfcc.build_hole_moments()
tp = gfcc.build_part_moments()
gfcc = cc.MomGFCCSD(ccsd, niter=(3, 5))
gfcc.kernel(hole_moments=th, part_moments=tp)
assert np.allclose(ip, gfcc.ipgfccsd(nroots=1)[0])

# Or use custom moments of the Green's function, via ndarrays which
# must enumerate at least the moments of order 0 through 2n+1 where
# n is the `niter` parameter in each sector. Note that physical
# moments must be in an orthogonal basis.

# For example, moments of the Hartree--Fock Green's function (powers
# of the Fock matrix), which should give exactly the MO energies
# (the other states will be linearly dependent):
f = np.diag(mf.mo_energy)
t = np.array([np.linalg.matrix_power(f, n) for n in range(5*2+2)])
th, tp = t.copy(), t.copy()
th[:, ccsd.nocc:, ccsd.nocc:] = 0.0
tp[:, :ccsd.nocc, :ccsd.nocc] = 0.0
gfcc = cc.MomGFCCSD(ccsd, niter=(3, 5))
gfcc.kernel(hole_moments=th, part_moments=tp)

# Or, moments from another post-HF Green's function method to
# approximate its spectrum, i.e. AGF2:
from pyscf.agf2 import AGF2
agf2 = AGF2(mf)
agf2.kernel()
gf = agf2.gf
th = gf.get_occupied().moment(np.arange(3*2+2))
tp = gf.get_virtual().moment(np.arange(5*2+2))
gfcc = cc.MomGFCCSD(ccsd, niter=(3, 5))
gfcc.hermi_moments = True
gfcc.hermi_solver = True
gfcc.kernel(hole_moments=th, part_moments=tp)
