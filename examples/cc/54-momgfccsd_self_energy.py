# Author: Oliver Backhouse <olbackhouse@gmail.com>

"""
Directly construct a pole representation of the 
self-energy via an implicit Dyson equation
for a Green's function computed at the CCSD level
via moment-constrained GFCCSD.

Ref: Backhouse, Booth, arXiv:2206.13198 (2022).
"""

import numpy as np
from pyscf import gto, scf, cc, lib
import scipy.linalg

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

# Run GF-CCSD:
gfcc = cc.MomGFCCSD(ccsd, niter=(4, 4))
gfcc.kernel()
ip = gfcc.ipgfccsd(nroots=1)[0]
ea = gfcc.eagfccsd(nroots=1)[0]

# Transform the Green's function poles from the GFCCSD calculation
# to poles of the self-energy. With the moment-conserving
# GFCCSD solver, this can be done statically without the need to
# numerically solve the Dyson equation. This procedure is described
# in arXiv:2206.13198 (2022).

# Combine hole and particle excitations:
e = np.concatenate([gfcc.eh, gfcc.ep], axis=0)
v = np.concatenate([gfcc.vh[0], gfcc.vp[0]], axis=1).T.conj()
u = np.concatenate([gfcc.vh[1], gfcc.vp[1]], axis=1).T.conj()

# Biorthogonalise physical vectors:
m = np.dot(v.T.conj(), u)
mv, mu = scipy.linalg.lu(m, permute_l=True)
v = np.dot(np.linalg.inv(mv), v.T.conj()).T.conj()
u = np.dot(u, np.linalg.inv(mu))

# Find a basis for the null space:
i = np.eye(u.shape[0]) - np.dot(v, u.T.conj())
w, v_rest = np.linalg.eig(i)
u_rest = np.linalg.inv(v_rest).T.conj()
u_rest = u_rest[:, np.abs(w) > 0.5] * w[np.abs(w) > 0.5][None]
v_rest = v_rest[:, np.abs(w) > 0.5] * w[np.abs(w) > 0.5][None]

# Biorthogonalise external vectors:
i = np.eye(u.shape[0]) - np.dot(v, u.T.conj())
w, v_rest = np.linalg.eig(i)
u_rest = np.linalg.inv(v_rest).T.conj()
u_rest = u_rest[:, np.abs(w) > 0.5] * w[np.abs(w) > 0.5][None]
v_rest = v_rest[:, np.abs(w) > 0.5] * w[np.abs(w) > 0.5][None]

# Combine physical and external vectors:
u = np.block([u, u_rest])
v = np.block([v, v_rest])

# Construct Hamiltonian, and rotate into arrowhead form:
h = np.dot(v.T.conj() * e[None], u)
w, v = np.linalg.eig(h[gfcc.nmo:, gfcc.nmo:])
v = np.block([
    [np.eye(gfcc.nmo), np.zeros((gfcc.nmo, w.size))],
    [np.zeros((w.size, gfcc.nmo)), v],
])
h = np.linalg.multi_dot((np.linalg.inv(v), h, v))

# Extract blocks:
phys = h[:gfcc.nmo, :gfcc.nmo]            # Static part of the self-energy
e_aux = np.diag(h[gfcc.nmo:, gfcc.nmo:])  # Energies of the self-energy
v_aux = h[:gfcc.nmo, gfcc.nmo:]           # Left couplings of the self-energy
u_aux = h[gfcc.nmo:, :gfcc.nmo].T.conj()  # Right couplings of the self-energy

# Diagonalise the self-energy to check the energies match:
e, v = np.linalg.eig(h)
e = e[np.einsum("xi,ix->i", v[:gfcc.nmo], np.linalg.inv(v)[:, :gfcc.nmo]).real > gfcc.weight_tol]
e = np.sort(e.real)

print("IP directly from GFCCSD:", ip)
print("IP recovered from self-energy:", -np.max(e[e < 0.5*(ea-ip)]))
