#!/usr/bin/env python

"""Single-state IC-CASPT2 for triplet O2: incore and outcore calculations.

Reproduce the CASSCF and CASPT2 reference energies from the original
pyscf.mrpt.caspt2 module demonstration using O2/cc-pVDZ and CAS(8e,6o).
"""

from pyscf import gto, scf, mcscf
from pyscf.mrpt.caspt2 import CASPT2

mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='cc-pvdz', spin=2, symmetry='d2h', verbose=3)
mf = scf.RHF(mol).run(conv_tol=1E-16)

# Single-state triplet reference: eight active electrons in six orbitals.
mc = mcscf.CASSCF(mf, 6, 8)
mc.fcisolver.conv_tol = 1e-14
mc.conv_tol = 1e-12
# Canonicalize the core/virtual orbitals and use natural active orbitals.
mc.canonicalization = True
mc.natorb = True
mc.kernel()
assert abs(mc.e_tot - -149.708657771235039) < 1E-8

# Correlate all electrons, without an IPEA shift.
mr = CASPT2(mc, frozen=0, ipea_shift=0.0, root=0)
mr.einsum_backend = 'numpy'  # No optional contraction backend required.
mr.io_method = 'incore'
mr.verbose = 4
mr.max_cycle = 20
mr.kernel()
assert abs(mr.e_tot - -149.98063465606262) < 1E-8

print('Incore CASPT2 total energy:', mr.e_tot)

# Repeat with disk storage for ccvv amplitudes. Each contraction block
# spans three core orbitals; both paths should reproduce the same energy.
mr.io_method = 'outcore'
mr.io_blksize = 3
mr.kernel()
assert abs(mr.e_tot - -149.98063465606262) < 1E-8
print('Outcore CASPT2 total energy:', mr.e_tot)
