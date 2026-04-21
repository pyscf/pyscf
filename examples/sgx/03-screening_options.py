#!/usr/bin/env python

'''
This example shows options for tuning P-junction screening in SGX calculations.
'''

from pyscf import gto
from pyscf import scf
from pyscf import sgx


mol = gto.M(
    atom='''
        O    0.   0.       0.
        H    0.   -0.757   0.587
        H    0.   0.757    0.587
        O    8.   0.       0.
        H    8.   -0.757   0.587
        H    8.   0.757    0.587
    ''',
    basis = 'ccpvdz',
)
# Direct K matrix for comparison
mf = scf.RHF(mol)
mf.kernel()

# pjs=True is equivalent to setting
# mf.with_df.optk = True
# mf.with_df.dfj = True
# which turns on P-junction screening
mf = sgx.sgx_fit(scf.RHF(mol), pjs=True)
mf.kernel()

# sgx_tol_energy and sgx_tol_potential place **approximate**
# upper bounds on the error due to density matrix screening
# in the exchange energy and K-matrix elements, respectively.
# The default for both is "auto" (which sets sgx_tol_energy=direct_sct_tol
# and sgx_tol_potential=sqrt(sgx_tol_energy)).
# For more aggressive screening, we recommend sgx_tol_energy=conv_tol
# aqnd sgx_tol_potential="auto".
mf.with_df.sgx_tol_energy = mf.conv_tol
mf.with_df.sgx_tol_potential = "auto"
mf.with_df.build()
mf.kernel()

# In most cases, this should converge since the energy error is bounded
# by the convergence threshold. If you have convergence issues,
# the first thing to try is setting rebuild_nsteps=1. By default, with
# mf.direct_scf=True, the K-matrix is built incrementally from changes
# in the density matirx, and then is rebuilt from scratch every
# rebuilt_nsteps steps. Incremental Fock build can cause errors to
# accumulate in some cases, so turning off incremental build by
# settings rebuild_nsteps=1 can fix this.
mf.with_df.rebuild_nsteps = 1
mf.kernel()

# If you still have convergence problems, decreasing the sgx_tol_energy
# and sgx_tol_potential should improve stability of the SCF.
mf.rebuild_nsteps = 5
mf.with_df.sgx_tol_potential = 1e-10
mf.with_df.build()
mf.kernel()

# To not reset the K-matrix at any point during SCF (except for when
# the integration grid changes), set rebuild_nsteps > max_cycle.
# This is generally not recommended but often works anyway and
# can increase speed somewhat.
mf.with_df.rebuild_nsteps = 100
mf.kernel()

# When the SCF gets close to converging, SGX switches from grids_level_i
# to grids_level_f. By default, grids_level_i=grids_level_j=2 and no
# grids switch occurs. But using a coarser grid at first can speed up
# initial SCF cycles.
mf.with_df.grids_level_i = 0
mf.with_df.grids_level_f = 2
mf.kernel()

# There are three different ways to screening the negligible 3-center
# integrals. The default is "sample_pos", which is recommended
# in most cases.
for bound_algo in ["ovlp", "sample", "sample_pos"]:
    mf.with_df.bound_algo = bound_algo
    mf.with_df.build()
    mf.kernel()

# If dfj is off at runtime, optk is not used because optk cannot accelerate
# the J-matrix evaluation.
mf.with_df.dfj = False
mf.kernel()

