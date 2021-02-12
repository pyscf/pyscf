import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.tools
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc import cc
from pyscf import embcc

cell = gto.Cell()
cell.a = 3.567 * np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]])/2
cell.atom = "C 0.0 0.0 0.0 ; C 0.25 0.25 0.25"
cell.precision = 1e-5
cell.verbose = 4
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pade"
cell.build()

cell = pyscf.pbc.tools.super_cell(cell, [1, 1, 2])

mf = scf.RHF(cell)
# For GDF:
#mf = mf.density_fit()
mf.kernel()
print("E(HF)= %.8f" % mf.e_tot)

# CCSD benchmark
ccsd = cc.CCSD(mf)
ccsd.kernel()
print("E(CCSD)= %.8f" % ccsd.e_tot)

# EmbCC
ccx = embcc.EmbCC(mf, minao="gth-szv", dmet_bath_tol=1e-4, bath_tol=1e-5)
ccx.make_atom_cluster(0, symmetry_factor=cell.natm)
ccx.kernel()

print("Number of active orbitals: %d out of %d" % (ccx.clusters[0].nactive, ccx.norb))
print("E(EmbCC)=      %.8f  (E(corr) = %.2f %%)" % (ccx.e_tot, 100.0*ccx.e_corr/ccsd.e_corr))
print("E(EmbCC+dMP2)= %.8f  (E(corr) = %.2f %%)" % (ccx.e_tot + ccx.e_delta_mp2,
    100.0*(ccx.e_corr+ccx.e_delta_mp2)/ccsd.e_corr))
