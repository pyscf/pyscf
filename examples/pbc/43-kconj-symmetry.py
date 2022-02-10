import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.gto import Cell
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.scf.addons import kconj_symmetry_

def get_symmetry_error(mf):
    dm1 = mf.make_rdm1()
    conj_indices = kpts_helper.conj_mapping(mf.mol, mf.kpts)
    ddm1 = (dm1 - dm1[conj_indices].conj())
    err = np.linalg.norm(ddm1)
    return err

# --- Example 1: 1D Oxygen chain with 4 k-points
# In this case, enforcing the (k,-k)-symmetry helps the SCF to converge
# to a lower energy solution

cell = Cell()
cell.atom = 'O 0 0 0 ; O 2 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[4,10,10]
cell.unit = 'A'
cell.output = '43-example-1.out'
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([4,1,1])

mf = scf.KRHF(cell, kpts).density_fit()
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

print("Regular RHF:          E= % .8f  symmetry error= %.5e" % (mf.e_tot, get_symmetry_error(mf)))
print("(k,-k)-symmetric RHF: E= % .8f  symmetry error= %.5e" % (mf_sym.e_tot, get_symmetry_error(mf_sym)))

del cell, kpts, mf, mf_sym

# --- Example 2: FeO in rocksalt structure with 3 x 3 x 3 k-points
# In this case, the (k,-k)-symmetry can be restored without a change in the Hartree-Fock energy

cell = Cell()
cell.a = 3.6 * np.asarray([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]])
internal = np.asarray([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]])
external = np.dot(internal, cell.a)
cell.atom = 'Fe %f %f %f ; O %f %f %f' % (*external[0], *external[1])
cell.basis = 'def2-svp'
cell.output = '43-example-2.out'
cell.verbose = 4
cell.exp_to_discard = 0.1
cell.build()

kpts = cell.make_kpts([3,3,3])

mf = scf.KRHF(cell, kpts).rs_density_fit(auxbasis='def2-svp-ri')
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

print("Regular RHF:          E= % .8f  symmetry error= %.5e" % (mf.e_tot, get_symmetry_error(mf)))
print("(k,-k)-symmetric RHF: E= % .8f  symmetry error= %.5e" % (mf_sym.e_tot, get_symmetry_error(mf_sym)))

del cell, kpts, mf, mf_sym

# --- Example 3: 1D Oxygen chain with 3 k-points
# In this case, no (k,-k)-symmetric solution can be converged, as the RHF solution has a
# degeneracy at the Fermi level, which can only be lifted by breaking (k,-k)-symmetry.
# Note the warning in the output in this case.

cell = Cell()
cell.atom = 'O 0 0 0 ; O 2 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[4,10,10]
cell.unit = 'A'
cell.output = '43-example-3.out'
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([3,1,1])

mf = scf.KRHF(cell, kpts).density_fit()
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

print("Regular RHF:          E= % .8f  symmetry error= %.5e" % (mf.e_tot, get_symmetry_error(mf)))
print("(k,-k)-symmetric RHF: E= % .8f  symmetry error= %.5e" % (mf_sym.e_tot, get_symmetry_error(mf_sym)))
