#!/usr/bin/env python

'''
The addon kconj_symmetry_ can be used to perform k-point sampled SCF calculations
which retain the phi(k) = phi(-k)* symmetry of the MOs phi.
'''

import numpy as np

from pyscf.pbc import scf
from pyscf.pbc.gto import Cell
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.scf.addons import kconj_symmetry_

print("1D oxygen chain")
print("---------------")

# --- Example 1: 1D Oxygen chain with 3 k-points
# The (k,-k)-symmetry is only broken slightly. With the kconj_symmetry_ decorator,
# it can be restored without a significant change of the Hartree-Fock energy:

cell = Cell()
cell.atom = 'O 0 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[2,10,10]
cell.unit = 'A'
cell.output = '43-example-1.out'
cell.verbose = 4
cell.build()
kpts = cell.make_kpts([3,1,1])

mf = scf.KRHF(cell, kpts).density_fit()
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

def get_symmetry_error(mf):
    dm1 = mf.make_rdm1()
    conj_indices = kpts_helper.conj_mapping(mf.mol, mf.kpts)
    ddm1 = (dm1 - dm1[...,conj_indices,:,:].conj())
    err = np.linalg.norm(ddm1)
    return err

print("Example 1: 1 atom, 3 k-points:")
print("  Regular RHF:           E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf.e_tot, mf.converged, get_symmetry_error(mf)))
print("  (k,-k)-symmetric RHF:  E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf_sym.e_tot, mf_sym.converged, get_symmetry_error(mf_sym)))

del cell, kpts, mf, mf_sym

# --- Example 2: 1D Oxygen chain with 2 atoms and 4 k-points
# The (k,-k)-symmetry is significantly broken. With the kconj_symmetry_ decorator,
# a lower energy solution can be converged:

cell = Cell()
cell.atom = 'O 0 0 0 ; O 2 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[4,10,10]
cell.unit = 'A'
cell.output = '43-example-2.out'
cell.verbose = 4
cell.build()
kpts = cell.make_kpts([4,1,1])

mf = scf.KRHF(cell, kpts).density_fit()
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

print("Example 2: 2 atoms, 4 k-points:")
print("  Regular RHF:           E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf.e_tot, mf.converged, get_symmetry_error(mf)))
print("  (k,-k)-symmetric RHF:  E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf_sym.e_tot, mf_sym.converged, get_symmetry_error(mf_sym)))

del cell, kpts, mf, mf_sym

# --- Example 3: 1D Oxygen chain with 4 k-points
# In this case, no (k,-k)-symmetric solution can be converged, as the RHF solution has a
# degeneracy at the Fermi level, which can only be lifted by breaking (k,-k)-symmetry.
# Note that a warning is logged in this case and mf.converged equals False:

cell = Cell()
cell.atom = 'O 0 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[2,10,10]
cell.unit = 'A'
cell.output = '43-example-3.out'
cell.verbose = 4
cell.build()
kpts = cell.make_kpts([4,1,1])

mf = scf.KRHF(cell, kpts).density_fit()
mf.kernel()

mf_sym = scf.KRHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel()

print("Example 3: 1 atom, 4 k-points:")
print("  Regular RHF:           E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf.e_tot, mf.converged, get_symmetry_error(mf)))
print("  (k,-k)-symmetric RHF:  E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf_sym.e_tot, mf_sym.converged, get_symmetry_error(mf_sym)))

del cell, kpts, mf, mf_sym


# --- Example 4: 1D Oxygen chain with 2 atoms and 4 k-points, antiferromagnetic order
# Same as example 2, but with UHF an antiferromagnetic solution with lower energy
# can be found:

cell = Cell()
cell.atom = 'O 0 0 0 ; O 2 0 0'
cell.dimension = 1
cell.basis = '6-31G'
cell.a = np.eye(3)*[4,10,10]
cell.unit = 'A'
cell.output = '43-example-4.out'
cell.verbose = 4
cell.build()

kpts = cell.make_kpts([4,1,1])

def break_spinsym(cell, dm1, delta=0.05):
    """Break spin symmetry of density-matrix, to converge AFM order"""
    start, stop = cell.aoslice_by_atom()[:,[2,3]]
    atm1 = np.s_[start[0]:stop[0]]
    atm2 = np.s_[start[1]:stop[1]]
    dm1a, dm1b = dm1.copy()
    # Atom 1: Majority spin = alpha
    dm1a[:,atm1,atm1] += delta*dm1b[:,atm1,atm1]
    dm1b[:,atm1,atm1] -= delta*dm1b[:,atm1,atm1]
    # Atom 2: Majority spin = beta
    dm1a[:,atm2,atm2] -= delta*dm1a[:,atm2,atm2]
    dm1b[:,atm2,atm2] += delta*dm1a[:,atm2,atm2]
    return np.stack((dm1a, dm1b), axis=0)

mf = scf.KUHF(cell, kpts).density_fit()
dm0 = break_spinsym(cell, mf.get_init_guess())
mf.kernel(dm0=dm0)

mf_sym = scf.KUHF(cell, kpts)
mf_sym.with_df = mf.with_df
mf_sym = kconj_symmetry_(mf_sym)
mf_sym.kernel(dm0=dm0)

print("Example 4: 2 atoms, 4 k-points, AFM order:")
print("  Regular UHF:           E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf.e_tot, mf.converged, get_symmetry_error(mf)))
print("  (k,-k)-symmetric UHF:  E= % 14.8f  converged= %5r  symmetry error= %.1e" % (
    mf_sym.e_tot, mf_sym.converged, get_symmetry_error(mf_sym)))
