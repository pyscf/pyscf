#!/usr/bin/env python

'''
CCSD with K-point sampling
'''

from pyscf.pbc import gto, scf, mpicc
from pyscf.pbc.examples.scf import run_khf
from pyscf.pbc.tools import pyscf_ase
from mpi4py import MPI

from ase.lattice import bulk
A2B = 1.889725989
ase_atom = bulk('C', 'diamond', a=3.53034833533)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

cell = gto.M(
    h = ase_atom.cell,
    atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom),
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    gs = [12]*3,
    verbose = 6,
)

nk = [1,1,3]
kpts = cell.make_kpts(nk)
print "Grid size",
print cell.gs

#
# Running HF
#
kmf = run_khf(cell, nk, gamma=True, exxdiv=None, conv_tol=1e-12)

comm.Barrier()
mo_coeff  = comm.bcast(kmf.mo_coeff,root=0)
mo_energy = comm.bcast(kmf.mo_energy,root=0)
mo_occ    = comm.bcast(kmf.mo_occ,root=0)
kpts      = comm.bcast(kmf.kpts,root=0)
kmf.mo_coeff = mo_coeff
kmf.mo_energy = mo_energy
kmf.mo_occ = mo_occ
kmf.kpts   = kpts
comm.Barrier()

#
# Running CCSD
#
kcc = mpicc.KRCCSD(kmf)
ecc, t1, t2 = kcc.kernel()
if rank == 0:
    print("cc energy (per unit cell) = %.17g" % ecc)
# Running EACCSD and EACCSD*
lew, lev = kcc.leaccsd(nroots=1, kptlist=[0])
ew, ev   = kcc.eaccsd(nroots=1,  kptlist=[0])
kcc.eaccsd_star(ew, ev, lev)

# Running IPCCSD and IPCCSD*
lew, lev = kcc.lipccsd(nroots=1, kptlist=[0])
ew, ev   = kcc.ipccsd(nroots=1,  kptlist=[0])
kcc.ipccsd_star(ew, ev, lev)
