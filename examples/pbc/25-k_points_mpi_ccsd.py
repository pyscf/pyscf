#!/usr/bin/env python

'''
Showing equivalence between the K-point CCSD and
the gamma-point CCSD for a diamond lattice.
'''

from pyscf.pbc import gto, scf, mpicc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

cell = gto.M(
    unit = 'B',
    a = [[ 0.,          3.37013733,  3.37013733],
         [ 3.37013733,  0.,          3.37013733],
         [ 3.37013733,  3.37013733,  0.        ]],
    mesh = [24,]*3,
    atom = '''C 0 0 0
              C 1.68506866 1.68506866 1.68506866''',
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    verbose = 4,
)

nk = [2,2,2]
kpts = cell.make_kpts(nk)

#
# Running HF
#
kpts -= kpts[0]
kmf = scf.KRHF(cell, kpts)
if rank == 0:
    kmf.kernel()

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
kcc.eaccsd_star_contract(ew, ev, lev)

# Running IPCCSD and IPCCSD*
lew, lev = kcc.lipccsd(nroots=1, kptlist=[0])
ew, ev   = kcc.ipccsd(nroots=1,  kptlist=[0])
kcc.ipccsd_star_contract(ew, ev, lev)
