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

rank = MPI.COMM_WORLD.Get_rank()

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
#kmf = scf.KRHF(cell, kpts, exxdiv=None)
#ehf = kmf.kernel()
kmf = run_khf(cell, nk, gamma=True, exxdiv=None, conv_tol=1e-12)

#
# Running CCSD
#
kcc = mpicc.KRCCSD(kmf)
ecc, t1, t2 = kcc.kernel()
if rank == 0:
    print("cc energy (per unit cell) = %.17g" % ecc)
kcc.eaccsd(nroots=1,kptlist=[0])
kcc.leaccsd(nroots=1,kptlist=[0])
