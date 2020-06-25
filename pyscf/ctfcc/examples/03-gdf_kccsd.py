#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Kpoint KCCSD with GDF integrals
Usage: mpirun -np 4 python 03-gdf_kccsd.py
'''
from mpi4py import MPI
from pyscf.pbc import scf, gto
from pyscf.ctfcc import KRCCSD
from pyscf.ctfcc.integrals import mpigdf
from pyscf.ctfcc.mpi_helper import rank, comm

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.mesh = [15,15,15]
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KRHF(cell, kpts, exxdiv=None)

mydf = mpigdf.GDF(cell, kpts)
mydf.build()
mydf.dump_to_file()

mf.with_df = mydf

if rank==0:
    mf.kernel()

comm.barrier()

mycc = KRCCSD(mf)
mycc.kernel()

mycc.ipccsd(nroots=2, kptlist=[0], koopmans=True)
mycc.eaccsd(nroots=2, kptlist=[0], koopmans=True)
