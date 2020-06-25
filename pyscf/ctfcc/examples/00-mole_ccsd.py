#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
A simple example to run molecular RCCSD with ctf parallelization
Usage: mpirun -np 4 python 00-mole_ccsd.py
'''
from pyscf import scf, gto
from pyscf.ctfcc import RCCSD
from pyscf.ctfcc.mpi_helper import rank

mol = gto.Mole(atom = 'H 0 0 0; F 0 0 1.1',
               basis = 'ccpvdz',
               verbose= 5)

mf = scf.RHF(mol)

if rank==0: #SCF only needs to run on one process
    mf.kernel()

mycc = RCCSD(mf)
mycc.kernel()

mycc.ipccsd(nroots=2, koopmans=True)
mycc.eaccsd(nroots=2, koopmans=True)
