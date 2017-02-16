#!/usr/bin/env python

'''
CCSD with K-point sampling
'''

from pyscf.pbc import gto, scf, cc
from pyscf.pbc.tools import pyscf_ase

from ase.lattice import bulk
ase_atom = bulk('C', 'diamond', a=3.5668)

cell = gto.M(
    h = ase_atom.cell,
    atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom),
    basis = 'gth-szv',
    pseudo = 'gth-pade',
    gs = [10]*3,
    verbose = 4,
)

nk = [2,2,2]
kpts = cell.make_kpts(nk)

#
# Running HF
#
kmf = scf.KRHF(cell, kpts, exxdiv=None)
ehf = kmf.kernel()

#
# Running CCSD
#
kcc = cc.KCCSD(kmf)
ecc, t1, t2 = kcc.kernel()
print("cc energy (per unit cell) = %.17g" % ecc)

