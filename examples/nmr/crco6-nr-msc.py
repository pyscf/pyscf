#!/usr/bin/env python
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import nmr

mol = gto.Mole()
mol.verbose = 5
bco = 1.14
bcc = 2.0105
mol.output = 'crco6-msc.out'
mol.atom = [
    ['Cr',(  0.000000,  0.000000,  0.000000)],
    ['C', (  bcc     ,  0.000000,  0.000000)],
    ['O', (  bcc+bco ,  0.000000,  0.000000)],
    ['C', ( -bcc     ,  0.000000,  0.000000)],
    ['O', ( -bcc-bco ,  0.000000,  0.000000)],
    ['C', (  0.000000,  bcc     ,  0.000000)],
    ['O', (  0.000000,  bcc+bco ,  0.000000)],
    ['C', (  0.000000, -bcc     ,  0.000000)],
    ['O', (  0.000000, -bcc-bco ,  0.000000)],
    ['C', (  0.000000,  0.000000,  bcc     )],
    ['O', (  0.000000,  0.000000,  bcc+bco )],
    ['C', (  0.000000,  0.000000, -bcc     )],
    ['O', (  0.000000,  0.000000, -bcc-bco )],
]

mol.basis = {'Cr': 'ccpvtz',
             'C' : 'ccpvtz',
             'O' : 'ccpvtz',}
mol.build()
nrscf = scf.RHF(mol)
nrscf.level_shift = .5
nrscf.diis_start_cycle = 2
nrscf.conv_tol = 1e-9
e = nrscf.scf()
print('E = %.15g, ref = -1719.915851708' % e)

import grad
g = grad.hf.RHF(nrscf)
print(g.grad())

m = nmr.hf.MSC(nrscf)
m.cphf = False
m.gauge_origin = None
print(m.msc())

