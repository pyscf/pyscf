#!/usr/bin/env python
#
# Author: Paul J. Robinson <pjrobinson@ucla.edu>
#

import numpy
from pyscf import lo
from pyscf.tools import chgcar
from pyscf.pbc import gto, scf
from functools import reduce
'''
This Benchmark show how to use the Cubegen command to make VASP chgcars for
visualizing periodic IBOs for example in diamond 
'''
cell = gto.Cell()
cell.a = '''
3.5668  0       0
0       3.5668  0
0       0       3.5668'''
cell.atom='''C 0.      0.      0.
   C 0.8917  0.8917  0.8917
   C 1.7834  1.7834  0.
   C 2.6751  2.6751  0.8917
   C 1.7834  0.      1.7834
   C 2.6751  0.8917  2.6751
   C 0.      1.7834  1.7834
   C 0.8917  2.6751  2.6751'''

cell.ke_cutoff = 100
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build(unit='Angstrom')
mf = scf.RHF(cell).run()


'''
generates IBOs
'''
mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
a = lo.iao.iao(cell, mo_occ)
# Orthogonalize IAO
a = lo.vec_lowdin(a, mf.get_ovlp())
#ibo must take the orthonormalized IAOs
ibo = lo.ibo.ibo(cell, mo_occ, iaos=a)


'''
Generates IBO files as VASP Chgcars
'''
for i in range(ibo.shape[1]):
    chgcar.orbital(cell, 'diamond_ibo{:02d}.chgcar'.format(i+1), ibo[:,i])
    print("wrote cube {:02d}".format(i+1))

'''
Makes Population Analysis with IAOs
'''
# transform mo_occ to IAO representation. Note the AO dimension is reduced
mo_occ = reduce(numpy.dot, (a.T, mf.get_ovlp(), mo_occ))

dm = numpy.dot(mo_occ, mo_occ.T) * 2
iao_mol = cell.copy()
iao_mol.build(False, False, basis='minao')
mullic = mf.mulliken_pop(iao_mol, dm, s=numpy.eye(iao_mol.nao_nr()))

