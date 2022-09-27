#!/usr/bin/env python
#
# Author: Paul J. Robinson <pjrobinson@ucla.edu>
#

'''
IBO generation, cube generation, and population analysis of benzene
'''

import numpy
from pyscf import gto, scf, lo, tools
from functools import reduce

benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

mol = gto.M(atom=benzene,
            basis='ccpvdz')
mf = scf.RHF(mol).run()

mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
a = lo.iao.iao(mol, mo_occ)

# Orthogonalize IAO
a = lo.vec_lowdin(a, mf.get_ovlp())

#
# Method 1, using Knizia's alogrithm to localize IAO orbitals
#
'''
Generate IBOS from orthogonal IAOs
'''

ibo = lo.ibo.ibo(mol, mo_occ, iaos=a)

'''
Print the IBOS into Gausian Cube files
'''

for i in range(ibo.shape[1]):
    tools.cubegen.orbital(mol, 'benzene_ibo1_{:02d}.cube'.format(i+1), ibo[:,i])

'''
Population Analysis with IAOS
'''
# transform mo_occ to IAO representation. Note the AO dimension is reduced
mo_occ = reduce(numpy.dot, (a.T, mf.get_ovlp(), mo_occ))

#constructs the density matrix in the new representation
dm = numpy.dot(mo_occ, mo_occ.T) * 2

#mullikan population analysis based on IAOs
iao_mol = mol.copy()
iao_mol.build(False, False, basis='minao')
mf.mulliken_pop(iao_mol, dm, s=numpy.eye(iao_mol.nao_nr()))

#
# Method 2, using the modified Pipek-Mezey localization module.
# Orthogonalization for IAOs is not required.
#
mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
iaos = lo.iao.iao(mol, mo_occ)
ibo = lo.ibo.ibo(mol, mo_occ, locmethod='PM', iaos=iaos).kernel()
for i in range(ibo.shape[1]):
    tools.cubegen.orbital(mol, 'benzene_ibo2_{:02d}.cube'.format(i+1), ibo[:,i])

