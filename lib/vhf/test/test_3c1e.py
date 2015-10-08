#!/usr/bin/env python
import numpy
from pyscf import gto, dft

mol = gto.M(atom=[
    ["O" , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)]], basis='ccpvdz')

grids = dft.gen_grid.Grids(mol)
#grids.level = 6
#grids.atom_grid = {'O': (100, 770), 'H': (100, 770)}
grids.build_()
ao = dft.numint.eval_ao(mol, grids.coords)
mat0 = numpy.einsum('pi,pj,pk,p->ijk', ao, ao, ao, grids.weights)

nao = mol.nao_nr()
mat1 = numpy.zeros((nao,nao,nao))
ip = 0
for ish in range(mol.nbas):
    jp = 0
    for jsh in range(mol.nbas):
        kp = 0
        for ksh in range(mol.nbas):
            buf = mol.intor_by_shell('cint3c1e_sph', (ish,jsh,ksh))
            di, dj, dk = buf.shape
            mat1[ip:ip+di,jp:jp+dj,kp:kp+dk] = buf
            val= numpy.linalg.norm(mat0[ip:ip+di,jp:jp+dj,kp:kp+dk]-buf)
            if val > 1e-4:
                print ish,jsh,ksh
                print mat0[ip:ip+di,jp:jp+dj,kp:kp+dk]
                print buf
                #exit()
            kp += dk
        jp += dj
    ip += di
print abs(mat0-mat1).sum()
print numpy.linalg.norm(mat0-mat1)
