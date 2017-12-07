r'''
Overlap of the cell-periodic part of AO Bloch function.

AO(k,r) = \sum_T e^{i k \dot T} \mu(r-T)
AO_{Bloch}(k,r) = e^{-i k\dot r) AO(k,r)

S_{k1,k2} = \int AO(k1,r) AO(k2,r) dr
'''

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import df

cell = gto.M(atom='H 0 0 0; H 1 0 1; H 0 1 0; H 1 .5 .5',
             basis=[[0, (1, 1)], [1, (1.2, 1)]],
             a=np.eye(3)*3)
k1, k2 = np.random.random((2,3))

#
# This is analytic FT to mimic this overlap integration
#
s_ft = df.ft_ao.ft_aopair(cell, -k1+k2, kpti_kptj=[k1,k2], q=np.zeros(3))

#
# Test the overlap using numerical integration
#
from pyscf.pbc.dft import gen_grid, numint
grids = gen_grid.UniformGrids(cell)
grids.build()
ao = numint.eval_ao(cell, grids.coords, kpt=k1)
u_k1 = np.einsum('x,xi->xi', np.exp(-1j*np.dot(grids.coords, k1)), ao)
ao = numint.eval_ao(cell, grids.coords, kpt=k2)
u_k2 = np.einsum('x,xi->xi', np.exp(-1j*np.dot(grids.coords, k2)), ao)
s_ref = np.einsum('x,xi,xj->ij', grids.weights, u_k1.conj(), u_k2)

print('max. Diff', abs(s_ref-s_ft).max())
