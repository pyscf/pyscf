#!/usr/bin/env python

'''
DFT+U with kpoint sampling
'''

from pyscf.pbc import gto, dft
cell = gto.Cell()
cell.unit = 'A'
cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
cell.a = '''0.      1.7834  1.7834
            1.7834  0.      1.7834
            1.7834  1.7834  0.    '''

cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.verbose = 4
cell.build()

kmesh = [2, 2, 2]
kpts = cell.make_kpts(kmesh)
# Add U term to the 2p orbital of the second Carbon atom
U_idx = ['1 C 2p']
U_val = [5.0]
mf = dft.KRKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, minao_ref='gth-szv')
print(mf.U_idx)
print(mf.U_val)
mf.run()

# When space group symmetry in k-point samples is enabled, the symmetry adapted
# DFT+U method will be invoked automatically.
kpts = cell.make_kpts(
    kmesh, wrap_around=True, space_group_symmetry=True, time_reversal_symmetry=True)
# Add U term to 2s and 2p orbitals
U_idx = ['2p', '2s']
U_val = [5.0, 2.0]
mf = dft.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, minao_ref='gth-szv')
print(mf.U_idx)
print(mf.U_val)
mf.run()
