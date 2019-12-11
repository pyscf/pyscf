#!/usr/bin/env python

'''
MO integral transformation for spinor integrals
'''

import h5py
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.ao2mo import r_outcore

mol = gto.M(
    atom = '''
    O   0.   0.       0.
    H   0.   -0.757   0.587
    H   0.   0.757    0.587
    ''',
    basis = 'sto3g',
)

mf = scf.DHF(mol).run()

n4c, nmo = mf.mo_coeff.shape
n2c = n4c // 2
nNeg = nmo // 2
mo = mf.mo_coeff[:n2c,nNeg:]
r_outcore.general(mol, (mo, mo, mo, mo), 'llll.h5', intor='int2e_spinor')
with h5py.File('llll.h5', 'r') as f:
    print('Number of DHF molecular orbitals %s' % (nmo//2))
    print('MO integrals for large component orbitals. Shape = %s'
          % str(f['eri_mo'].shape))
