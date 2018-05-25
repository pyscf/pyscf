#!/usr/bin/env python


'''
PM localization for PBC systems.  It supports gamma point only.
'''

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf import lo

cell = gto.Cell()
cell.atom = '''
 C                  3.17500000    3.17500000    3.17500000
 H                  2.54626556    2.54626556    2.54626556
 H                  3.80373444    3.80373444    2.54626556
 H                  2.54626556    3.80373444    3.80373444
 H                  3.80373444    2.54626556    3.80373444
'''
cell.basis = 'sto3g'
cell.a = np.eye(3) * 6.35
cell.build()

mf = scf.RHF(cell).density_fit().run()
lmo = lo.PM(cell, mf.mo_coeff[:,1:5]).kernel()

nk = [1, 1, 1]
abs_kpts = cell.make_kpts(nk)
kmf = scf.KRHF(cell, abs_kpts).density_fit().run()
lmo = lo.PM(cell, kmf.mo_coeff[0][:,1:5]).kernel()
