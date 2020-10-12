#!/usr/bin/env python

'''
An efficient einsum function using BLAS or TBLIS (https://github.com/devinamatthews/tblis)
as the tensor contraction engine. By default, pyscf.lib.einsum calls BLAS gemm
functions for tensor contraction. If tblis is available (see the installation
instruction http://pyscf.org/pyscf/install.html#tblis), pyscf.lib.einsum calls
TBLIS library at the end. The main advantage of TBLIS library is its small
memory overhead.
'''

import time
import numpy as np
from pyscf import lib

a = np.random.random((25,120,20,115))
b = np.random.random((115,83,25,40))

t0 = time.time()
np_results = np.einsum('xaby,ycxd->cadb', a, b)
print('Time for np.einsum %.2f s' % (time.time() - t0))

t0 = time.time()
pyscf_results = lib.einsum('xaby,ycxd->cadb', a, b)
print('Time for pyscf lib.einsum %.2f s' % (time.time() - t0))

print('Difference between two einsum function %.5g' % abs(pyscf_results - np_results).max())
