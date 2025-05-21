#!/usr/bin/env python

'''
Example calculation using the DF-RMP2 code.

This package includes two implementations of the Density-Fitted MP2 method,
provided by the modules pyscf.mp.dfmp2 and pyscf.mp.dfmp2_native. 
The dfmp2 module is based on the APIs and structure of the pyscf.mp.mp2 code,
while the dfmp2_native module provides a different API. 

Other differences include:
- The dfmp2 module is generally more efficient in memory usage and OpenMP parallelization.
- The dfmp2_native module supports the computation of the relaxed one-particle density matrix.

Relevant examples:
10-dfmp2.py 10-dfump2.py 10-dfgmp2.py 11-dfmp2-density.py 12-dfump2-natorbs.py
'''

import pyscf

mol = pyscf.M(
atom = '''
H   0.0   0.0   0.0
F   0.0   0.0   1.1
''',
basis = 'cc-pVDZ')

mf = mol.RHF().run()

# Option 1: Utilize the dfmp2.DFMP2 implementation via the mf.DFMP2 function
mf.DFMP2().run()

# When mean-field calculation is a density fitting HF method, the .DFMP2()
# method is identical to the standard .MP2 method
mf = mf.density_fit().run()
mf.MP2().run()

# Option 2: Use the native DFMP2 implementation
from pyscf.mp.dfmp2_native import DFMP2
DFMP2(mf).run()

# The dfmp2_native MP2 implementation can be executed within a "with" context.
# Temporary files will be automatically deleted upon exiting the context.
with DFMP2(mf) as pt:
    pt.kernel()
    natocc, _ = pt.make_natorbs()
    print()
    print(natocc)
