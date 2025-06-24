#!/usr/bin/env python

'''
Example calculation using the DF-UMP2 code.

This package includes two implementations of the Density-Fitted MP2 method,
provided by the modules pyscf.mp.dfump2 and pyscf.mp.dfump2_native. 
The dfump2 module is based on the APIs and structure of the pyscf.mp.mp2 code,
while the dfump2_native module provides a different API. 

Other differences include:
- The dfump2 module is generally more efficient in memory usage and OpenMP parallelization.
- The dfump2_native module supports the computation of the relaxed one-particle density matrix.

Relevant examples:
10-dfmp2.py 10-dfump2.py 10-dfgmp2.py 11-dfmp2-density.py 12-dfump2-natorbs.py
'''


mol = pyscf.M(atm='''
O'   0.    0.  0.
O'   1.21  0.  0.''',
spin=2,
basis='def2-SVP',
verbose=4)

mf = mol.UHF().run()

# Option 1: Utilize the dfump2.DFMP2 implementation via the mf.DFMP2 function
mf.DFMP2().run()

# When mean-field calculation is a density fitting HF method, the .DFMP2()
# method is identical to the standard .MP2 method
mf = mf.density_fit().run()
mf.MP2().run()

# Option 2: Use the native DFMP2 implementation
from pyscf.mp.dfump2_native import DFUMP2
DFUMP2(mf).run()

# Executing within a "with" context for cleaning up temporary files
with DFUMP2(mf) as pt:
    pt.kernel()
    natocc, _ = pt.make_natorbs()
    print()
    print(natocc)
