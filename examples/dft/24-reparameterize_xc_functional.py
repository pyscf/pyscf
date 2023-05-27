#!/usr/bin/env python

'''
Running DFT calculations with reparameterized functionals in Libxc.

See also
* Source code of the reparameterization library `24-reparameterize_xc_functional.c`;
* pyscf/lib/dft/libxc_itrf.c for how the callback is inserted;
* Example 24-custom_xc_functional.py to customize XC functionals using the
  functionals provided by Libxc or XcFun library.
'''

from pyscf import gto
from pyscf import dft
import ctypes
import numpy as np

mol = gto.M(
    atom = '''
    O  0.   0.       0.
    H  0.   -0.757   0.587
    H  0.   0.757    0.587 ''',
    basis = 'ccpvdz')

# Load the reparameterization library
try:
    param_lib = ctypes.cdll.LoadLibrary('24-reparameterize_xc_functional.so')
except Exception as e:
    print('Unable to load reparameterization library:')
    print(e)
    print()
    print('Follow the instruction in `24-reparameterize_xc_functional.c` to compile the library')
    exit(1)

# Run normal B97-1
print('Normal B97-1')
mf = dft.RKS(mol, 'B971')
e_b971 = mf.kernel()

# Run normal B97-2
print('\nNormal B97-2')
mf.xc = 'B972'
e_b972 = mf.kernel()

# Run normal B97-3
print('\nNormal B97-3')
mf.xc = 'B973'
e_b973 = mf.kernel()

# Install callback
dft.libxc.libxc_install_init_callback(param_lib.init_callback)

# Construct XC based on B97-2, but set its parameter to be B97-1
print('\nReparameterized B97-2: will be the same as B97-1')
mf.xc = 'B972'
param = np.array([0.789518, 0.573805, 0.660975, 0.0, 0.0,
                  0.0820011, 2.71681, -2.87103, 0.0, 0.0,
                  0.955689, 0.788552, -5.47869, 0.0, 0.0,
])
param_lib.set_param(param.ctypes)
e = mf.kernel()
print('difference:', e - e_b971)

# New parameters will retain until one manually removes it, even
# when a new `mf` object is created.
print('\nRerun reparameterized B97-2: will be the same as B97-1')
mf = dft.RKS(mol, 'B972')
e = mf.kernel()
print('difference:', e - e_b971)

# Set parameter to be B97-3 and rerun
# This is an example to modify the HF exchange percentage.
# Note that percentage of HF exchange is handled by PySCF.
# One needs to specify a new HF exchange percentage using the `mf.xc` property.
# Changing the HF exchange percentage in Libxc using the callback has no effect.
print('\nReparameterized B97-2: will be the same as B97-3')
# Put the DIFFERENCE of HF exchange between B97-3 and B97-2 in `mf.xc`
mf.xc = 'B972 + 0.059288 * HF'
param = np.array([0.7334648, 0.292527, 3.338789, -10.51158, 10.60907,
                  0.5623649, -1.32298, 6.359191, -7.464002, 1.827082,
                  1.13383, -2.811967, 7.431302, -1.969342, -11.74423,
])
param_lib.set_param(param.ctypes)
e = mf.kernel()
print('difference:', e - e_b973)

# Reset parameters
print('\nAfter removing callback: will be the same as normal B97-2')
dft.libxc.libxc_remove_init_callback()
mf.xc = 'B972'
e = mf.kernel()
print('difference:', e - e_b972)

