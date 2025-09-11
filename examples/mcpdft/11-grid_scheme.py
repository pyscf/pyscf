#!/usr/bin/env python

from pyscf import gto, scf, dft
from pyscf import mcpdft

# See also pyscf/examples/dft/11-grid-scheme.py

mol = gto.M(
    verbose = 0,
    atom = '''
    o    0    0.       0.
    h    0    -0.757   0.587
    h    0    0.757    0.587''',
    basis = '6-31g')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF(mf, 'tLDA', 4, 4).run()
print('Default grid setup.  E = %.12f' % mc.e_tot)

# See pyscf/dft/radi.py for more radial grid schemes
mc.grids.radi_method = dft.mura_knowles
print('radi_method = mura_knowles.  E = %.12f' % mc.kernel()[0])

# All in one command:
mc = mcpdft.CASSCF (mf, 'tLDA', 4, 4,
                    grids_attr={'radi_method': dft.gauss_chebyshev})
print('radi_method = gauss_chebyshev.  E = %.12f' % mc.kernel ()[0])

# Or inline with an already-built mc object:
e = mc.compute_pdft_energy_(grids_attr={'radi_method': dft.delley})[0]
print('radi_method = delley.  E = %.12f' % e)

# All grids attributes can be addressed in any of the ways above
# There is also a shortcut to address grids.level:

mc = mcpdft.CASSCF(mf, 'tLDA', 4, 4, grids_level=4).run()
print('grids.level = 4.  E = %.12f' % mc.e_tot)

e = mc.compute_pdft_energy_(grids_level=2)[0]
print('grids.level = 2.  E = %.12f' % e)

