#!/usr/bin/env python

'''
Use multi-grid to accelerate DFT numerical integration.

Relevant examples: 27-multigrid2.py, 31-pbc_0D_as_mol.py
'''

import numpy
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid

cell = gto.M(
    verbose = 4,
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'gth-dzvp',
    pseudo = 'gth-pade'
)

mf = dft.UKS(cell)
mf.xc = 'lda,vwn'

# multigrid integrator supports only the semi-local XC functionals (LDA, GGA and
# meta-GGA). For these functionals, the multigrid_numint() method automatically
# configures the required settings to enable multigrid integration algorithm.
# Specifically, the mf._numint attribute is replaced by an instance of the
# MultiGridNumInt class. This change does not affect the HFX integrator.
mf = mf.multigrid_numint()
mf.kernel()

# Multigrid integrator supports k-point sampling calculations.
kpts = cell.make_kpts([4,4,4])
mf = dft.KRKS(cell, kpts)
mf.xc = 'lda,vwn'
mf = mf.multigrid_numint()
mf.kernel()

# The multigrid integrator is compatible with the second-order (Newton) SCF
# solver. The .newton()) method can be applied to multigrid-based DFT
# calculations in the same way as for standard SCF methods.
mf = mf.newton()
mf.kernel()

# The multigrid integrator also supports analytical nuclear gradients.
# Gradient calculations can be enabled using the same API as in standard
# gradient evaluations.
mf.Gradients().run()
