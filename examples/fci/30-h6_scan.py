#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Energy curve by FCI
'''

from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci

for r in numpy.arange(2.4, 5.1, .2):
    mol = gto.M(
        atom = [['H', (0, 0, 0)],
                ['H', (r, 0, 0)],
                ['H', (0, r, 0)],
                ['H', (r, r, 0)],
                ['H', (0, r*2, 0)],
                ['H', (r, r*2, 0)]],
        basis = '6-31g',
        verbose = 0,
    )
    myhf = scf.RHF(mol).run()
    c = myhf.mo_coeff

    h1e = reduce(numpy.dot, (c.T, myhf.get_hcore(), c))
    eri = ao2mo.incore.full(myhf._eri, c)
    e, civec = fci.direct_spin0.kernel(h1e, eri, c.shape[1], mol.nelectron,
                                       tol=1e-14, lindep=1e-15, max_cycle=100)
    s2, m = fci.spin_op.spin_square(civec, c.shape[1], mol.nelectron)

    # or use the factory function fci.FCI generate a FCI object for any given orbitals
    myci = fci.FCI(mol, c)
    e, civec = myci.kernel()

    print('r = %f, E = %g, S^2 = %g, 2S+1 = %g' % (r, e, s2, m))
