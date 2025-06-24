#!/usr/bin/env python

'''
In TDDFT calculations, the contribution from NLC part is typically very small (< 0.1 meV),
and is disabled by default. This examples shows how to enable the NLC part in
TDDFT calculations.
'''

import pyscf

mol = gto.M(
    atom = '''
    O  0.0000  0.7375 -0.0528
    O  0.0000 -0.7375 -0.1528
    H  0.8190  0.8170  0.4220
    H -0.8190 -0.8170  0.4220
    ''',
    basis = 'def2-svp'
)
mol.verbose = 4

mf = mol.RKS(xc='wb97x-v').run()

mytd = tddft.TDA(mf)
mytd.exclude_nlc = False
mytd.kernel()

mytd = tddft.TDDFT(mf)
mytd.exclude_nlc = False
mytd.kernel()
