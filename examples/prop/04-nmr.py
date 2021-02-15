#!/usr/bin/env python

'''
Computing NMR shielding constants
'''

from pyscf import gto, dft
from pyscf.prop import nmr
mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            ''',
            basis='ccpvdz', verbose=3)
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.run()

nmr.RKS(mf).kernel()

