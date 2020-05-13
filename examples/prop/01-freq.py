#!/usr/bin/env python

'''
Frequency and normal modes
'''

from pyscf import gto, dft
from pyscf.prop.freq import rks

mol = gto.M(atom='''
            O 0 0      0
            H 0 -0.757 0.587
            H 0  0.757 0.587''',
            basis='ccpvdz', verbose=4)
mf = dft.RKS(mol).run()

w, modes = rks.Freq(mf).kernel()

