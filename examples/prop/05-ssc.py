#!/usr/bin/env python

'''
Computing nuclear spin-spin coupling constants
'''

from pyscf import gto, scf, dft
from pyscf.prop import ssc
mol = gto.M(atom='''
            O 0 0      0
            H 0 -0.757 0.587
            H 0  0.757 0.587''',
            basis='ccpvdz')

mf = scf.UHF(mol).run()
ssc.UHF(mf).kernel()

mf = dft.UKS(mol).set(xc='b3lyp').run()
ssc.UKS(mf).kernel()

