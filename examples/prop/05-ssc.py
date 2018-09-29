#!/usr/bin/env python

'''
Computing nuclear spin-spin coupling constants
'''

from pyscf import gto, scf
from pyscf.prop import ssc
mol = gto.M(atom='''
            C 0 0 0
            O 0 0 1.1747
            ''',
            basis='ccpvdz', verbose=3)

mf = scf.UHF(mol)
mf.run()

ssc.UHF(mf).kernel()

