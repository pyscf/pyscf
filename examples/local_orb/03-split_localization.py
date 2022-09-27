#!/usr/bin/env python

'''
Boys localization, Edmiston-Ruedenberg localization and Pipek-Mezey
localization
'''

import numpy
from pyscf import gto, scf
from pyscf import lo
from pyscf.tools import molden

mol = gto.M(
    atom = '''
C    0.000000000000     1.398696930758     0.000000000000
C    0.000000000000    -1.398696930758     0.000000000000
C    1.211265339156     0.699329968382     0.000000000000
C    1.211265339156    -0.699329968382     0.000000000000
C   -1.211265339156     0.699329968382     0.000000000000
C   -1.211265339156    -0.699329968382     0.000000000000
H    0.000000000000     2.491406946734     0.000000000000
H    0.000000000000    -2.491406946734     0.000000000000
H    2.157597486829     1.245660462400     0.000000000000
H    2.157597486829    -1.245660462400     0.000000000000
H   -2.157597486829     1.245660462400     0.000000000000
H   -2.157597486829    -1.245660462400     0.000000000000''',
    basis = '6-31g')
mf = scf.RHF(mol).run()

pz_idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
loc_orb = lo.Boys(mol, mf.mo_coeff[:,pz_idx]).kernel()
molden.from_mo(mol, 'boys.molden', loc_orb)

loc_orb = lo.ER(mol, mf.mo_coeff[:,pz_idx]).kernel()
molden.from_mo(mol, 'edmiston.molden', loc_orb)

loc_orb = lo.PM(mol, mf.mo_coeff[:,pz_idx]).kernel()
molden.from_mo(mol, 'pm.molden', loc_orb)

