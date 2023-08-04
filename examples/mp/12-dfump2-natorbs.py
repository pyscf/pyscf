#!/usr/bin/env python
#

'''
DF-MP2 natural orbitals for the allyl radical
'''

from pyscf.gto import Mole
from pyscf.scf import UHF
from pyscf.tools import molden
from pyscf.mp.dfump2_native import DFMP2

mol = Mole()
mol.atom = '''
C    -1.1528    -0.1151    -0.4645
C     0.2300    -0.1171    -0.3508
C     0.9378     0.2246     0.7924
H     0.4206     0.5272     1.7055
H     2.0270     0.2021     0.8159
H    -1.6484    -0.3950    -1.3937
H    -1.7866     0.1687     0.3784
H     0.8086    -0.4120    -1.2337
'''
mol.basis = 'def2-TZVP'
mol.spin = 1
mol.build()

mf = UHF(mol).run()

# MP2 natural occupation numbers and natural orbitals
natocc, natorb = DFMP2(mf).make_natorbs()
# store the natural orbitals in a molden file
molden.from_mo(mol, 'allyl_mp2nat.molden', natorb, occ=natocc)