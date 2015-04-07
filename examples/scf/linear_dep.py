import numpy
from pyscf import gto, scf
mol = gto.Mole()
mol.verbose = 3
mol.atom = [['H', (0, 0, i*1.0)] for i in range(10)]
mol.basis = 'aug-ccpvdz'
mol.build()

mf = scf.RHF(mol)
print('condition number', numpy.linalg.cond(mf.get_ovlp()))
mf.scf()
