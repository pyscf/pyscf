#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf

'''
Second order SCF algorithm by decorating the scf object with scf.newton
function.  (New in PySCF-1.1)

Second order SCF method need orthonormal orbitals and the corresponding
occupancy as the initial guess.
'''

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-1
mf.kernel()
mo_init = mf.mo_coeff
mocc_init = mf.mo_occ

mf = scf.newton(scf.RHF(mol))
energy = mf.kernel(mo_init, mocc_init)
print('E = %.12f, ref = -76.026765672992' % energy)

mf = scf.UKS(mol).newton()  # Using stream style
energy = mf.kernel((mo_init,mo_init), (mocc_init*.5,mocc_init*.5))
print('E = %.12f, ref = -75.854689662296' % energy)


