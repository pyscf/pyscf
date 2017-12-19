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
# The newton algorithm will automatically generate initial orbitals if initial
# guess is not given.
energy = mf.kernel()
print('E = %.12f, ref = -75.854702461713' % energy)



# Note You should first set mf.xc then apply newton method because this will
# correctly set up the underlying orbital gradients.  If you first create
# mf = mf.newton() then change mf.xc, the orbital Hessian will be computed
# with the updated xc functional while the orbital gradients are computed with
# the old xc functional.
# In some scenario, you can use this character to approximate the
# orbital Hessian, ie, computing the orbital Hessian with approximated XC
# functional if the accurate Hessian is not applicable.
mf = scf.UKS(mol)
mf.xc = 'pbe,pbe'
mf = mf.newton()
energy = mf.kernel()
print('E = %.12f, ref = -76.333457658990' % energy)

mf = scf.UKS(mol)
mf = mf.newton()
mf.xc = 'pbe,pbe'
energy = mf.kernel()
print('E = %.12f, ref = -75.854702461713' % energy)
