#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf
from pyscf import fci

'''
Local spin <Psi |a><a| S^2 |Psi>
'''


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', (0., 0.,  0.7)],
    ['O', (0., 0., -0.7)],]
mol.basis = '6-31g'
mol.build()
mf = scf.RHF(mol)
mf.kernel()

ncas = 4
nelec = (4,2)
mc = mcscf.casci.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([5,7,8,9])
mocas = mo[:,5:9]
e, ecas, ci0 = mc.casci(mo)
print('RHF-CASCI total energy of O2', e)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', (0., 0.,  0.7)],
    ['O', (0., 0., -0.7)],]
mol.basis = '6-31g'
mol.spin = 2
mol.build()
mf = scf.UHF(mol)
mf.kernel()

ncas = 4
nelec = (4,2)
mc = mcscf.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([5,6,8,9])
mocas = mo[:,5:9]
e, ecas, ci0 = mc.casci(mo)
print('\n')
print('UHF-CASCI total energy of O2', e)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin (CAS space) for O = %.7f, 2S+1 = %.7f' % ss)


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', (0., 0.,  0.7)],
    ['O', (0., 0., -0.7)],
    ['H', (8, 0.,  0.7)],
    ['H', (8, 0., -0.7)],]

mol.basis = '6-31g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ncas = 6
nelec = (5,3)
mc = mcscf.casci.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([5,7,8,9,10,11])
mocas = mo[:,5:11]
e, ecas, ci0 = mc.casci(mo)
print('\n')
print('RHF-CASCI total energy of O2+H2 %.12f' % e)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,22))
print('local spin for H2 = %.7f, 2S+1 = %.7f' % ss)


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', (0., 0.,  0.7)],
    ['O', (0., 0., -0.7)],
    ['O', (8, 0.,  0.7)],
    ['O', (8, 0., -0.7)],]

mol.basis = '6-31g'
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ncas = 8
nelec = (6,6)
mc = mcscf.casci.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([9,10,13,14,15,16,17,18])
mocas = mo[:,10:18]
e, ecas, ci0 = mc.casci(mo)
print('\n')
print('RHF-CASCI total energy of O2+O2 singlet %.12f, which is incorrect' % e)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,36))
print('local spin for another O2 = %.7f, 2S+1 = %.7f' % ss)


ncas = 8
nelec = (8,4)
mc = mcscf.casci.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([9,10,13,14,15,16,17,18])
mocas = mo[:,10:18]
e, ecas, ci0 = mc.casci(mo)
print('\n')
print('RHF-CASCI total energy of O2+O2 quintet %.12f = monomer*2, correct' % e)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,36))
print('local spin for another O2 = %.7f, 2S+1 = %.7f' % ss)

