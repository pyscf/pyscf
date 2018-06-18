#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Local spin expectation value <Psi |(local S^2)|Psi >

The local S^2 operator only couples the orbitals specified in aolst. The
cross term which involves the interaction between the local part (in aolst)
and non-local part (not in aolst) is not included. As a result, the value
of local_spin is not additive. In other words, if local_spin is computed
twice with the complementary aolst in the two runs, the summation does not
equal to the S^2 of the entire system.
'''

import numpy
from pyscf import gto, scf, mcscf
from pyscf import fci


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
mc.kernel(mo)
print('RHF-CASCI total energy of O2', mc.e_tot)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ao_list_for_atom1 = mol.search_ao_label(['0 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1)
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
mc.kernel(mo)
print('\n')
print('UHF-CASCI total energy of O2', mc.e_tot)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ao_list_for_atom1 = mol.search_ao_label(['0 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1)
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
mc.kernel(mo)
print('\n')
print('RHF-CASCI total energy of O2+H2 %.12f' % mc.e_tot)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ao_list_for_atom1 = mol.search_ao_label(['0 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1)
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_atom2 = mol.search_ao_label(['1 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom2)
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_H = mol.search_ao_label([' H '])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_H)
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
mc.fix_spin_()
mo = mc.sort_mo([9,10,13,14,15,16,17,18])
mocas = mo[:,10:18]
mc.kernel(mo)
print('\n')
print('RHF-CASCI total energy of O2+O2 singlet %.12f = monomer*2' % mc.e_tot)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ao_list_for_atom1 = mol.search_ao_label(['0 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1)
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_atom1_2 = mol.search_ao_label(['0 O', '1 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1_2)
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_atom3_4 = mol.search_ao_label(['2 O', '3 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom3_4)
print('local spin for another O2 = %.7f, 2S+1 = %.7f' % ss)


ncas = 8
nelec = (8,4)
mc = mcscf.casci.CASCI(mf, ncas, nelec)
mo = mc.sort_mo([9,10,13,14,15,16,17,18])
mocas = mo[:,10:18]
mc.kernel(mo)
print('\n')
print('RHF-CASCI total energy of O2+O2 quintet %.12f = monomer*2' % mc.e_tot)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
ao_list_for_atom1 = mol.search_ao_label(['0 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1)
print('local spin for O = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_atom1_2 = mol.search_ao_label(['0 O', '1 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom1_2)
print('local spin for O2 = %.7f, 2S+1 = %.7f' % ss)
ao_list_for_atom3_4 = mol.search_ao_label(['2 O', '3 O'])
ss = fci.spin_op.local_spin(mc.ci, ncas, nelec, mocas, mf.get_ovlp(),
                            ao_list_for_atom3_4)
print('local spin for another O2 = %.7f, 2S+1 = %.7f' % ss)

