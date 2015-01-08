import numpy
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo
from pyscf import fci
from pyscf import mcscf


mol = gto.Mole()
mol.verbose = 0
mol.output = None
mol.atom = [
    ['O', (0., 0.,  0.7)],
    ['O', (0., 0., -0.7)],]
mol.basis = '6-31g'
mol.build()
mf = scf.RHF(mol)
mf.scf()

ncas = 4
nelec = (4,2)
mc = mcscf.casci.CASCI(mol, mf, ncas, nelec)
mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [5,7,8,9], 1)
mocas = mo[:,5:9]
e, ecas, ci0 = mc.casci(mo=mo)
print('RHF-CASCI total energy of O2', e+mol.get_enuc())
ss = fci.spin_square(ci0, ncas, nelec, mocas, mf.get_ovlp())
print('S^2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O', ss)


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
mf.scf()

ncas = 4
nelec = (4,2)
mc = mcscf.CASCI(mol, mf, ncas, nelec)
mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [[5,6,8,9],[6,7,8,9]], 1)
mocas = (mo[0][:,5:9], mo[1][:,5:9])
e, ecas, ci0 = mc.casci(mo=mo)
print('UHF-CASCI total energy of O2', e+mol.get_enuc())
ss = fci.spin_square(ci0, ncas, nelec, mocas, mf.get_ovlp())
print('S^2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O', ss)


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
mf.scf()

ncas = 6
nelec = (5,3)
mc = mcscf.casci.CASCI(mol, mf, ncas, nelec)
mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [5,7,8,9,10,11], 1)
mocas = mo[:,5:11]
e, ecas, ci0 = mc.casci(mo=mo)
print('RHF-CASCI total energy of O2+H2', e+mol.get_enuc())

ss = fci.spin_square(ci0, ncas, nelec, mocas, mf.get_ovlp())
print('S^2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,22))
print('local spin for H2', ss)


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
mf.scf()

ncas = 8
nelec = (6,6)
mc = mcscf.casci.CASCI(mol, mf, ncas, nelec)
mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [9,10,13,14,15,16,17,18], 1)
mocas = mo[:,10:18]
e, ecas, ci0 = mc.casci(mo=mo)
print('RHF-CASCI total energy of O2+O2 singlet, which is incorrect', e+mol.get_enuc())
ss = fci.spin_square(ci0, ncas, nelec, mocas, mf.get_ovlp())
print('S^2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,36))
print('local spin for another O2', ss)


ncas = 8
nelec = (8,4)
mc = mcscf.casci.CASCI(mol, mf, ncas, nelec)
mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, [9,10,13,14,15,16,17,18], 1)
mocas = mo[:,10:18]
e, ecas, ci0 = mc.casci(mo=mo)
print('RHF-CASCI total energy of O2+O2 quintet, = monomer*2, correct', e+mol.get_enuc())
ss = fci.spin_square(ci0, ncas, nelec, mocas, mf.get_ovlp())
print('S^2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(9))
print('local spin for O', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18))
print('local spin for O2', ss)
ss = fci.spin_op.local_spin(ci0, ncas, nelec, mocas, mf.get_ovlp(), range(18,36))
print('local spin for another O2', ss)
