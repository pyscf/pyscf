#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto, scf, mcscf

'''
UHF-CASSCF allows different alpha and beta orbitals.
'''

mol = gto.Mole()
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()

mf = scf.UHF(mol)
print('E(UHF) = %.15g' % mf.kernel())

mc = mcscf.UCASSCF(mf, 4, (4,2))
emc1 = mc.kernel()[0]
print('* Triplet with UHF-CASSCF, E(UCAS) = %.15g' % emc1)
# Generally, 2s+1 is not 3
print('  S^2 = %.6f, 2S+1 = %.6f' % mcscf.spin_square(mc))


print('\n')
mol.spin = 0
mol.build()
mf = scf.UHF(mol)
print('E(UHF) = %.15g' % mf.kernel())

mc = mcscf.UCASSCF(mf, 4, 6)
emc1 = mc.kernel()[0]
print('* Singlet with UHF-CASSCF, E(UCAS) = %.15g' % emc1)
# Generally, 2s+1 is not 1
print('  S^2 = %.6f, 2S+1 = %.6f' % mcscf.spin_square(mc))

# A small system illustrates independent alpha/beta active-orbital selection.
# sort_mo uses 1-based indices; each list contains ncas orbital indices.
small_mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g')
small_mf = scf.UHF(small_mol).run()
ucas = mcscf.UCASCI(small_mf, 3, (1, 1))
mo = ucas.sort_mo(([2, 3, 4], [2, 3, 5]))

# UCASCI retains these orbitals and can compute several CI roots.
ucas.fcisolver.nroots = 2
ucas.kernel(mo)
ucas.analyze()

# UCASSCF instead optimizes the orbitals. Both one-step (the default kernel)
# and two-step optimization accept the same pair of initial orbital arrays.
one_step = mcscf.UCASSCF(small_mf, 3, (1, 1))
one_step.kernel(mo)
two_step = mcscf.UCASSCF(small_mf, 3, (1, 1))
two_step.mc2step(one_step.mo_coeff)
