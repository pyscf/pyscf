#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import mcscf
from pyscf import fci
from pyscf.tools import dump_mat


mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_o2_uhf'
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]

mol.basis = {'O': 'cc-pvdz',
             'C': 'cc-pvdz',}
mol.spin = 2
mol.build()

m = scf.UHF(mol)
print('UHF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 4, (4,2))
mc.stdout.write('** Triplet with UHF-CASSCF**\n')
emc1 = mc.mc1step()[0]
print('CASSCF = %.15g' % (emc1 + mol.get_enuc()))
ncore = mc.ncore
nocc = (ncore[0] + mc.ncas, ncore[1] + mc.ncas)
ovlp = reduce(numpy.dot, (mc.mo_coeff[0][:,ncore[0]:nocc[0]].T, m.get_ovlp(),
                          mc.mo_coeff[1][:,ncore[1]:nocc[1]]))
ss,s2 = fci.spin_square_with_overlap(ovlp, mc.ci, 4, (4,2))
print('s^2 = %.6f, 2s+1 = %.6f' % (ss,s2))


mol.spin = 0
mol.build()
m = scf.UHF(mol)
print('UHF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 4, 6)
mc.stdout.write('** Singlet with UHF-CASSCF **\n')
emc1 = mc.mc1step()[0]

print('CASSCF = %.15g' % (emc1 + mol.get_enuc()))
ncore = mc.ncore
nocc = (ncore[0] + mc.ncas, ncore[1] + mc.ncas)
ovlp = reduce(numpy.dot, (mc.mo_coeff[0][:,ncore[0]:nocc[0]].T, m.get_ovlp(),
                          mc.mo_coeff[1][:,ncore[1]:nocc[1]]))
ss,s2 = fci.spin_square_with_overlap(ovlp, mc.ci, 4, (4,2))
print('s^2 = %.6f, 2s+1 = %.6f' % (ss,s2))
