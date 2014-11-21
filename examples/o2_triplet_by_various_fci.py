#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import mcscf
from pyscf import fci
from pyscf.tools import dump_mat


mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_o2'
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]

mol.basis = {'O': 'cc-pvdz',
             'C': 'cc-pvdz',}
mol.charge = 2
mol.build()

m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

mol.nelectron += 2
mc = mcscf.CASSCF(mol, m, 4, (4,2))
mc.fcisolver = fci.direct_spin1
label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]

mc.stdout.write('** Triplet, using spin1 ci solver **\n')
emc1 = mc.mc1step()[0]
print('CASSCF = %.15g' % (emc1 + mol.get_enuc()))
dm1a, dm1b = mcscf.addons.make_rdm1s(mc, mc.ci, mc.mo_coeff)
mc.stdout.write('spin alpha\n')
dump_mat.dump_tri(m.stdout, dm1a, label)
mc.stdout.write('spin beta\n')
dump_mat.dump_tri(m.stdout, dm1b, label)

s = reduce(numpy.dot, (mc.mo_coeff.T, m.get_ovlp(), m.mo_coeff))
idx = numpy.argwhere(abs(s)>.5)
for i,j in idx:
    mol.stdout.write('<mo-mcscf|mo-hf> %d, %d, %12.8f\n' % (i,j,s[i,j]))
mc.stdout.write('** Largest CI components **\n')
mol.stdout.write('%s\n' % str(fci.addons.large_ci(mc.ci, 4, (4,2))))


mc.stdout.write('** Triplet,  using MS0 ci solver **\n')

mol.charge = 0
mol.build(False, False)
m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 4, 6)
mc.fcisolver = fci.direct_ms0
mo = mcscf.addons.sort_mo(mc, m.mo_coeff, [5,7,8,9], 1)
na = fci.cistring.num_strings(4, 3)
ci0 = numpy.zeros((na,na))
addr = fci.cistring.str2addr(4, 3, int('1011',2))
ci0[addr,0] = numpy.sqrt(.5)
ci0[0,addr] =-numpy.sqrt(.5)
emc1 = mc.mc1step(mo, ci0)[0]
print('CASSCF = %.15g' % (emc1 + mol.get_enuc()))
dm1a, dm1b = mcscf.addons.make_rdm1s(mc, mc.ci, mc.mo_coeff)
mc.stdout.write('spin alpha\n')
dump_mat.dump_tri(m.stdout, dm1a, label)
mc.stdout.write('spin beta\n')
dump_mat.dump_tri(m.stdout, dm1b, label)

s = reduce(numpy.dot, (mc.mo_coeff.T, m.get_ovlp(), m.mo_coeff))
idx = numpy.argwhere(abs(s)>.5)
for i,j in idx:
    mol.stdout.write('<mo-mcscf|mo-hf> %d, %d, %12.8f\n' % (i,j,s[i,j]))
mc.stdout.write('** Largest CI components **\n')
mol.stdout.write('%s\n' % str(fci.addons.large_ci(mc.ci, 4, 6)))


mc.stdout.write('** Symmetry-broken singlet, using spin0 ci solver **\n')

mol.charge = 0
mol.build(False, False)
m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

mc = mcscf.CASSCF(mol, m, 6, 6)
mc.fcisolver = fci.direct_spin0
emc1 = mc.mc1step()[0]
print('CASSCF = %.15g' % (emc1 + mol.get_enuc()))
dm1a, dm1b = mcscf.addons.make_rdm1s(mc, mc.ci, mc.mo_coeff)
mc.stdout.write('spin alpha\n')
dump_mat.dump_tri(m.stdout, dm1a, label)
mc.stdout.write('spin beta\n')
dump_mat.dump_tri(m.stdout, dm1b, label)

s = reduce(numpy.dot, (mc.mo_coeff.T, m.get_ovlp(), m.mo_coeff))
idx = numpy.argwhere(abs(s)>.5)
for i,j in idx:
    mol.stdout.write('<mo-mcscf|mo-hf> %d, %d, %12.8f\n' % (i,j,s[i,j]))
mc.stdout.write('** Largest CI components **\n')
mol.stdout.write('%s\n' % str(fci.addons.large_ci(mc.ci, 6, 6)))

