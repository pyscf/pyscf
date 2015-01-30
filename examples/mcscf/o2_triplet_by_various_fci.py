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
mol.spin = 2
mol.build()

m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

# Default running of MCSCF
mc = mcscf.CASSCF(m, 4, (4,2))
mc.stdout.write('** Triplet, using spin1 ci solver **\n')
emc1 = mc.mc1step()[0]

mc.analyze()
print('CASSCF = %.15g' % emc1)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))

mc.stdout.write('** Triplet,  using MS0 ci solver **\n')

mol.build(False, False)
m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

mc = mcscf.CASSCF(m, 4, 6)
# change the CAS space FCI solver. e.g. to DMRG, FCIQMC
mc.fcisolver = fci.direct_spin1
# Initial guess of MCSCF with given CI coefficients
na = fci.cistring.num_strings(4, 3)
ci0 = numpy.zeros((na,na))
addr = fci.cistring.str2addr(4, 3, int('1011',2))
ci0[addr,0] = numpy.sqrt(.5)
ci0[0,addr] =-numpy.sqrt(.5)
ci0 = None
emc1 = mc.mc1step(ci0=ci0)[0]
mc.analyze()
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
print('CASSCF = %.15g' % emc1)


mc.stdout.write('** Symmetry-broken singlet, using spin0 ci solver **\n')

mol.spin = 0
mol.build(False, False)
m = scf.RHF(mol)
print('HF     = %.15g' % m.scf())

mc = mcscf.CASSCF(m, 6, 6)
mc.fcisolver = fci.direct_spin0
# Change CAS active space
# MO index for CAS space to generate initial guess
caspace = [6,7,8,9,10,12]
mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caspace, 1)
emc1 = mc.mc1step(mo)[0]

mc.analyze()
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
print('CASSCF = %.15g' % emc1)

