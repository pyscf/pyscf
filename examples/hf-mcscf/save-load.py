#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf import mcscf
import pyscf.mcscf.mc1step

mol = gto.Mole()
mol.verbose = 5
b = 1.6
mol.output = 'out_hf-%2.1f' % b
mol.atom = [
    ["F", (0., 0., 0.)],
    ["H", (0., 0., b)],]

mol.basis = {'F': 'cc-pvdz',
             'H': 'cc-pvdz',}
mol.build()
m = scf.RHF(mol)
m.set_mo_occ = scf.addons.dynamic_occ(m, 1e-3)
m.scf()

mc = mcscf.CASSCF(mol, m, 6, (4,2))
mc.fcisolver = pyscf.fci.direct_spin1
def save_mo_coeff(mo_coeff, imacro, imicro):
    if imacro % 3 == 2:
        print imacro, imicro
        fname = 'mcscf-mo-%d-%d.npy' % (imacro+1, imicro+1)
        numpy.save(fname, mo_coeff)
mc.save_mo_coeff = save_mo_coeff
mc.max_orb_stepsize = .01 # max. orbital-rotation angle
mc.max_cycle_micro = 1    # small value for frequently call CI solver
mc.max_cycle_macro = 10
mc.conv_threshold = 1e-4
e1 = mc.mc1step()[0]

mc = mcscf.CASSCF(mol, m, 6, (4,2))
mc.fcisolver = pyscf.fci.direct_spin1
mc.max_orb_stepsize = .05
mc.max_cycle_micro = 3
mc.conv_threshold = 1e-8
mo = numpy.load('mcscf-mo-6-1.npy')
e1 = mc.mc1step(mo)[0]

