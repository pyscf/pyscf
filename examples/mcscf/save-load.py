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
m.scf()

mc = mcscf.CASSCF(m, 6, (4,2))  # 6 active orbitals, 4 alpha, 2 beta electrons
# Change the default CASSCF save_mo_coeff function. Frequently save CASSCF
# orbitals.
def save_mo_coeff(envs):
    imacro = envs['imacro']
    imicro = envs['imicro']
    if imacro % 3 == 2:
        fname = 'mcscf-mo-%d-%d.npy' % (imacro+1, imicro+1)
        print('Save MO of step %d-%d in file %s' % (imacro+1, imicro+1, fname))
        numpy.save(fname, envs['mo_coeff'])
mc.callback = save_mo_coeff
mc.max_orb_stepsize = .01 # max. orbital-rotation angle
mc.max_cycle_micro = 1    # small value for frequently calling CI solver
mc.max_cycle_macro = 10
mc.conv_tol = 1e-4
e1 = mc.kernel()[0]       # take 1-step CASSCF algorithm by default

mc = mcscf.CASSCF(m, 6, (4,2))
mc.fcisolver = pyscf.fci.direct_spin1
mc.max_orb_stepsize = .05
mc.max_cycle_micro = 3
mc.conv_tol = 1e-8
mo = numpy.load('mcscf-mo-6-1.npy')
e1 = mc.kernel(mo)[0]

