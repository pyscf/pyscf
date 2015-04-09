#!/usr/bin/env python
from pyscf import gto
from pyscf import scf
from pyscf import tools
import pyscf.tools.c60struct

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_c60'
mol.atom = [('C', c) for c in tools.c60struct.make60(1.46,1.38)]

mol.basis = {'C': 'ccpvtz',}
mol.build()

mf = scf.density_fit(scf.RHF(mol))
mf.verbose = 5
mf.chkfile = 'c60tz.chkfile'
mf.level_shift_factor=.5
mf.conv_tol = 1e-7
mf.scf()

mf = scf.RHF(mol)
mf.chkfile = 'c60tz.chkfile'
mf.init_guess = 'chkfile'
mf.conv_tol = 1e-8
print(mf.scf() - -2272.4201163243)
