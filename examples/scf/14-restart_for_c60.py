#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import scf
from pyscf.tools import c60struct

'''
Initial guess from chkfile.

Caculation can be restarted by providing attribute .chkfile as initial guess.

The intermediate results are saved in the file specified by .chkfile attribute.
If no filename is assigned to .chkfile, chkfile will be initialized to a
temproray file in the temporory directory (controlled by TMPDIR enviroment)
and the temporary file will be deleted automatically if calculation is
successfully finished.  If the calculation is failed, we can search in the
output for such message  "chkfile to save SCF result = XXXXX"  for the chkfile
of current calculation.  This chkfile can be used as the restart point.
'''

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_c60'
mol.atom = [('C', c) for c in c60struct.make60(1.46,1.38)]

mol.basis = {'C': 'ccpvtz',}
mol.build()

mf = scf.density_fit(scf.RHF(mol))
mf.verbose = 5
# Save resutls in chkfile
mf.chkfile = 'c60tz.chkfile'
mf.level_shift = .5
mf.conv_tol = 1e-7
mf.scf()

# There are many methods to restart calculation from chkfile.
# 1. to set .chkfile and .init_guess attributes.  The program will read
# .chkfile and generate initial guess
mf = scf.RHF(mol)
mf.chkfile = 'c60tz.chkfile'
mf.init_guess = 'chkfile'
mf.conv_tol = 1e-8
print(mf.scf() - -2272.4201163243)

# 2. function from_chk can read results from a given chkfile and project the
# density matrix to current system as initial guess
mf = scf.RHF(mol)
dm = mf.from_chk('c60tz.chkfile')
mf.scf(dm)

# 3. If the system saved by chkfile is the same to the current system, one can
# directly access the chkfile to get orbital coefficients and compute the
# density matrix as initial guess
mf = scf.RHF(mol)
mo_coeff = scf.chkfile.load('c60tz.chkfile', 'scf/mo_coeff')
mo_occ = scf.chkfile.load('c60tz.chkfile', 'scf/mo_occ')
dm = mf.make_rdm1(mo_coeff, mo_occ)
mf.scf(dm)

# 4. The last treatments can be simplified.  By calling .__dict__.update
# function to refresh SCF objects, mf.mo_coeff and mf.mo_occ etc get default
# values.  make_rdm1 function can take the default value to generate a density
# matrix.
mf = scf.RHF(mol)
mf.__dict__.update(scf.chkfile.load('c60tz.chkfile', 'scf'))
dm = mf.make_rdm1()
mf.scf(dm)

