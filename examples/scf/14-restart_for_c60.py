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
mf.chkfile = 'c60tz.chkfile'
mf.level_shift = .5
mf.conv_tol = 1e-7
mf.scf()

mf = scf.RHF(mol)
mf.chkfile = 'c60tz.chkfile'
mf.init_guess = 'chkfile'
mf.conv_tol = 1e-8
print(mf.scf() - -2272.4201163243)
