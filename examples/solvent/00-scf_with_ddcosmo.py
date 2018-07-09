#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example of using solvent model in the mean-field calculations.
'''

from pyscf import gto, scf
from pyscf.solvent import ddcosmo, ddpcm

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
           )
mf = scf.RHF(mol)

ddcosmo.ddcosmo_for_scf(mf).run()
ddpcm.ddpcm_for_scf(mf).run()
