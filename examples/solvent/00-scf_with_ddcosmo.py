#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example of using solvent model in the mean-field calculations.
'''

from pyscf import gto, scf, dft
from pyscf import solvent

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)
mf = scf.RHF(mol)
solvent.ddCOSMO(mf).run()

mf = dft.UKS(mol)
mf.xc = 'b3lyp'
solvent.ddPCM(mf).run()

# Once solvent module is imported, PySCF-1.6.1 and newer supports the .ddCOSMO
# and .ddPCM methods to create solvent model.
from pyscf import solvent
mf = mf.ddCOSMO()
# Adjust solvent model by modifying the attribute .with_solvent
mf.with_solvent.eps = 32.613  # methanol dielectric constant
# Set eps = -1 for the conductor limit of the dielectric model.
# mf.with_solvent.eps = -1
mf.run()

# The solvent object also accepts a density matrix directly and returns the
# solvation energy together with its AO potential matrix. This is useful when
# incorporating the solvent response in a custom SCF procedure.
dm = scf.RHF(mol).run().make_rdm1()
ddpcm = solvent.ddPCM(mol)
e_solvent, v_solvent = ddpcm.kernel(dm)
print('ddPCM energy and potential shape:', e_solvent, v_solvent.shape)
