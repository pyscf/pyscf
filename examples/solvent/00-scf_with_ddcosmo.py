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
mf.run()

# Instead of assigning the dielectric constant, the solvent can be specified by
# its name. This sets .eps as well as .eps_optical, the optical dielectric
# constant required by the non-equilibrium solvation of excited states.
# Available solvents are the keys of pyscf.solvent.smd.solvent_db
mf.with_solvent.solvent = 'methanol'
print(mf.with_solvent.eps, mf.with_solvent.eps_optical)
mf.with_solvent.solvent = 'DMSO'   # names are matched case-insensitively
print(mf.with_solvent.eps, mf.with_solvent.eps_optical)

# The solvent name can also be given to the .ddCOSMO() and .ddPCM() methods of
# a mean-field object
mf = scf.RHF(mol).ddCOSMO('methanol')
mf.run()
mf = scf.RHF(mol).ddPCM('methanol')
mf.run()
