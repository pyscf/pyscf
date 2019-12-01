#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM/MM charges + implicit solvent model
'''

import numpy
from pyscf import gto, scf, qmmm, solvent
from pyscf.data import radii

mol = gto.M(atom='''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
            ''',
            verbose = 4)

numpy.random.seed(1)
coords = numpy.random.random((5,3)) * 10
charges = (numpy.arange(5) + 1.) * .1

class QMMMMole(gto.Mole):
    def __init__(self, mol, coords, charges):
        mm_atoms = [('C', c) for c in coords]
        mm_mol = gto.M(atom=mm_atoms, basis={})
        self.qm_mol = mol
        self.mm_mol = mm_mol
        self.mm_charges = charges
        qmmm_mol = mol + mm_mol
        self.__dict__.update(qmmm_mol.__dict__)

    def atom_charge(self, atom_id):
        if atom_id < self.qm_mol.natm:
            return self.qm_mol.atom_charge(atom_id)
        else:
            return self.mm_charges[atom_id-self.qm_mol.natm]

    def atom_charges(self):
        return numpy.append(self.qm_mol.atom_charges(), self.mm_charges)

# Make a giant system include both QM and MM particles
qmmm_mol = QMMMMole(mol, coords, charges)

# The solvent model is based on the giant system
sol = solvent.DDCOSMO(qmmm_mol)

# According to Lipparini's suggestion in issue #446
sol.radii_table = radii.VDW

#
# The order to apply solvent model and QM/MM charges does not affect results
#
# ddCOSMO-QMMM-SCF
#
mf = scf.RHF(mol)
mf = qmmm.mm_charge(mf, coords, charges)
mf = solvent.ddCOSMO(mf, sol)
mf.run()

#
# QMMM-ddCOSMO-SCF
#
mf = scf.RHF(mol)
mf = solvent.ddCOSMO(mf, sol)
mf = qmmm.mm_charge(mf, coords, charges)
mf.run()
