#!/usr/bin/env python

'''
State average over custom FCI solvers

This example shows how to put custom FCI solvers into one state-average solver
using the method state_average_mix_.
'''

import numpy as np
import pyscf

mol = pyscf.M(
    atom = [
        ['C', ( 0., 0.    , -.485)],
        ['C', ( 0., 0.    ,  .485)],],
    basis = 'cc-pvdz',
    symmetry = True)
mf = mol.RHF()
mf.irrep_nelec = {'A1g': 4, 'E1gx': 0, 'E1gy': 0, 'A1u': 4,
                  'E1uy': 2, 'E1ux': 2, 'E2gx': 0, 'E2gy': 0, 'E2uy': 0, 'E2ux': 0}
ehf = mf.kernel()
#mf.analyze()

class FakeFCI(pyscf.fci.direct_spin1_symm.FCI):
    def __init__(self, mol, nelec=None):
        self.nelec = nelec
        self.fake_potential = None
        super().__init__(mol)

    def kernel(self, h1, h2, norb, nelec, *args, **kwargs):
        if self.nelec is not None:
            nelec = self.nelec
        if self.fake_potential is not None:
            h1 = h1 + np.identity(norb) * self.fake_potential
        return super().kernel(h1, h2, norb, nelec, *args, **kwargs)

    def make_rdm1(self, civec, norb, nelec):
        if self.nelec is not None:
            nelec = self.nelec
        return super().make_rdm1(civec, norb, nelec)

    def make_rdm12(self, civec, norb, nelec):
        if self.nelec is not None:
            nelec = self.nelec
        return super().make_rdm12(civec, norb, nelec)

    def spin_square(self, civec, norb, nelec):
        if self.nelec is not None:
            nelec = self.nelec
        return super().spin_square(civec, norb, nelec)

solver1 = FakeFCI(mol)
solver1.nelec = (4, 3)
solver1.wfnsym= 'A1u'
solver1.nroots = 1
solver1.fake_potential = 0.01

solver2 = FakeFCI(mol)
solver2.nelec = (3, 3)
solver2.wfnsym= 'A1g'
solver2.nroots = 2

mc = pyscf.mcscf.CASSCF(mf, 8, 8)
mc = pyscf.mcscf.addons.state_average_mix_(mc, [solver1, solver2], weights=(0.5, 0.25, 0.25))
mc.verbose = 4
mc.run()

