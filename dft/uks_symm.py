#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Unrestricted Kohn-Sham
'''

from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf.dft import uks


class UKS(pyscf.scf.uhf_symm.UHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.uhf_symm.UHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.uhf_symm.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = uks.get_veff_
    energy_elec = uks.energy_elec


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.symmetry = 1
    mol.build()

    m = UKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

