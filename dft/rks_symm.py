#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham
'''

from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf.dft import rks
from pyscf.dft import uks


class RKS(pyscf.scf.hf_symm.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.RHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.hf_symm.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return rks.get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = self.get_hcore()
        return rks.energy_elec(self, dm, h1e)


class ROKS(pyscf.scf.hf_symm.ROHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.ROHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.hf_symm.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functional'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return uks.get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = self.get_hcore()
        return uks.energy_elec(self, dm, h1e)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = 'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.symmetry = 1
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf())
