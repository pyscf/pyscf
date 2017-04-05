#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham
'''

from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import rks
from pyscf.dft import uks


class RKS(pyscf.scf.hf_symm.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.RHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.hf_symm.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = rks.get_veff
    energy_elec = rks.energy_elec


class ROKS(pyscf.scf.hf_symm.ROHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf_symm.ROHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.hf_symm.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags()

    get_veff = uks.get_veff
    energy_elec = uks.energy_elec


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 2
    mol.output = '/dev/null'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.symmetry = 1
    mol.build()

    m = RKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

