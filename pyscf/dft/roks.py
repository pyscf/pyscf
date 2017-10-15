#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic restricted open-shell Kohn-Sham
'''

from pyscf.lib import logger
from pyscf.scf import rohf
from pyscf.dft.uks import get_veff, energy_elec
from pyscf.dft import rks


class ROKS(rohf.ROHF):
    '''Restricted open-shell Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol):
        rohf.ROHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        rohf.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = energy_elec
    define_xc_ = rks.define_xc_


if __name__ == '__main__':
    from pyscf import gto
    from pyscf.dft import xcfun
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = ROKS(mol)
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

    m = ROKS(mol)
    m._numint.libxc = xcfun
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

