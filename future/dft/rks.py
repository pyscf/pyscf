#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Kohn-Sham
'''

import time
import numpy
import pyscf.lib
import pyscf.lib.logger as log
import pyscf.scf
from pyscf.scf import _vhf
from pyscf.dft import vxc
from pyscf.dft import gen_grid


class RKS(pyscf.scf.hf.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf.RHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._keys = set(self.__dict__.keys()).union(['_keys'])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        log.info(self, 'XC functionals = %s', self.xc)
        log.info(self, 'DFT grids: %s', self.grids.becke_scheme.__doc__)
        #TODO:for k,v in self.mol.grids.items():
        #TODO:    log.info(self, '%s   radi %d, angular %d', k, *v)

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functional'''
        t0 = (time.clock(), time.time())
        if self.grids.coords is None:
            self.grids.setup_grids()
            t0 = log.timer(self, 'seting up grids', *t0)

        x_code, c_code = vxc.parse_xc_name(self.xc)
        n, self._exc, vx = vxc.nr_vxc(mol, self.grids, x_code, c_code, \
                                      dm, spin=1, relativity=0)
        log.debug(self, 'nelec by numeric integration = %s', n)
        t0 = log.timer(self, 'vxc', *t0)

        vj, vk = self.get_jk(mol, dm, hermi)
        self._ecoul = numpy.einsum('ij,ji', dm, vj) * .5

        hyb = vxc.hybrid_coeff(x_code, spin=1)
        if abs(hyb) > 1e-10:
            vk = vk * hyb * .5
            self._exc -= numpy.einsum('ij,ji', dm, vk) * .5
            vx -= vk
        return vj + vx

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None:
            h1e = mf.get_hcore()
        e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
        tot_e = e1 + self._ecoul + self._exc
        log.debug(self, 'Ecoul = %s  Exc = %s', self._ecoul, self._exc)
        return tot_e, self._ecoul+self._exc



if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = 'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = {
        'He': 'cc-pvdz'}
    mol.grids = { 'He': (10, 14),}
    mol.build()

    m = RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    #m.init_guess = '1e'
    print(m.scf()) #-2.8519879
