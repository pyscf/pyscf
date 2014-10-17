#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Kohn-Sham
'''

import time
import numpy
from pyscf import lib
from pyscf import scf
import pyscf.lib.logger as log
import vxc
import gen_grid
import pyscf.scf._vhf


class RKS(scf.hf.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        scf.hf.RHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)

    def dump_flags(self):
        scf.hf.RHF.dump_flags(self)
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

        if self._is_mem_enough():
            if self._eri is None:
                self._eri = scf._vhf.int2e_sph(mol._atm, mol._bas, mol._env)
            vj, vk = scf.hf.dot_eri_dm(self._eri, dm, hermi=hermi)
        else:
            if self.direct_scf:
                vj, vk = scf._vhf.direct(dm-dm_last, mol._atm, \
                                         mol._bas, mol._env, self.opt, \
                                         hermi=hermi)
            else:
                vj, vk = scf._vhf.direct(dm, mol._atm, mol._bas, mol._env, \
                                         hermi=hermi)
        log.timer(self, 'vj and vk', *t0)
        self._ecoul = lib.trace_ab(dm, vj) * .5

        hyb = vxc.hybrid_coeff(x_code, spin=1)
        if abs(hyb) > 1e-10:
            vk = vk * hyb * .5
            self._exc -= lib.trace_ab(dm, vk) * .5
            vx -= vk
        return vj + vx

    def calc_tot_elec_energy(self, veff, dm, mo_energy, mo_occ):
        sum_mo_energy = numpy.dot(mo_energy, mo_occ)
        coul_dup = lib.trace_ab(dm, veff)
        tot_e = sum_mo_energy - coul_dup + self._ecoul + self._exc
        log.debug(self, 'Ecoul = %s  Exc = %s', self._ecoul, self._exc)
        return tot_e, self._ecoul, self._exc



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

##############
# SCF result
    m = RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    #m.init_guess('1e')
    print(m.scf()) #-2.8519879
