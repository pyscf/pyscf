#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham
'''

import time
import numpy
from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import vxc
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf.dft import uks


def get_veff_(ks, mol, dm, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional'''
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.setup_grids_()
        t0 = logger.timer(ks, 'seting up grids', *t0)

    x_code, c_code = vxc.parse_xc_name(ks.xc)
    #n, ks._exc, vx = vxc.nr_vxc(mol, ks.grids, x_code, c_code,
    #                              dm, spin=1, relativity=0)
    if ks._numint is None:
        n, ks._exc, vx = numint.nr_vxc(mol, ks.grids, x_code, c_code,
                                       dm, spin=mol.spin, relativity=0)
    else:
        n, ks._exc, vx = \
                ks._numint.nr_vxc(mol, ks.grids, x_code, c_code,
                                  dm, spin=mol.spin, relativity=0)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    hyb = vxc.hybrid_coeff(x_code, spin=1)

    if abs(hyb) < 1e-10:
        vj = ks.get_j(mol, dm, hermi)
    elif (ks._eri is not None or ks._is_mem_enough() or
        not ks.direct_scf):
        vj, vk = ks.get_jk(mol, dm, hermi)
    else:
        if isinstance(vhf_last, numpy.ndarray):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi=hermi)
            vj += ks._vj_last
            vk += ks._vk_last
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
        ks._vj_last, ks._vk_last = vj, vk

    if abs(hyb) > 1e-10:
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            ks._exc -= numpy.einsum('ij,ji', dm, vk) * .5 * hyb*.5
        vhf = vj - vk * (hyb * .5)
    else:
        vhf = vj

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        ks._ecoul = numpy.einsum('ij,ji', dm, vj) * .5
    return vhf + vx


def energy_elec(ks, dm, h1e):
    e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
    tot_e = e1 + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


class RKS(pyscf.scf.hf.RHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf.RHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = ks.get_hcore()
        return energy_elec(self, dm, h1e)


class ROKS(pyscf.scf.hf.ROHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.hf.ROHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.hf.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functional'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return uks.get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = ks.get_hcore()
        return uks.energy_elec(self, dm, h1e)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = 'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.build()

    m = RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
