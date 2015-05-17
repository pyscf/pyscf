#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Unrestricted Kohn-Sham
'''

import time
import numpy
from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import vxc
from pyscf.dft import gen_grid
from pyscf.dft import numint


def get_veff_(ks, mol, dm, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional'''
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5,dm*.5))
    nset = len(dm) // 2
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.setup_grids_()
        t0 = logger.timer(ks, 'seting up grids', *t0)

    x_code, c_code = vxc.parse_xc_name(ks.xc)
    n, ks._exc, vx = \
            ks._numint.nr_uks(mol, ks.grids, x_code, c_code,
                                dm, verbose=ks.verbose)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    if (ks._eri is not None or ks._is_mem_enough() or
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

    hyb = vxc.hybrid_coeff(x_code, spin=1)
    if abs(hyb) > 1e-10:
        if nset == 1:
            ks._exc -=(numpy.einsum('ij,ji', dm[0], vk[0])
                        +numpy.einsum('ij,ji', dm[1], vk[1])) * .5 * hyb
        vhf = pyscf.scf.uhf._makevhf(vj, vk*hyb, nset)
    else:
        if nset == 1:
            vhf = vj[0] + vj[1]
        else:
            vhf = vj[:nset] + vj[nset:]
        vhf = numpy.array((vhf,vhf))
    if nset == 1:
        ks._ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj[0]+vj[1]) * .5
    return vhf + vx


def energy_elec(ks, dm, h1e):
    e1 = numpy.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])
    tot_e = e1 + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


class UKS(pyscf.scf.uhf.UHF):
    ''' Restricted Kohn-Sham '''
    def __init__(self, mol):
        pyscf.scf.uhf.UHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.uhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functional'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        if h1e is None: h1e = self.get_hcore()
        return energy_elec(self, dm, h1e)



if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = 'out_uks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = UKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf())

