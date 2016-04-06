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
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf.dft import rks


def get_veff_(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff_` fore more details'''
    if mol is None: mol = self.mol
    if dm is None:
        dm = ks.make_rdm1()
    elif isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5,dm*.5))
    nset = len(dm) // 2
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build_()
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    n, ks._exc, vx = ks._numint.nr_uks_(mol, ks.grids, ks.xc, dm, hermi=hermi)
    logger.debug(ks, 'nelec by numeric integration = %s', n)
    t0 = logger.timer(ks, 'vxc', *t0)

    libxc = ks._numint.libxc
    hyb = libxc.hybrid_coeff(ks.xc, spin=(mol.spin>0)+1)

    if abs(hyb) < 1e-10:
        if (ks._eri is not None or not ks.direct_scf or
            not hasattr(ks, '_dm_last') or
            not isinstance(vhf_last, numpy.ndarray)):
            vj = ks.get_j(mol, dm, hermi)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += ks._vj_last
            ks._dm_last = dm
            ks._vj_last = vj
        if nset == 1:
            vhf = vj[0] + vj[1]
        else:
            vhf = vj[:nset] + vj[nset:]
        vhf = numpy.array((vhf,vhf))
    else:
        if (ks._eri is not None or not ks.direct_scf or
            not hasattr(ks, '_dm_last') or
            not isinstance(vhf_last, numpy.ndarray)):
            vj, vk = ks.get_jk(mol, dm, hermi)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vj += ks._vj_last
            vk += ks._vk_last
            ks._dm_last = dm
            ks._vj_last, ks._vk_last = vj, vk
        vhf = pyscf.scf.uhf._makevhf(vj, vk*hyb, nset)

        if nset == 1:
            ks._exc -=(numpy.einsum('ij,ji', dm[0], vk[0]) +
                       numpy.einsum('ij,ji', dm[1], vk[1])) * .5 * hyb
    if nset == 1:
        ks._ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj[0]+vj[1]) * .5

    if small_rho_cutoff > 1e-20 and nset == 1:
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(mol, dm[0]+dm[1], ks.grids,
                                           small_rho_cutoff)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks._numint.non0tab = None
    return vhf + vx


def energy_elec(ks, dm, h1e=None, vhf=None):
    if h1e is None:
        h1e = ks.get_hcore()
    e1 = numpy.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])
    tot_e = e1 + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


class UKS(pyscf.scf.uhf.UHF):
    '''Unrestricted Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol):
        pyscf.scf.uhf.UHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.uhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff_
    energy_elec = energy_elec

    def define_xc_(self, description):
        self.xc = description
        return self


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
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

