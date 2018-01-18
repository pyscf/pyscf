#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Unrestricted Kohn-Sham
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import uhf
from pyscf.dft import rks


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = self.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    t0 = (time.clock(), time.time())

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        if ks.small_rho_cutoff > 1e-20 and ground_state:
            ks.grids = rks.prune_small_rho_grids_(ks, mol, dm[0]+dm[1], ks.grids)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(mol, ks.grids, ks.xc, dm)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=mol.spin)
    if abs(hyb) < 1e-10:
        vk = None
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vj', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += vhf_last.vj
        else:
            vj = ks.get_j(mol, dm, hermi)
        vxc += vj[0] + vj[1]
    else:
        if (ks._eri is None and ks.direct_scf and
            getattr(vhf_last, 'vk', None) is not None):
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vj += vhf_last.vj
            vk += vhf_last.vk
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
        vxc += vj[0] + vj[1] - vk * hyb

        if ground_state:
            exc -=(numpy.einsum('ij,ji', dm[0], vk[0]) +
                   numpy.einsum('ij,ji', dm[1], vk[1])) * hyb * .5
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj[0]+vj[1]) * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    e1 = numpy.einsum('ij,ji', h1e, dm[0]) + numpy.einsum('ij,ji', h1e, dm[1])
    tot_e = e1.real + vhf.ecoul + vhf.exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
    return tot_e, vhf.ecoul+vhf.exc


class UKS(uhf.UHF):
    '''Unrestricted Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol):
        uhf.UHF.__init__(self, mol)
        rks._dft_common_init_(self)

    def dump_flags(self):
        uhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = energy_elec
    define_xc_ = rks.define_xc_


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    #mol.grids = { 'He': (10, 14),}
    mol.build()

    m = UKS(mol)
    m.xc = 'b3lyp'
    print(m.scf())  # -2.89992555753

