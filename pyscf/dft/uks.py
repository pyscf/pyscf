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
from pyscf.scf import uhf
from pyscf.dft import rks


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = self.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        # Filter grids only for the first time setting up grids
        small_rho_cutoff = 0

    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))

    if hermi == 2:  # because rho = 0
        n, ks._exc, vx = (0,0), 0, 0
    else:
        n, ks._exc, vx = ks._numint.nr_uks(mol, ks.grids, ks.xc, dm)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=mol.spin)
    if abs(hyb) < 1e-10:
        if (ks._eri is not None or not ks.direct_scf or
            ks._dm_last is None or
            not isinstance(vhf_last, numpy.ndarray)):
            vj = ks.get_j(mol, dm, hermi)
        else:
            ddm = dm - numpy.asarray(ks._dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += ks._vj_last
            ks._dm_last = dm
            ks._vj_last = vj
        vhf = vj[0] + vj[1]
        vhf = numpy.asarray((vhf,vhf))
    else:
        if (ks._eri is not None or not ks.direct_scf or
            ks._dm_last is None or
            not isinstance(vhf_last, numpy.ndarray)):
            vj, vk = ks.get_jk(mol, dm, hermi)
        else:
            ddm = dm - numpy.asarray(ks._dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi)
            vj += ks._vj_last
            vk += ks._vk_last
            ks._dm_last = dm
            ks._vj_last, ks._vk_last = vj, vk
        vhf = vj[0] + vj[1] - vk * hyb

        if ground_state:
            ks._exc -=(numpy.einsum('ij,ji', dm[0], vk[0]) +
                       numpy.einsum('ij,ji', dm[1], vk[1])) * .5 * hyb
    if ground_state:
        ks._ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj[0]+vj[1]) * .5

    nelec = mol.nelec
    if (small_rho_cutoff > 1e-20 and ground_state and
        abs(n[0]-nelec[0]) < 0.01*n[0] and abs(n[1]-nelec[1]) < 0.01*n[1]):
        idx = ks._numint.large_rho_indices(mol, dm[0]+dm[1], ks.grids,
                                           small_rho_cutoff)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks.grids.non0tab = ks.grids.make_mask(mol, ks.grids.coords)
    return vhf + vx


def energy_elec(ks, dm, h1e=None, vhf=None):
    if h1e is None:
        h1e = ks.get_hcore()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = numpy.array((dm*.5, dm*.5))
    e1 = numpy.einsum('ij,ji', h1e, dm[0]) + numpy.einsum('ij,ji', h1e, dm[1])
    tot_e = e1.real + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


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

    def define_xc_(self, description):
        raise RuntimeError('define_xc_ method is depercated.  '
                           'Set mf.xc = %s instead.' % description)


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

