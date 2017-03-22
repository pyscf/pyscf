#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''

import time
import numpy as np
from pyscf import lib
from pyscf.pbc.scf import uhf as pbcuhf
from pyscf.pbc.scf import kuhf
from pyscf.pbc.dft import krks
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build()
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    dm = np.asarray(dm)
    nao = dm.shape[-1]
    # ndim = 4 : dm.shape = (alpha_beta, nkpts, nao, nao)
    ground_state = (dm.ndim == 4 and kpts_band is None)
    nkpts = len(kpts)

    if hermi == 2:  # because rho = 0
        n, ks._exc, vx = 0, 0, 0
    else:
        n, ks._exc, vx = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm, 1,
                                           kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=(cell.spin>0)+1)
    if abs(hyb) < 1e-10:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vhf = lib.asarray([vj[0]+vj[1]] * 2)
    else:
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vhf = pbcuhf._makevhf(vj, vk*hyb)

        if ground_state:
            ks._exc -= (np.einsum('Kij,Kji', dm[0], vk[0]) +
                        np.einsum('Kij,Kji', dm[1], vk[1])).real * .5 * hyb * (1./nkpts)

    if ground_state:
        ks._ecoul = np.einsum('Kij,Kji', dm[0]+dm[1], vj[0]+vj[1]).real * .5 * (1./nkpts)

    if small_rho_cutoff > 1e-20 and ground_state:
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(cell, dm, ks.grids,
                                           small_rho_cutoff, kpts)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - np.count_nonzero(idx))
        ks.grids.coords  = np.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = np.asarray(ks.grids.weights[idx], order='C')
        ks._numint.non0tab = None
    return vhf + vx


class KUKS(kuhf.KUHF):
    '''RKS class adapted for PBCs with k-point sampling.
    '''
    def __init__(self, cell, kpts):
        kuhf.KUHF.__init__(self, cell, kpts)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.UniformGrids(cell)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
        # Note Do not refer to .with_df._numint because gs/coords may be different
        self._numint = numint._KNumInt(kpts)
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        kuhf.KUHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff

    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf=None):
        if h1e_kpts is None: h1e_kpts = self.get_hcore(self.cell, self.kpts)
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        nkpts = len(h1e_kpts)
        e1 = 1./nkpts *(np.einsum('kij,kji', h1e_kpts, dm_kpts[0]) +
                        np.einsum('kij,kji', h1e_kpts, dm_kpts[1])).real

        tot_e = e1 + self._ecoul + self._exc
        logger.debug(self, 'E1 = %s  Ecoul = %s  Exc = %s', e1, self._ecoul, self._exc)
        return tot_e, self._ecoul + self._exc


