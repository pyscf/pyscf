#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Restricted Kohn-Sham for periodic systems at a single k-point

See Also:
    pyscf.pbc.dft.krks.py : Non-relativistic Restricted Kohn-Sham for periodic
                            systems with k-point sampling
'''

import time
import numpy
import pyscf.dft
from pyscf import lib
from pyscf.pbc.scf import uhf as pbcuhf
from pyscf.lib import logger
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.dft import rks
from pyscf.dft.rks import _attach_xc


def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpt=None, kpt_band=None):
    '''Coulomb + XC functional for UKS.  See pyscf/pbc/dft/uks.py
    :func:`get_veff` fore more details.
    '''
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpt is None: kpt = ks.kpt
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:  # RHF DM
        dm = numpy.asarray((dm*.5,dm*.5))

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, dm, 0,
                                        kpt, kpt_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    # ndim = 3 : dm.shape = ([alpha,beta], nao, nao)
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=cell.spin)
    if abs(hyb) < 1e-10:
        vj = ks.get_j(cell, dm, hermi, kpt, kpt_band)
        vxc += vj[0] + vj[1]
    else:
        vj, vk = ks.get_jk(cell, dm, hermi, kpt, kpt_band)
        vxc += vj[0] + vj[1] - vk * hyb

        if ground_state:
            exc -=(numpy.einsum('ij,ji', dm[0], vk[0]) +
                   numpy.einsum('ij,ji', dm[1], vk[1])).real * hyb * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj[0]+vj[1]).real * .5
    else:
        ecoul = None

    vxc = _attach_xc(vxc, ecoul, exc, vj=None, vk=None)

    nelec = cell.nelec
    if (small_rho_cutoff > 1e-20 and ground_state and
        abs(n[0]-nelec[0]) < 0.01*n[0] and abs(n[1]-nelec[1]) < 0.01*n[1]):
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(cell, dm, ks.grids,
                                           small_rho_cutoff, kpt)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks.grids.non0tab = ks.grids.make_mask(cell, ks.grids.coords)
    return vxc


class UKS(pbcuhf.UHF):
    '''UKS class adapted for PBCs.

    This is a literal duplication of the molecular UKS class with some `mol`
    variables replaced by `cell`.
    '''
    def __init__(self, cell, kpt=numpy.zeros(3)):
        pbcuhf.UHF.__init__(self, cell, kpt)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.UniformGrids(cell)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        pbcuhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = pyscf.dft.uks.energy_elec

    density_fit = rks._patch_df_beckegrids(pbcuhf.UHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(pbcuhf.UHF.mix_density_fit)
