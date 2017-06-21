#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
4-component Dirac-Kohn-Sham
'''

import time
import numpy
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.dft import rks
from pyscf.dft import gen_grid
from pyscf.dft import r_numint


def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.  The ._exc and ._ecoul attributes
            will be updated after return.  Attributes ._dm_last, ._vj_last and
            ._vk_last might be changed if direct SCF method is applied.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.  If vhf_last is not given,
            the function will not call direct_scf and attacalites ._dm_last,
            ._vj_last and ._vk_last will not be updated.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    dm = numpy.asarray(dm)
    nao = dm.shape[-1]
    ground_state = (dm.ndim == 2)

    if hermi == 2:  # because rho = 0
        n, ks._exc, vx = 0, 0, 0
    else:
        n, ks._exc, vx = ks._numint.nr_vxc(mol, ks.grids, ks.xc, dm, hermi=hermi)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=mol.spin)
    if abs(hyb) < 1e-10:
        if (ks._eri is not None or not ks.direct_scf or
            not hasattr(ks, '_dm_last') or
            not isinstance(vhf_last, numpy.ndarray)):
            vhf = vj = ks.get_j(mol, dm, hermi)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj = ks.get_j(mol, ddm, hermi)
            vj += ks._vj_last
            ks._dm_last = dm
            vhf = ks._vj_last = vj
    else:
        raise NotImplementedError
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
        vhf = vj - vk * hyb

        if ground_state:
            ks._exc -= numpy.einsum('ij,ji', dm, vk) * .5 * hyb

    if ground_state:
        ks._ecoul = numpy.einsum('ij,ji', dm, vj) * .5

    if (small_rho_cutoff > 1e-20 and ground_state and
        abs(n-mol.nelectron) < 0.01*n):
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(mol, dm, ks.grids, small_rho_cutoff)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks.grids.non0tab = ks.grids.make_mask(mol, ks.grids.coords)
    return vhf + vx


def energy_elec(ks, dm, h1e=None, vhf=None):
    return rks.energy_elec(ks, dm, h1e, vhf)


class UKS(dhf.UHF):
    def __init__(self, mol):
        dhf.UHF.__init__(self, mol)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(self.mol)
        self.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
        self._numint = r_numint._RNumInt()
        self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])

    def dump_flags(self):
        dhf.UHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = energy_elec

DKS = UKS


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

    m = DKS(mol)
    print(m.scf())

