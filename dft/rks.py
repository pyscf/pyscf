#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic restricted Kohn-Sham
'''

import time
import numpy
from pyscf.lib import logger
import pyscf.scf
from pyscf.dft import gen_grid
from pyscf.dft import numint


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
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        nset = 1
    else:
        nset = len(dm)

    t0 = (time.clock(), time.time())
    if ks.grids.coords is None:
        ks.grids.build()
        small_rho_cutoff = ks.small_rho_cutoff
        t0 = logger.timer(ks, 'setting up grids', *t0)
    else:
        small_rho_cutoff = 0

    hyb = ks._numint.hybrid_coeff(ks.xc, spin=(mol.spin>0)+1)

    if hermi == 2:  # because rho = 0
        n, ks._exc, vx = 0, 0, 0
    else:
        n, ks._exc, vx = ks._numint.nr_rks(mol, ks.grids, ks.xc, dm, hermi=hermi)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

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
        vhf = vj - vk * (hyb * .5)

        if nset == 1:
            ks._exc -= numpy.einsum('ij,ji', dm, vk) * .5 * hyb*.5

    if nset == 1:
        ks._ecoul = numpy.einsum('ij,ji', dm, vj) * .5

    if small_rho_cutoff > 1e-20 and nset == 1:
        # Filter grids the first time setup grids
        idx = ks._numint.large_rho_indices(mol, dm, ks.grids, small_rho_cutoff)
        logger.debug(ks, 'Drop grids %d',
                     ks.grids.weights.size - numpy.count_nonzero(idx))
        ks.grids.coords  = numpy.asarray(ks.grids.coords [idx], order='C')
        ks.grids.weights = numpy.asarray(ks.grids.weights[idx], order='C')
        ks._numint.non0tab = None
    return vhf + vx


def energy_elec(ks, dm, h1e=None, vhf=None):
    r'''Electronic part of RKS energy.

    Args:
        ks : an instance of DFT class

        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian

    Returns:
        RKS electronic energy and the 2-electron part contribution
    '''
    if h1e is None:
        h1e = ks.get_hcore()
    e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
    tot_e = e1 + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


class RKS(pyscf.scf.hf.RHF):
    __doc__ = '''Restricted Kohn-Sham\n''' + pyscf.scf.hf.SCF.__doc__ + '''
    Attributes for RKS:
        xc : str
            'X_name,C_name' for the XC functional.  Default is 'lda,vwn'
        grids : Grids object
            grids.level (0 - 9)  big number for large mesh grids. Default is 3

            radii_adjust
                | radi.treutler_atomic_radii_adjust (default)
                | radi.becke_atomic_radii_adjust
                | None : to switch off atomic radii adjustment

            grids.atomic_radii
                | radi.BRAGG_RADII  (default)
                | radi.COVALENT_RADII
                | None : to switch off atomic radii adjustment

            grids.radi_method  scheme for radial grids
                | radi.treutler  (default)
                | radi.delley
                | radi.mura_knowles
                | radi.gauss_chebyshev

            grids.becke_scheme  weight partition function
                | gen_grid.original_becke  (default)
                | gen_grid.stratmann

            grids.prune  scheme to reduce number of grids
            | gen_grid.nwchem_prune  (default)
                | gen_grid.sg1_prune
                | gen_grid.treutler_prune
                | None : to switch off grids pruning

            grids.symmetry  True/False  to symmetrize mesh grids (TODO)

            grids.atom_grid  Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.

        small_rho_cutoff : float
            Drop grids if their contribution to total electrons smaller than
            this cutoff value.  Default is 1e-7.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'b3lyp'
    >>> mf.kernel()
    -76.415443079840458
    '''
    def __init__(self, mol):
        pyscf.scf.hf.RHF.__init__(self, mol)
        _dft_common_init_(self)

    def dump_flags(self):
        pyscf.scf.hf.RHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    get_veff = get_veff
    energy_elec = energy_elec

    def define_xc_(self, description):
        self.xc = description
        return self

def _dft_common_init_(mf):
    mf.xc = 'LDA,VWN'
    mf.grids = gen_grid.Grids(mf.mol)
    mf.small_rho_cutoff = 1e-7  # Use rho to filter grids
##################################################
# don't modify the following attributes, they are not input options
    mf._ecoul = 0
    mf._exc = 0
    mf._numint = numint._NumInt()
    mf._keys = mf._keys.union(['xc', 'grids', 'small_rho_cutoff'])


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

    m = RKS(mol)
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405

    m = RKS(mol)
    m._numint.libxc = xcfun
    m.xc = 'b88,lyp'
    print(m.scf())  # -2.8978518405
