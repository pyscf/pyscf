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


def get_veff_(ks, mol, dm, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will change the ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.  The ._exc and ._ecoul attributes
            will be updated after return.  Attributes ._dm_last, ._vj_last and
            ._vk_last might be changed if direct SCF method is applied.
        mol : an instance of :class:`Mole`

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

    hyb = vxc.hybrid_coeff(x_code, spin=(mol.spin>0)+1)

    if abs(hyb) < 1e-10:
        vj = ks.get_j(mol, dm, hermi)
    elif (ks._eri is not None or ks._is_mem_enough() or not ks.direct_scf):
        vj, vk = ks.get_jk(mol, dm, hermi)
    else:
        if (ks.direct_scf and isinstance(vhf_last, numpy.ndarray) and
            hasattr(ks, '_dm_last')):
            ddm = numpy.asarray(dm) - numpy.asarray(ks._dm_last)
            vj, vk = ks.get_jk(mol, ddm, hermi=hermi)
            vj += ks._vj_last
            vk += ks._vk_last
        else:
            vj, vk = ks.get_jk(mol, dm, hermi)
        ks._dm_last = dm
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
    e1 = numpy.einsum('ji,ji', h1e.conj(), dm).real
    tot_e = e1 + ks._ecoul + ks._exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', ks._ecoul, ks._exc)
    return tot_e, ks._ecoul+ks._exc


class RKS(pyscf.scf.hf.RHF):
    __doc__ = '''Restricted Kohn-Sham\n''' + pyscf.scf.hf.SCF.__doc__ + '''
    Attributes for RKS:
        xc : str
            'X_name,C_name' for the XC functional
        grids : Grids object
            grids.level (0 - 6)  big number for large mesh grids, default is 3

            grids.atomic_radii  can be one of
                | radi.treutler_atomic_radii_adjust(mol, radi.BRAGG_RADII)
                | radi.treutler_atomic_radii_adjust(mol, radi.COVALENT_RADII)
                | radi.becke_atomic_radii_adjust(mol, radi.BRAGG_RADII)
                | radi.becke_atomic_radii_adjust(mol, radi.COVALENT_RADII)
                | None,          to switch off atomic radii adjustment

            grids.radi_method  scheme for radial grids, can be one of
                | radi.treutler
                | radi.gauss_chebyshev

            grids.becke_scheme  weight partition function, can be one of
                | gen_grid.stratmann
                | gen_grid.original_becke

            grids.prune_scheme  scheme to reduce number of grids, can be one of
                | gen_grid.sg1_prune
                | gen_grid.nwchem_prune
                | gen_grid.treutler_prune

            grids.symmetry  True/False  to symmetrize mesh grids (TODO)

            grids.atom_grid  Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'b3lyp'
    >>> mf.kernel()
    -76.415443079840458
    '''
    def __init__(self, mol):
        pyscf.scf.hf.RHF.__init__(self, mol)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
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
        if h1e is None: h1e = self.get_hcore()
        return energy_elec(self, dm, h1e)


class ROKS(pyscf.scf.rohf.ROHF):
    '''Restricted open-shell Kohn-Sham
    See pyscf/dft/rks.py RKS class for the usage of the attributes'''
    def __init__(self, mol):
        pyscf.scf.rohf.ROHF.__init__(self, mol)
        self.xc = 'LDA,VWN'
        self.grids = gen_grid.Grids(mol)
##################################################
# don't modify the following attributes, they are not input options
        self._ecoul = 0
        self._exc = 0
        self._numint = numint._NumInt()
        self._keys = self._keys.union(['xc', 'grids'])

    def dump_flags(self):
        pyscf.scf.rohf.ROHF.dump_flags(self)
        logger.info(self, 'XC functionals = %s', self.xc)
        self.grids.dump_flags()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb + XC functional'''
        from pyscf.dft import uks
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return uks.get_veff_(self, mol, dm, dm_last, vhf_last, hermi)

    def energy_elec(self, dm, h1e=None, vhf=None):
        from pyscf.dft import uks
        if h1e is None: h1e = self.get_hcore()
        return uks.energy_elec(self, dm, h1e)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.build()

    m = RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
