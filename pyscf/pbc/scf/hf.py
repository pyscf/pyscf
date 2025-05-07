#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf.pbc.scf.khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import sys

import numpy as np
import h5py
from pyscf.scf import hf as mol_hf
from pyscf import lib
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.pbc import gto
from pyscf.pbc import tools
from pyscf.pbc.gto.pseudo import get_pp
from pyscf.pbc.scf import addons
from pyscf.pbc import df
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import __config__


def get_ovlp(cell, kpt=np.zeros(3)):
    '''Get the overlap AO matrix.
    '''
    precision = cell.precision * 1e-5
    rcut = max(cell.rcut, gto.estimate_rcut(cell, precision))
    with lib.temporary_env(cell, rcut=rcut, precision=precision):
        # Avoid pbcopt's prescreening in the lattice sum, for better accuracy
        s = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=kpt,
                           pbcopt=lib.c_null_ptr())
    s = np.asarray(s)
    hermi_error = abs(s - np.rollaxis(s.conj(), -1, -2)).max()
    if hermi_error > cell.precision and hermi_error > 1e-12:
        logger.warn(cell, '%.4g error found in overlap integrals. '
                    'cell.precision  or  cell.rcut  can be adjusted to '
                    'improve accuracy.', hermi_error)

    if cell.verbose >= logger.DEBUG:
        cond = np.max(lib.cond(s))
        if cond * precision > 1e2:
            prec = 1e7 / cond
            rmin = gto.estimate_rcut(cell, prec*1e-5)
            logger.warn(cell, 'Singularity detected in overlap matrix.  '
                        'Integral accuracy may be not enough.\n      '
                        'You can adjust  cell.precision  or  cell.rcut  to '
                        'improve accuracy.  Recommended settings are\n      '
                        'cell.precision < %.2g\n      '
                        'cell.rcut > %.4g', prec, rmin)
    return s


def get_hcore(cell, kpt=np.zeros(3)):
    '''Get the core Hamiltonian AO matrix.
    '''
    hcore = get_t(cell, kpt)
    if cell.pseudo:
        hcore += get_pp(cell, kpt)
    else:
        hcore += get_nuc(cell, kpt)
    if len(cell._ecpbas) > 0:
        from pyscf.pbc.gto import ecp
        hcore += ecp.ecp_int(cell, kpt)
    return hcore


def get_t(cell, kpt=np.zeros(3)):
    '''Get the kinetic energy AO matrix.
    '''
    return cell.pbc_intor('int1e_kin', hermi=1, kpts=kpt)


def get_nuc(cell, kpt=np.zeros(3)):
    '''Get the bare periodic nuc-el AO matrix, with G=0 removed.

    See Martin (12.16)-(12.21).
    '''
    return df.FFTDF(cell).get_nuc(kpt)


def get_j(cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            An arbitrary "band" k-point at which J is evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpts_band, with_k=False)[0]


def get_jk(mf, cell, dm, hermi=1, vhfopt=None, kpt=np.zeros(3),
           kpts_band=None, with_j=True, with_k=True, omega=None, **kwargs):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : ``(3,)`` ndarray or ``(*,3)`` ndarray
            An arbitrary "band" k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    return df.FFTDF(cell).get_jk(dm, hermi, kpt, kpts_band, with_j, with_k,
                                 omega, exxdiv=mf.exxdiv)


def get_bands(mf, kpts_band, cell=None, dm=None, kpt=None):
    '''Get energy bands at the given (arbitrary) 'band' k-points.

    Returns:
        mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
            Bands energies E_n(k)
        mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
            Band orbitals psi_n(k)
    '''
    if cell is None: cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt

    kpts_band = np.asarray(kpts_band)
    single_kpt_band = (getattr(kpts_band, 'ndim', None) == 1)
    kpts_band = kpts_band.reshape(-1,3)

    fock = mf.get_hcore(cell, kpts_band)
    fock = fock + mf.get_veff(cell, dm, kpt=kpt, kpts_band=kpts_band)
    s1e = mf.get_ovlp(cell, kpts_band)
    nkpts = len(kpts_band)
    mo_energy = []
    mo_coeff = []
    for k in range(nkpts):
        e, c = mf.eig(fock[k], s1e[k])
        mo_energy.append(e)
        mo_coeff.append(c)

    if single_kpt_band:
        mo_energy = mo_energy[0]
        mo_coeff = mo_coeff[0]
    return mo_energy, mo_coeff


def init_guess_by_chkfile(cell, chkfile_name, project=None, kpt=None):
    '''Read the HF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, (nao,nao) ndarray
    '''
    from pyscf.pbc.scf import uhf
    dm = uhf.init_guess_by_chkfile(cell, chkfile_name, project, kpt)
    return dm[0] + dm[1]

get_fock = mol_hf.get_fock
get_occ = mol_hf.get_occ
get_grad = mol_hf.get_grad
make_rdm1 = mol_hf.make_rdm1
energy_elec = mol_hf.energy_elec


def dip_moment(cell, dm, unit='Debye', verbose=logger.NOTE,
               grids=None, rho=None, kpt=np.zeros(3), origin=None):
    ''' Dipole moment in the cell (is it well defined)?

    Args:
         cell : an instance of :class:`Cell`

         dm (ndarray) : density matrix

    Return:
        A list: the dipole moment on x, y and z components
    '''
    from pyscf.pbc import tools
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if cell.dimension != 3:
        # raise NotImplementedError
        logger.warn(cell, 'Dipole moment for low-dimension system is not supported.')
        return np.zeros(3)

    log = logger.new_logger(cell, verbose)
    a = cell.lattice_vectors()
    b = np.linalg.inv(a).T

    if grids is None:
        grids = gen_grid.UniformGrids(cell)

    if rho is None:
        rho = numint.NumInt().get_rho(cell, dm, grids, kpt, cell.max_memory)

    # With the optimal origin of the unti cell, the net dipole in the unit
    # cell should be strictly zero. However, the integral grids are often not
    # enough to produce the zero dipole. Errors are caused by the sub-optimal
    # origin and the numerical integration.
    if origin is None:
        origin = _search_dipole_gauge_origin(cell, grids, rho, log)

    def shift_grids(r):
        r_frac = (r - origin).dot(b.T)
        r_frac5 = r_frac.round(5)
        # With origin at the center of the unit cell Grids on the edge
        # (r_frac == +/-0.5) of the new cell may lead to unbalanced
        # contributions to the dipole moment.
        r_frac[r_frac5== 0.5] = 0
        r_frac[r_frac5==-0.5] = 0
        r_frac[r_frac5 > 0.5] -= 1
        r_frac[r_frac5 <-0.5] += 1
        r = lib.dot(r_frac, a)
        return r

    r = shift_grids(grids.coords)
    e_dip = np.einsum('g,g,gx->x', rho, grids.weights, r)

    charges = cell.atom_charges()
    r = shift_grids(cell.atom_coords())
    nuc_dip = np.einsum('g,gx->x', charges, r)
    dip = nuc_dip - e_dip

    if unit.upper() == 'DEBYE':
        dip *= nist.AU2DEBYE
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *dip)
    return dip

def _search_dipole_gauge_origin(cell, grids, rho, log):
    '''Optimize the position of the unit cell in crystal. With a proper unit
    cell, the nuclear charge center and the electron density center can be
    placed at the same point (zero net dipole). This function returns the
    coordinates of the center of the unit cell.
    '''
    from pyscf.pbc.dft import gen_grid
    a = cell.lattice_vectors()
    b = np.linalg.inv(a).T
    charges = cell.atom_charges()
    coords  = cell.atom_coords()
    nelec = np.dot(rho, grids.weights)

    # The dipole moment in the crystal is not uniquely defined. Depending on
    # the position and the shape of the unit cell, the value of dipole moment
    # can be very different. The optimization below searches the origin of a
    # cell inside which the nuclear charge center and electron density charge
    # center locate at the same point.
    if (cell.dimension == 3 and
        # For orthogonal lattice only
        abs(np.diag(a.diagonal())).max() and
        isinstance(grids, gen_grid.UniformGrids)):

        r_nuc_frac = coords.dot(b.T)
        grid_frac = [np.fft.fftfreq(x) for x in grids.mesh]
        wxyz = grids.weights.reshape(grids.mesh)
        rhoxyz = rho.reshape(grids.mesh)

        def search_orig(ix):
            nx = grids.mesh[ix]
            den = np.einsum('xyz,xyz->'+('xyz'[ix]), rhoxyz, wxyz)

            nuc_x = r_nuc_frac[:,ix]
            grid_x = grid_frac[ix]
            # possible origin of the unit cell
            possible_orig = grid_x

            # shifts the unit cell
            possible_x = grid_x - possible_orig[:,None]
            possible_x[possible_x < 0] += 1
            possible_nuc_x = nuc_x - possible_orig[:,None]
            possible_nuc_x[possible_nuc_x < 0] += 1

            # Handle the grids on the edge of a unit cell.
            possible_nuc_x[possible_nuc_x==0] = .5
            possible_nuc_x[possible_nuc_x==1] = .5
            possible_x[possible_x==0] = .5
            possible_x[possible_x==1] = .5

            # dip in fractional coordinates
            dip  = np.einsum('sx,x->s', possible_nuc_x, charges)
            dip -= np.einsum('sx,x->s', possible_x, den)
            # Put the net charge at the center of the unit cell
            dip -= .5 * (charges.sum() - nelec)

            idx = abs(dip).argmin()
            dip_min = dip[idx]

            if abs(dip_min) > 1e-4:
                dip_nearest1 = dip[idx-1]
                dip_nearest2 = dip[(idx+1)%nx]
                if dip_nearest1 * dip_nearest2 > 0:
                    if dip_nearest1 * dip_min > dip_nearest2 * dip_min:
                        idx_nearest = idx - 1
                    else:
                        idx_nearest = idx + 1
                else:
                    if dip_nearest1 * dip_min < dip_nearest2 * dip_min:
                        idx_nearest = idx - 1
                    else:
                        idx_nearest = idx + 1

                dip_nearest = dip[idx_nearest%nx]
                if dip_nearest * dip_min < 0:
                    # extrapolation
                    idx = (idx_nearest*dip_min-idx*dip_nearest) / (dip_min-dip_nearest)

            # wraparound
            if idx >= nx // 2:
                idx -= nx
            shift = idx / nx * a[ix]
            return shift

        orig_x = search_orig(0)
        orig_y = search_orig(1)
        orig_z = search_orig(2)
        origin = orig_x + orig_y + orig_z
        log.debug1('optimized cell origin = %s', origin)
        log.debug1('  origin_x %s', orig_x)
        log.debug1('  origin_y %s', orig_y)
        log.debug1('  origin_z %s', orig_z)
        center = origin + .5 * a.sum(axis=0)
        log.debug1('gauge origin = %s', center)
    else:
        # If the grids are non-cubic grids, regenerating the grids is expensive if
        # the position or the shape of the unit cell is changed. The position of
        # the unit cell is not optimized. The gauge origin is set to the nuclear
        # charge center of the original unit cell.
        center = np.einsum('i,ix->x', charges, coords) / charges.sum()

    return center

def get_rho(mf, dm=None, grids=None, kpt=None):
    '''Compute density in real space
    '''
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if dm is None:
        dm = mf.make_rdm1()
    if getattr(dm, 'ndim', None) != 2:  # UHF
        dm = dm[0] + dm[1]
    if grids is None:
        grids = gen_grid.UniformGrids(mf.cell)
    if kpt is None:
        kpt = mf.kpt
    ni = numint.NumInt()
    return ni.get_rho(mf.cell, dm, grids, kpt, mf.max_memory)

def _dip_correction(mf):
    '''Makov-Payne corrections for charged systems.'''
    from pyscf.pbc import tools
    from pyscf.pbc.dft import gen_grid
    log = logger.new_logger(mf)
    cell = mf.cell
    a = cell.lattice_vectors()
    b = np.linalg.inv(a).T

    grids = gen_grid.UniformGrids(cell)
    ke_cutoff = gto.estimate_ke_cutoff(cell, 1e-5)
    grids.mesh = tools.cutoff_to_mesh(a, ke_cutoff)

    dm = mf.make_rdm1()
    rho = mf.get_rho(dm, grids)
    origin = _search_dipole_gauge_origin(cell, grids, rho, log)

    def shift_grids(r):
        r_frac = lib.dot(r - origin, b.T)
        # Grids on the boundary (r_frac == +/-0.5) of the new cell may lead to
        # unbalanced contributions to the dipole moment. Exclude them from the
        # dipole and quadrupole
        r_frac[r_frac== 0.5] = 0
        r_frac[r_frac==-0.5] = 0
        r_frac[r_frac > 0.5] -= 1
        r_frac[r_frac <-0.5] += 1
        r = lib.dot(r_frac, a)
        return r

    #           SC              BCC             FCC
    madelung = (-2.83729747948, -3.63923344951, -4.58486207411)
    vol = cell.vol
    L = vol ** (1./3)
    chg = cell.charge

    # epsilon is the dielectric constant of the system. For systems
    # surrounded by vacuum, epsilon = 1.
    epsilon = 1

    # Energy correction of point charges of a simple cubic lattice.
    de_mono = - chg**2 * np.array(madelung) / (2 * L * epsilon)

    # dipole energy correction
    r_e = shift_grids(grids.coords)
    r_nuc = shift_grids(cell.atom_coords())
    charges = cell.atom_charges()
    e_dip = np.einsum('g,g,gx->x', rho, grids.weights, r_e)
    nuc_dip = np.einsum('g,gx->x', charges, r_nuc)
    dip = nuc_dip - e_dip
    de_dip = -2.*np.pi/(3*cell.vol) * np.linalg.norm(dip)**2

    # quadrupole energy correction
    if abs(a - np.eye(3)*L).max() > 1e-5:
        logger.warn(mf, 'System is not cubic cell. Quadrupole energy '
                    'correction is inaccurate since it is developed based on '
                    'cubic cell.')
    e_quad = np.einsum('g,g,gx,gx->', rho, grids.weights, r_e, r_e)
    nuc_quad = np.einsum('g,gx,gx->', charges, r_nuc, r_nuc)
    quad = nuc_quad - e_quad
    de_quad = 2.*np.pi/(3*cell.vol) * quad

    de = de_mono + de_dip + de_quad
    return de_mono, de_dip, de_quad, de

def makov_payne_correction(mf):
    '''Makov-Payne correction (Phys. Rev. B, 51, 4014)
    '''
    cell = mf.cell
    logger.note(mf, 'Makov-Payne correction for charged 3D PBC systems')
    # PRB 51 (1995), 4014
    # PRB 77 (2008), 115139
    if cell.dimension != 3:
        logger.warn(mf, 'Correction for low-dimension PBC systems'
                    'is not available.')
        return 0

    de_mono, de_dip, de_quad, de = _dip_correction(mf)

    if mf.verbose >= logger.NOTE:
        write = mf.stdout.write
        write('Corrections (AU)\n')
        write('       Monopole      Dipole          Quadrupole    total\n')
        write('SC   %12.8f   %12.8f   %12.8f   %12.8f\n' %
              (de_mono[0], de_dip   , de_quad   , de[0]))
        write('BCC  %12.8f   %12.8f   %12.8f   %12.8f\n' %
              (de_mono[1], de_dip   , de_quad   , de[1]))
        write('FCC  %12.8f   %12.8f   %12.8f   %12.8f\n' %
              (de_mono[2], de_dip   , de_quad   , de[2]))
    return de


class SCF(mol_hf.SCF):
    '''SCF base class adapted for PBCs.

    Attributes:
        kpt : (3,) ndarray
            The AO k-point in Cartesian coordinates, in units of 1/Bohr.

        exxdiv : str
            Exchange divergence treatment, can be one of

            | None : ignore G=0 contribution in exchange
            | 'ewald' : Ewald probe charge correction [JCP 122, 234102 (2005); DOI:10.1063/1.1926272]

        with_df : density fitting object
            Default is the FFT based DF model. For all-electron calculation,
            MDF model is favored for better accuracy.  See also :mod:`pyscf.pbc.df`.
    '''

    _keys = {'cell', 'exxdiv', 'with_df', 'rsjk'}

    init_direct_scf = lib.invalid_method('init_direct_scf')
    get_bands = get_bands
    get_rho = get_rho

    def __init__(self, cell, kpt=np.zeros(3),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        self.kpt = kpt
        self.conv_tol = max(cell.precision * 10, 1e-8)

    @property
    def kpt(self):
        if 'kpt' in self.__dict__:
            # To handle the attribute kpt loaded from chkfile
            self.kpt = self.__dict__.pop('kpt')
        return self.with_df.kpts.reshape(3)
    @kpt.setter
    def kpt(self, x):
        self.with_df.kpts = np.reshape(x, (1, 3))
        if self.rsjk:
            self.rsjk.kpts = self.with_df.kpts

    @property
    def kpts(self):
        return self.with_df.kpts

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpt' in self.__dict__:
            self.kpt = self.__dict__.pop('kpt')

        if self.rsjk:
            if not np.all(self.rsjk.kpts == self.kpt):
                self.rsjk = self.rsjk.__class__(cell, self.kpt)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def reset(self, cell=None):
        '''Reset cell and relevant attributes associated to the old cell object'''
        if cell is not None:
            self.cell = cell
        self.with_df.reset(cell)
        return self

    # used by hf kernel
    @property
    def mol(self):
        return self.cell
    @mol.setter
    def mol(self, x):
        self.cell = x

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'kpt = %s', self.kpt)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, [self.kpt])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*cell.nelectron * -.5)
        if getattr(self, 'smearing_method', None) is not None:
            logger.info(self, 'Smearing method = %s', self.smearing_method)
        logger.info(self, 'DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags(verbose)
        return self

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
            isinstance(self.with_df, df.df.DF)):
            logger.warn(self, 'exxdiv %s is not supported in DF or MDF',
                        self.exxdiv)

        if self.verbose >= logger.DEBUG:
            s = self.get_ovlp()
            cond = np.max(lib.cond(s))
            if cond * 1e-17 > self.conv_tol:
                logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                            'SCF may be inaccurate and hard to converge.', cond)
        return self

    def get_hcore(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if cell.pseudo:
            nuc = self.with_df.get_pp(kpt)
        else:
            nuc = self.with_df.get_nuc(kpt)
        if len(cell._ecpbas) > 0:
            from pyscf.pbc.gto import ecp
            nuc += ecp.ecp_int(cell, kpt)
        return nuc + cell.pbc_intor('int1e_kin', 1, 1, kpt)

    def get_ovlp(self, cell=None, kpt=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        return get_ovlp(cell, kpt)

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        r'''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.
        for particular k-point (kpt).

        When kpts_band is given, the J, K matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]
            K_{pq} = \sum_{rs} (pr|sq) dm[r,s]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt

        cpu0 = (logger.process_clock(), logger.perf_counter())
        dm = np.asarray(dm)
        nao = dm.shape[-1]

        if (not omega and kpts_band is None and
            # TODO: generate AO integrals with rsjk algorithm
            not self.rsjk and
            (self.exxdiv == 'ewald' or not self.exxdiv) and
            (self._eri is not None or cell.incore_anyway or
             self._is_mem_enough())):
            if self._eri is None:
                logger.debug(self, 'Building PBC AO integrals incore')
                self._eri = self.with_df.get_ao_eri(kpt, compact=True)
            vj, vk = mol_hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)

            if (with_k and self.exxdiv == 'ewald' and
                # Note: for dimension=0 system, when the ERIs are generated by AFTDF,
                # the G=0 contributions are not properly encountered in the integral.
                # They must be explicitly evaluated.
                (cell.dimension != 0 or isinstance(self.with_df, df.AFTDF))):
                from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
                # G=0 is not included in the ._eri integrals
                _ewald_exxdiv_for_G0(self.cell, kpt, dm.reshape(-1,nao,nao),
                                     vk.reshape(-1,nao,nao))
        elif self.rsjk:
            vj, vk = self.rsjk.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                      with_j, with_k, omega, exxdiv=self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm.reshape(-1,nao,nao), hermi, kpt, kpts_band,
                                         with_j, with_k, omega, exxdiv=self.exxdiv)

        if with_j:
            vj = _format_jks(vj, dm, kpts_band)
        if with_k:
            vk = _format_jks(vk, dm, kpts_band)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_j(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        r'''Compute J matrix for the given density matrix and k-point (kpt).
        When kpts_band is given, the J matrices on kpts_band are evaluated.

            J_{pq} = \sum_{rs} (pq|rs) dm[s,r]

        where r,s are orbitals on kpt. p and q are orbitals on kpts_band
        if kpts_band is given otherwise p and q are orbitals on kpt.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, with_k=False,
                           omega=omega)[0]

    def get_k(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
              omega=None):
        '''Compute K matrix for the given density matrix.
        '''
        return self.get_jk(cell, dm, hermi, kpt, kpts_band, with_j=False,
                           omega=omega)[1]

    def get_veff(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                 kpt=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpt is None: kpt = self.kpt
        vj, vk = self.get_jk(cell, dm, hermi, kpt, kpts_band)
        vhf = vj - vk * .5
        return vhf

    def get_jk_incore(self, cell=None, dm=None, hermi=1, kpt=None, omega=None,
                      **kwargs):
        '''Get Coulomb (J) and exchange (K) following :func:`scf.hf.RHF.get_jk_`.

        *Incore* version of Coulomb and exchange build only.
        Currently RHF always uses PBC AO integrals (unlike RKS), since
        exchange is currently computed by building PBC AO integrals.
        '''
        if omega:
            raise NotImplementedError
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if self._eri is None:
            self._eri = self.with_df.get_ao_eri(kpt, compact=True)
        return self.get_jk(cell, dm, hermi, kpt)

    def energy_nuc(self):
        cell = self.cell
        if (cell.dimension == 0 and
            # Integrals in GDF and RSJK are all computed in real space.
            # Nuclear repulsion should be computed in real space.
            not isinstance(self.with_df, df.AFTDF)):
            from pyscf.gto.mole import classical_coulomb_energy
            return classical_coulomb_energy(cell)
        return cell.enuc

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, cell=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if cell is None:
            cell = self.cell
        rho = kwargs.pop('rho', None)
        if rho is None:
            rho = self.get_rho(dm)
        return dip_moment(cell, dm, unit, verbose, rho=rho, kpt=self.kpt, **kwargs)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        mol_hf.SCF._finalize(self)

        if self.cell.charge != 0:
            makov_payne_correction(self)
        return self

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if cell is None: cell = self.cell
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        dm = normalize_dm_(self, dm, s1e)
        return dm

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return mol_hf.SCF.init_guess_by_1e(self, cell)

    def init_guess_by_chkfile(self, chk=None, project=None, kpt=None):
        if chk is None: chk = self.chkfile
        if kpt is None: kpt = self.kpt
        return init_guess_by_chkfile(self.cell, chk, project, kpt)
    def from_chk(self, chk=None, project=None, kpt=None):
        return self.init_guess_by_chkfile(chk, project, kpt)

    def dump_chk(self, envs_or_file):
        '''Serialize the SCF object and save it to the specified chkfile.

        Args:
            envs_or_file:
                If this argument is a file path, the serialized SCF object is
                saved to the file specified by this argument.
                If this attribute is a dict (created by locals()), the necessary
                variables are saved to the file specified by the attribute mf.chkfile.
        '''
        mol_hf.SCF.dump_chk(self, envs_or_file)
        if isinstance(envs_or_file, str):
            with lib.H5FileWrap(envs_or_file, 'a') as fh5:
                fh5['scf/kpt'] = self.kpt
        elif self.chkfile:
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpt'] = self.kpt
        return self

    def _is_mem_enough(self):
        nao = self.cell.nao_nr()
        if abs(self.kpt).sum() < 1e-9:
            mem_need = nao**4*8/4/1e6
        else:
            mem_need = nao**4*16/1e6
        return mem_need + lib.current_memory()[0] < self.max_memory*.95

    def analyze(self, verbose=None, with_meta_lowdin=None, **kwargs):
        raise NotImplementedError

    def mulliken_pop(self):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df=with_df)

    def rs_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import rsdf_jk
        return rsdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def mix_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import mdf_jk
        return mdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def sfx2c1e(self):
        from pyscf.pbc.x2c import sfx2c1e
        return sfx2c1e.sfx2c1e(self)
    x2c = x2c1e = sfx2c1e

    def to_rhf(self):
        '''Convert the input mean-field object to a RHF/ROHF/RKS/ROKS object'''
        return addons.convert_to_rhf(self)

    def to_uhf(self):
        '''Convert the input mean-field object to a UHF/UKS object'''
        return addons.convert_to_uhf(self)

    def to_ghf(self):
        '''Convert the input mean-field object to a GHF/GKS object'''
        return addons.convert_to_ghf(self)

    def to_kscf(self):
        '''Convert gamma point SCF object to k-point SCF object
        '''
        from pyscf.pbc.scf.addons import convert_to_kscf
        return convert_to_kscf(self)

    def jk_method(self, J='FFTDF', K=None):
        '''
        Set up the schemes to evaluate Coulomb and exchange matrix

        FFTDF: planewave density fitting using Fast Fourier Transform
        AFTDF: planewave density fitting using analytic Fourier Transform
        GDF: Gaussian density fitting
        MDF: Gaussian and planewave mix density fitting
        RS: range-separation JK builder
        RSDF: range-separation density fitting
        '''
        if K is None:
            K = J

        if J != K:
            raise NotImplementedError('J != K')

        if 'DF' in J or 'DF' in K:
            if 'DF' in J and 'DF' in K:
                assert J == K
            else:
                df_method = J if 'DF' in J else K
                self.with_df = getattr(df, df_method)(self.cell, self.kpts)

        if 'RS' in J or 'RS' in K:
            self.rsjk = RangeSeparatedJKBuilder(self.cell, self.kpts)
            self.rsjk.verbose = self.verbose

        # For nuclear attraction
        if J == 'RS' and K == 'RS' and not isinstance(self.with_df, df.GDF):
            self.with_df = df.GDF(self.cell, self.kpts)

        nuc = self.with_df.__class__.__name__
        logger.debug1(self, 'Apply %s for J, %s for K, %s for nuc', J, K, nuc)
        return self

    def to_gpu(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError


class KohnShamDFT:
    '''A mock DFT base class

    The base class is defined in the pbc.dft.rks module. This class can
    be used to verify if an SCF object is an pbc-Hartree-Fock method or an
    pbc-DFT method. It should be overwritten by the actual KohnShamDFT class
    when loading dft module.
    '''


class RHF(SCF):

    analyze = mol_hf.RHF.analyze
    spin_square = mol_hf.RHF.spin_square
    stability = mol_hf.RHF.stability
    to_gpu = lib.to_gpu

    def to_ks(self, xc='HF'):
        '''Convert to RKS object.
        '''
        from pyscf.pbc import dft
        return self._transfer_attrs_(dft.RKS(self.cell, xc=xc))

    def convert_from_(self, mf):
        '''Convert given mean-field object to RHF/ROHF'''
        addons.convert_to_rhf(mf, self)
        return self


def _format_jks(vj, dm, kpts_band):
    if kpts_band is None:
        vj = vj.reshape(dm.shape)
    elif kpts_band.ndim == 1:  # a single k-point on bands
        vj = vj.reshape(dm.shape)
    elif getattr(dm, "ndim", 0) == 2:
        vj = vj[0]
    return vj

def normalize_dm_(mf, dm, s1e=None):
    '''
    Scale density matrix to make it produce the correct number of electrons.
    '''
    cell = mf.cell
    if s1e is None:
        s1e = mf.get_ovlp(cell)
    ne = lib.einsum('ij,ji->', dm, s1e).real
    if abs(ne - cell.nelectron) > 0.01:
        logger.debug(mf, 'Big error detected in the electron number '
                     'of initial guess density matrix (Ne/cell = %g)!\n'
                     '  This can cause huge error in Fock matrix and '
                     'lead to instability in SCF for low-dimensional '
                     'systems.\n  DM is normalized wrt the number '
                     'of electrons %s', ne, cell.nelectron)
        dm *= cell.nelectron / ne
    return dm
