#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import sys

from functools import reduce
import numpy as np
import scipy.linalg
import h5py
from pyscf.pbc.scf import hf as pbchf
from pyscf import lib
from pyscf.scf import hf as mol_hf
from pyscf.lib import logger
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile  # noqa
from pyscf.pbc import tools
from pyscf.pbc import df
from pyscf.pbc.scf.rsjk import RangeSeparatedJKBuilder
from pyscf.pbc.lib.kpts import KPoints
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'pbc_scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'pbc_scf_analyze_pre_orth_method', 'ANO')
CHECK_COULOMB_IMAG = getattr(__config__, 'pbc_scf_check_coulomb_imag', True)


def get_ovlp(mf, cell=None, kpts=None):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    return pbchf.get_ovlp(cell, kpts)


def get_hcore(mf, cell=None, kpts=None):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    from pyscf.pbc.dft.multigrid import MultiGridNumInt
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    if hasattr(mf, '_numint') and isinstance(mf._numint, MultiGridNumInt):
        ni = mf._numint
    else:
        ni = mf.with_df
    if cell.pseudo:
        nuc = ni.get_pp(kpts)
    else:
        nuc = ni.get_nuc(kpts)
    if len(cell._ecpbas) > 0:
        from pyscf.pbc.gto import ecp
        nuc += lib.asarray(ecp.ecp_int(cell, kpts))
    t = lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    return nuc + t


def get_j(mf, cell, dm_kpts, kpts, kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.  It needs to be Hermitian.

    Kwargs:
        kpts_band : (k,3) ndarray
            A list of arbitrary "band" k-points at which to evaluate the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, with_k=False)[0]


def get_jk(mf, cell, dm_kpts, kpts, kpts_band=None, with_j=True, with_k=True,
           omega=None, **kwargs):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point. It needs to be Hermitian.

    Kwargs:
        kpts_band : (3,) ndarray
            A list of arbitrary "band" k-point at which to evaluate the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    return df.FFTDF(cell).get_jk(dm_kpts, kpts, kpts_band, with_j, with_k,
                                 omega, exxdiv=mf.exxdiv)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
    f_kpts = h1e_kpts + vhf_kpts
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f_kpts

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s_kpts is None: s_kpts = mf.get_ovlp()
    if dm_kpts is None: dm_kpts = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
        f_kpts = [mol_hf.damping(f, f_prev, damp_factor) for f,f_prev in zip(f_kpts,fock_last)]
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f_kpts = [mol_hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return lib.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''Fermi level
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ

    # mo_energy_kpts and mo_occ_kpts are k-point RHF quantities
    assert (mo_energy_kpts[0].ndim == 1)
    assert (mo_occ_kpts[0].ndim == 1)

    # occ array in mo_occ_kpts may have different size. See issue #250
    nocc = sum(mo_occ.sum() for mo_occ in mo_occ_kpts) / 2
    # nocc may not be perfect integer when smearing is enabled
    nocc = int(nocc.round(3))
    fermi = np.sort(np.hstack(mo_energy_kpts))[nocc-1]

    for k, mo_e in enumerate(mo_energy_kpts):
        mo_occ = mo_occ_kpts[k]
        if mo_occ[mo_e > fermi].sum() > 1.:
            logger.warn(mf, 'Occupied band above Fermi level: \n'
                        'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    return fermi

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts)
    nocc = mf.cell.tot_electrons(nkpts) // 2

    mo_energy = np.sort(np.hstack(mo_energy_kpts))
    nmo = mo_energy.size
    if nocc > nmo:
        raise RuntimeError('Failed to assign occupancies. '
                           f'Nocc ({nocc}) > Nmo ({nmo})')
    fermi = mo_energy[nocc-1]
    mo_occ_kpts = []
    for mo_e in mo_energy_kpts:
        mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)

    if nocc < nmo:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g (no LUMO)', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                         np.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
        np.set_printoptions(threshold=1000)

    return mo_occ_kpts


def get_grad(mo_coeff_kpts, mo_occ_kpts, fock):
    '''
    returns 1D array of gradients, like non K-pt version
    note that occ and virt indices of different k pts now occur
    in sequential patches of the 1D array
    '''
    nkpts = len(mo_occ_kpts)
    grad_kpts = [mol_hf.get_grad(mo_coeff_kpts[k], mo_occ_kpts[k], fock[k])
                 for k in range(nkpts)]
    return np.hstack(grad_kpts)


def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''One particle density matrices for all k-points.

    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    '''
    nkpts = len(mo_occ_kpts)
    dm = [mol_hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k]) for k in range(nkpts)]
    return lib.tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(dm_kpts)
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts, h1e_kpts)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if CHECK_COULOMB_IMAG and abs(e_coul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real


def analyze(mf, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    if verbose is None:
        verbose = mf.verbose
    mf.dump_scf_summary(verbose)

    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    pop, chg = mf.mulliken_meta(mf.cell, dm, s=ovlp_ao, verbose=verbose)
    dip = None
    if with_meta_lowdin:
        return (pop, chg), dip
    else:
        raise NotImplementedError
        #return mf.mulliken_pop(mf.cell, dm, s=ovlp_ao, verbose=verbose)


def _make_rdm1_meta(cell, dm_ao_kpts, kpts, pre_orth_method, s):
    from pyscf.lo import orth
    from pyscf.pbc.tools import k2gamma

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
    nkpts, nao = dm_ao_kpts.shape[:2]
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh)
    s_sc = k2gamma.to_supercell_ao_integrals(cell, kpts, s, kmesh=kmesh, force_real=False)
    orth_coeff = orth.orth_ao(scell, 'meta_lowdin', pre_orth_method, s=s_sc)[:,:nao] # cell 0 only
    c_inv = np.dot(orth_coeff.T.conj(), s_sc)
    c_inv = lib.einsum('aRp,Rk->kap', c_inv.reshape(nao,nkpts,nao), phase)
    dm = lib.einsum('kap,kpq,kbq->ab', c_inv, dm_ao_kpts, c_inv.conj())

    return dm


def mulliken_meta(cell, dm_ao_kpts, kpts, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''A modified Mulliken population analysis, based on meta-Lowdin AOs.
    The results are equivalent to the corresponding supercell calculation.
    '''
    log = logger.new_logger(cell, verbose)

    if s is None:
        s = get_ovlp(None, cell=cell, kpts=kpts)

    dm = _make_rdm1_meta(cell, dm_ao_kpts, kpts, pre_orth_method, s)

    log.note(' ** Mulliken pop on meta-lowdin orthogonal AOs **')
    return mol_hf.mulliken_pop(cell, dm, np.eye(dm.shape[0]), log)


def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)
    mo_coeff = []
    mo_energy = []
    for k, mo in enumerate(mo_coeff_kpts):
        mo1 = np.empty_like(mo)
        mo_e = np.empty_like(mo_occ_kpts[k])
        occidx = mo_occ_kpts[k] == 2
        viridx = ~occidx
        for idx in (occidx, viridx):
            if np.count_nonzero(idx) > 0:
                orb = mo[:,idx]
                f1 = reduce(np.dot, (orb.T.conj(), fock[k], orb))
                e, c = scipy.linalg.eigh(f1)
                mo1[:,idx] = np.dot(orb, c)
                mo_e[idx] = e
        mo_coeff.append(mo1)
        mo_energy.append(mo_e)
    return mo_energy, mo_coeff

def _cast_mol_init_guess(fn):
    def fn_init_guess(mf, cell=None, kpts=None):
        if cell is None: cell = mf.cell
        if kpts is None: kpts = mf.kpts
        dm = fn(cell)
        nkpts = len(kpts)
        dm_kpts = np.asarray([dm] * nkpts)
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = [dm.mo_coeff] * nkpts
            mo_occ = [dm.mo_occ] * nkpts
            dm_kpts = lib.tag_array(dm_kpts, mo_coeff=mo_coeff, mo_occ=mo_occ)
        return dm_kpts
    fn_init_guess.__name__ = fn.__name__
    fn_init_guess.__doc__ = (
        'Generates initial guess density matrix and the orbitals of the initial '
        'guess DM ' + fn.__doc__)
    return fn_init_guess

def init_guess_by_minao(cell, kpts=None):
    '''Generates initial guess density matrix and the orbitals of the initial
    guess DM based on ANO basis.
    '''
    return KSCF(cell).init_guess_by_minao(cell, kpts)

def init_guess_by_atom(cell, kpts=None):
    '''Generates initial guess density matrix and the orbitals of the initial
    guess DM based on the superposition of atomic HF density matrix.
    '''
    return KSCF(cell).init_guess_by_atom(cell, kpts)

def init_guess_by_chkfile(cell, chkfile_name, project=None, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    from pyscf.pbc.scf import kuhf
    dm = kuhf.init_guess_by_chkfile(cell, chkfile_name, project, kpts)
    return dm[0] + dm[1]


def dip_moment(cell, dm_kpts, unit='Debye', verbose=logger.NOTE,
               grids=None, rho=None, kpts=np.zeros((1,3))):
    ''' Dipole moment in the cell (is it well defined)?

    Args:
         cell : an instance of :class:`Cell`

         dm_kpts (a list of ndarrays) : density matrices of k-points

    Return:
        A list: the dipole moment on x, y and z components
    '''
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if grids is None:
        grids = gen_grid.UniformGrids(cell)
    if rho is None:
        rho = numint.KNumInt().get_rho(cell, dm_kpts, grids, kpts, cell.max_memory)
    return pbchf.dip_moment(cell, dm_kpts, unit, verbose, grids, rho, kpts)

def get_rho(mf, dm=None, grids=None, kpts=None):
    '''Compute density in real space
    '''
    from pyscf.pbc.dft import gen_grid
    from pyscf.pbc.dft import numint
    if dm is None:
        dm = mf.make_rdm1()
    if getattr(dm[0], 'ndim', None) != 2:  # KUHF
        dm = dm[0] + dm[1]
    if grids is None:
        grids = gen_grid.UniformGrids(mf.cell)
    if kpts is None:
        kpts = mf.kpts
    ni = numint.KNumInt()
    return ni.get_rho(mf.cell, dm, grids, kpts, mf.max_memory)

def gen_response(mf, mo_coeff=None, mo_occ=None,
                 singlet=None, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.scf._response_functions import _get_jk, _get_k
    cell = mf.cell
    kpts = mf.kpts
    if (singlet is None or singlet) and hermi != 2:
        def vind(dm1, kshift=0):
            vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
            return vj - .5 * vk
    else:
        def vind(dm1, kshift=0):
            return -.5 * _get_k(mf, cell, dm1, hermi, kpts, kshift)
    return vind

class KSCF(pbchf.SCF):
    '''SCF base class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    conv_tol_grad = getattr(__config__, 'pbc_scf_KSCF_conv_tol_grad', None)

    _keys = {'cell', 'exx_built', 'exxdiv', 'with_df', 'rsjk'}

    mol = pbchf.SCF.mol

    check_sanity = pbchf.SCF.check_sanity
    init_direct_scf = lib.invalid_method('init_direct_scf')
    get_hcore = get_hcore
    get_ovlp = get_ovlp
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    get_jk_incore = lib.invalid_method('get_jk_incore')
    energy_elec = energy_elec
    energy_nuc = pbchf.SCF.energy_nuc
    get_rho = get_rho
    init_guess_by_minao = _cast_mol_init_guess(mol_hf.init_guess_by_minao)
    init_guess_by_atom = _cast_mol_init_guess(mol_hf.init_guess_by_atom)
    _finalize = pbchf.SCF._finalize
    canonicalize = canonicalize

    def __init__(self, cell, kpts=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        mol_hf.SCF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        # Range separation JK builder
        self.rsjk = None

        self.exxdiv = exxdiv
        if kpts is not None:
            self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)

        self.exx_built = False

    @property
    def mo_energy_kpts(self):
        return self.mo_energy

    @property
    def mo_coeff_kpts(self):
        return self.mo_coeff

    @property
    def mo_occ_kpts(self):
        return self.mo_occ

    @property
    def kpts(self):
        if 'kpts' in self.__dict__:
            # To handle the attribute kpts loaded from chkfile
            self.kpts = self.__dict__.pop('kpts')
        return self.with_df.kpts

    @kpts.setter
    def kpts(self, x):
        kpts = np.reshape(x, (-1,3))
        self.with_df.kpts = kpts
        if self.rsjk:
            self.rsjk.kpts = kpts

    @property
    def kmesh(self):
        '''The number of k-points along each axis in the first Brillouin zone'''
        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
        kpts = self.kpts
        kmesh = kpts_to_kmesh(kpts)
        if len(kpts) != np.prod(kmesh):
            logger.WARN(self, 'K-points specified in %s are not Monkhorst-Pack %s grids',
                        self, kmesh)
        return kmesh

    @kmesh.setter
    def kmesh(self, x):
        self.kpts = self.cell.make_kpts(x)

    def build(self, cell=None):
        # To handle the attribute kpt or kpts loaded from chkfile
        if 'kpts' in self.__dict__:
            self.kpts = self.__dict__.pop('kpts')

        # "vcut_ws" precomputing is triggered by pbc.tools.pbc.get_coulG
        #if self.exxdiv == 'vcut_ws':
        #    if self.exx_built is False:
        #        self.precompute_exx()
        #    logger.info(self, 'WS alpha = %s', self.exx_alpha)

        kpts = self.kpts
        if self.rsjk:
            if not np.all(self.rsjk.kpts == kpts):
                self.rsjk = self.rsjk.__class__(cell, kpts)

        # for GDF and MDF
        with_df = self.with_df
        if len(kpts) > 1 and getattr(with_df, '_j_only', False):
            logger.warn(self, 'df.j_only cannot be used with k-point HF')
            with_df._j_only = False
            with_df.reset()

        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def reset(self, cell=None):
        pbchf.SCF.reset(self, cell)
        self.exx_built = False
        return self

    def dump_flags(self, verbose=None):
        mol_hf.SCF.dump_flags(self, verbose)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'N kpts = %d', len(self.kpts))
        logger.debug(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = tools.pbc.madelung(cell, [self.kpts])
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            nkpts = len(self.kpts)
            # FIXME: consider the fractional num_electron or not? This maybe
            # relates to the charged system.
            nelectron = float(self.cell.tot_electrons(nkpts)) / nkpts
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*nelectron * -.5)
        if getattr(self, 'smearing_method', None) is not None:
            logger.info(self, 'Smearing method = %s', self.smearing_method)
        logger.info(self, 'DF object = %s', self.with_df)
        if not getattr(self.with_df, 'build', None):
            # .dump_flags() is called in pbc.df.build function
            self.with_df.dump_flags(verbose)
        return self

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        raise NotImplementedError

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return mol_hf.SCF.init_guess_by_1e(self, cell)

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None,
              kpts_band=None, omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                           with_k=False, omega=omega)[0]

    def get_k(self, cell=None, dm_kpts=None, hermi=1, kpts=None,
              kpts_band=None, omega=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band,
                           with_j=False, omega=omega)[1]

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            vj, vk = self.rsjk.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                      with_j, with_k, omega=omega, exxdiv=self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                         with_j, with_k, omega=omega, exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        return vj - vk * .5

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)
        return get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []

        for k in range(nkpts):
            e, c = self._eigh(h_kpts[k], s_kpts[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return eig_kpts, mo_coeff_kpts

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

    def make_rdm2(self, mo_coeff_kpts, mo_occ_kpts, **kwargs):
        raise NotImplementedError

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_hcore(cell, kpts_band)
        fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        mo_energy, mo_coeff = self.eig(fock, s1e)
        if single_kpt_band:
            mo_energy = mo_energy[0]
            mo_coeff = mo_coeff[0]
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=None, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

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
                fh5['scf/kpts'] = self.kpts
        elif self.chkfile:
            with lib.H5FileWrap(self.chkfile, 'a') as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

    def analyze(mf, verbose=None, with_meta_lowdin=WITH_META_LOWDIN, **kwargs):
        raise NotImplementedError

    def mulliken_meta(self, cell=None, dm=None, kpts=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        raise NotImplementedError

    def mulliken_pop(self):
        raise NotImplementedError

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, cell=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if cell is None:
            cell = self.cell
        rho = kwargs.pop('rho', None)
        if rho is None:
            rho = self.get_rho(dm)
        return dip_moment(cell, dm, unit, verbose, rho=rho, kpts=self.kpts, **kwargs)

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import df_jk
        return df_jk.density_fit(self, auxbasis, with_df=with_df)

    def rs_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import rsdf_jk
        return rsdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def mix_density_fit(self, auxbasis=None, with_df=None):
        from pyscf.pbc.df import mdf_jk
        return mdf_jk.density_fit(self, auxbasis, with_df=with_df)

    def newton(self):
        from pyscf.pbc.scf import newton_ah
        return newton_ah.newton(self)

    def sfx2c1e(self):
        from pyscf.pbc.x2c import sfx2c1e
        return sfx2c1e.sfx2c1e(self)
    x2c = x2c1e = sfx2c1e

    def to_rhf(self):
        '''Convert the input mean-field object to a KRHF/KROHF/KRKS/KROKS object'''
        return addons.convert_to_rhf(self)

    def to_uhf(self):
        '''Convert the input mean-field object to a KUHF/KUKS object'''
        return addons.convert_to_uhf(self)

    def to_ghf(self):
        '''Convert the input mean-field object to a KGHF/KGKS object'''
        return addons.convert_to_ghf(self)

    def to_kscf(self):
        '''Convert to k-point SCF object
        '''
        return self

    def to_khf(self):
        '''Disable point group symmetry
        '''
        return self

    def convert_from_(self, mf):
        raise NotImplementedError

class KRHF(KSCF):
    '''RHF class with k-point sampling (default: gamma point).
    '''

    analyze = analyze
    spin_square = mol_hf.RHF.spin_square
    gen_response = gen_response
    to_gpu = lib.to_gpu

    def check_sanity(self):
        cell = self.cell
        if isinstance(self.kpts, KPoints):
            nkpts = self.kpts.nkpts
        else:
            nkpts = len(self.kpts)
        if cell.spin != 0 and nkpts % 2 != 0:
            logger.warn(self, 'Problematic nelec %s and number of k-points %d '
                        'found in KRHF method.', cell.nelec, nkpts)
        return KSCF.check_sanity(self)

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm = mol_hf.SCF.get_init_guess(self, cell, key)
        nkpts = len(self.kpts)
        if dm.ndim == 2:
            # dm[nao,nao] at gamma point -> dm_kpts[nkpts,nao,nao]
            dm = np.repeat(dm[None,:,:], nkpts, axis=0)
        dm_kpts = dm

        ne = lib.einsum('kij,kji->', dm_kpts, s1e).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relate to the charged system.
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 0.01*nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne/nkpts, nelectron/nkpts)
            dm_kpts *= (nelectron / ne).reshape(-1,1,1)
        return dm_kpts

    @lib.with_doc(mulliken_meta.__doc__)
    def mulliken_meta(self, cell=None, dm=None, kpts=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpts is None: kpts = self.kpts
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, kpts, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    def Gradients(self):
        from pyscf.pbc.grad import krhf
        return krhf.Gradients(self)

    def stability(self,
                  internal=getattr(__config__, 'pbc_scf_KSCF_stability_internal', True),
                  external=getattr(__config__, 'pbc_scf_KSCF_stability_external', False),
                  verbose=None,
                  return_status=False):
        from pyscf.pbc.scf.stability import rhf_stability
        return rhf_stability(self, internal, external, verbose, return_status)

    def to_ks(self, xc='HF'):
        '''Convert to RKS object.
        '''
        from pyscf.pbc import dft
        return self._transfer_attrs_(dft.KRKS(self.cell, self.kpts, xc=xc))

    def convert_from_(self, mf):
        '''Convert given mean-field object to KRHF/KROHF'''
        addons.convert_to_rhf(mf, self)
        return self
