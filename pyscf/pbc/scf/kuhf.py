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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

from functools import reduce
import numpy as np
import scipy.linalg
from pyscf.scf import hf as mol_hf
from pyscf.scf import uhf as mol_uhf
from pyscf.pbc.scf import khf
from pyscf.pbc.scf import uhf as pbcuhf
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'pbc_scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'pbc_scf_analyze_pre_orth_method', 'ANO')
CHECK_COULOMB_IMAG = getattr(__config__, 'pbc_scf_check_coulomb_imag', True)


canonical_occ = canonical_occ_ = addons.canonical_occ_


def make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs):
    '''Alpha and beta spin one particle density matrices for all k-points.

    Returns:
        dm_kpts : (2, nkpts, nao, nao) ndarray
    '''
    nkpts = len(mo_occ_kpts[0])
    nao, nmo = mo_coeff_kpts[0][0].shape
    def make_dm(mos, occs):
        return [np.dot(mos[k]*occs[k], mos[k].T.conj()) for k in range(nkpts)]
    dm_kpts =(make_dm(mo_coeff_kpts[0], mo_occ_kpts[0]) +
              make_dm(mo_coeff_kpts[1], mo_occ_kpts[1]))
    return lib.asarray(dm_kpts).reshape(2,nkpts,nao,nao)

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
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

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor

    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts =([mol_hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta)
                  for k, s in enumerate(s_kpts)],
                 [mol_hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb)
                  for k, s in enumerate(s_kpts)])
    return lib.asarray(f_kpts)

def get_fermi(mf, mo_energy_kpts=None, mo_occ_kpts=None):
    '''A pair of Fermi level for spin-up and spin-down orbitals
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    if mo_occ_kpts is None: mo_occ_kpts = mf.mo_occ

    # mo_energy_kpts and mo_occ_kpts are k-point UHF quantities
    assert(mo_energy_kpts[0][0].ndim == 1)
    assert(mo_occ_kpts[0][0].ndim == 1)

    nocca = sum(mo_occ.sum() for mo_occ in mo_occ_kpts[0])
    noccb = sum(mo_occ.sum() for mo_occ in mo_occ_kpts[1])
    # nocc may not be perfect integer when smearing is enabled
    nocca = int(nocca.round(3))
    noccb = int(noccb.round(3))

    fermi_a = np.sort(np.hstack(mo_energy_kpts[0]))[nocca-1]
    fermi_b = np.sort(np.hstack(mo_energy_kpts[1]))[noccb-1]

    for k, mo_e in enumerate(mo_energy_kpts[0]):
        mo_occ = mo_occ_kpts[0][k]
        if mo_occ[mo_e > fermi_a].sum() > 0.5:
            logger.warn(mf, 'Alpha occupied band above Fermi level: \n'
                        'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    for k, mo_e in enumerate(mo_energy_kpts[1]):
        mo_occ = mo_occ_kpts[1][k]
        if mo_occ[mo_e > fermi_b].sum() > 0.5:
            logger.warn(mf, 'Beta occupied band above Fermi level: \n'
                        'k=%d, mo_e=%s, mo_occ=%s', k, mo_e, mo_occ)
    return (fermi_a, fermi_b)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''

    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nocc_a, nocc_b = mf.nelec
    mo_energy = np.sort(np.hstack(mo_energy_kpts[0]))
    fermi_a = mo_energy[nocc_a-1]
    mo_occ_kpts = [[], []]
    for mo_e in mo_energy_kpts[0]:
        mo_occ_kpts[0].append((mo_e <= fermi_a).astype(np.double))
    if nocc_a < len(mo_energy):
        logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', fermi_a, mo_energy[nocc_a])
    else:
        logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', fermi_a)

    if nocc_b > 0:
        mo_energy = np.sort(np.hstack(mo_energy_kpts[1]))
        fermi_b = mo_energy[nocc_b-1]
        for mo_e in mo_energy_kpts[1]:
            mo_occ_kpts[1].append((mo_e <= fermi_b).astype(np.double))
        if nocc_b < len(mo_energy):
            logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g', fermi_b, mo_energy[nocc_b])
        else:
            logger.info(mf, 'beta HOMO = %.12g  (no LUMO because of small basis) ', fermi_b)

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  alpha mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[0][k][mo_occ_kpts[0][k]> 0],
                         mo_energy_kpts[0][k][mo_occ_kpts[0][k]==0])
        logger.debug(mf, '     k-point                  beta  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[1][k][mo_occ_kpts[1][k]> 0],
                         mo_energy_kpts[1][k][mo_occ_kpts[1][k]==0])
        np.set_printoptions(threshold=1000)

    return mo_occ_kpts


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(h1e_kpts)
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], h1e_kpts)
    e1+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], h1e_kpts)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], vhf_kpts[0]) * 0.5
    e_coul+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], vhf_kpts[1]) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if CHECK_COULOMB_IMAG and abs(e_coul.imag > mf.cell.precision*10):
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real


def mulliken_meta(cell, dm_ao_kpts, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''A modified Mulliken population analysis, based on meta-Lowdin AOs.

    Note this function only computes the Mulliken population for the gamma
    point density matrix.
    '''
    from pyscf.lo import orth
    if s is None:
        s = khf.get_ovlp(cell)
    log = logger.new_logger(cell, verbose)
    log.note('Analyze output for *gamma point*.')
    log.info('    To include the contributions from k-points, transform to a '
             'supercell then run the population analysis on the supercell\n'
             '        from pyscf.pbc.tools import k2gamma\n'
             '        k2gamma.k2gamma(mf).mulliken_meta()')
    log.note("KUHF mulliken_meta")
    dm_ao_gamma = dm_ao_kpts[:,0,:,:].real
    s_gamma = s[0,:,:].real
    c = orth.restore_ao_character(cell, pre_orth_method)
    orth_coeff = orth.orth_ao(cell, 'meta_lowdin', pre_orth_ao=c, s=s_gamma)
    c_inv = np.dot(orth_coeff.T, s_gamma)
    dm_a = reduce(np.dot, (c_inv, dm_ao_gamma[0], c_inv.T.conj()))
    dm_b = reduce(np.dot, (c_inv, dm_ao_gamma[1], c_inv.T.conj()))

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mol_uhf.mulliken_pop(cell, (dm_a,dm_b), np.eye(orth_coeff.shape[0]), log)


def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_fock(dm=dm)

    def eig_(fock, mo_coeff, idx, es, cs):
        if np.count_nonzero(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = reduce(np.dot, (orb.T.conj(), fock, orb))
            e, c = scipy.linalg.eigh(f1)
            es[idx] = e
            cs[:,idx] = np.dot(orb, c)

    mo_coeff = [[], []]
    mo_energy = [[], []]
    for k, mo in enumerate(mo_coeff_kpts[0]):
        mo1 = np.empty_like(mo)
        mo_e = np.empty_like(mo_occ_kpts[0][k])
        occidxa = mo_occ_kpts[0][k] == 1
        viridxa = ~occidxa
        eig_(fock[0][k], mo, occidxa, mo_e, mo1)
        eig_(fock[0][k], mo, viridxa, mo_e, mo1)
        mo_coeff[0].append(mo1)
        mo_energy[0].append(mo_e)
    for k, mo in enumerate(mo_coeff_kpts[1]):
        mo1 = np.empty_like(mo)
        mo_e = np.empty_like(mo_occ_kpts[1][k])
        occidxb = mo_occ_kpts[1][k] == 1
        viridxb = ~occidxb
        eig_(fock[1][k], mo, occidxb, mo_e, mo1)
        eig_(fock[1][k], mo, viridxb, mo_e, mo1)
        mo_coeff[1].append(mo1)
        mo_energy[1].append(mo_e)
    return mo_energy, mo_coeff

def init_guess_by_chkfile(cell, chkfile_name, project=None, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    from pyscf import gto
    chk_cell, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_cell, cell)

    if kpts is None:
        kpts = scf_rec['kpts']

    if 'kpt' in scf_rec:
        chk_kpts = scf_rec['kpt'].reshape(-1,3)
    elif 'kpts' in scf_rec:
        chk_kpts = scf_rec['kpts']
    else:
        chk_kpts = np.zeros((1,3))

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if 'kpts' not in scf_rec:  # gamma point or single k-point
        if mo.ndim == 2:
            mo = np.expand_dims(mo, axis=0)
            mo_occ = np.expand_dims(mo_occ, axis=0)
        else:  # UHF
            mo = [np.expand_dims(mo[0], axis=0),
                  np.expand_dims(mo[1], axis=0)]
            mo_occ = [np.expand_dims(mo_occ[0], axis=0),
                      np.expand_dims(mo_occ[1], axis=0)]

    if project:
        s = cell.pbc_intor('int1e_ovlp', kpts=kpts)
    def fproj(mo, kpts):
        if project:
            mo = addons.project_mo_nr2nr(chk_cell, mo, cell, kpts)
            for k, c in enumerate(mo):
                norm = np.einsum('pi,pi->i', c.conj(), s[k].dot(c))
                mo[k] /= np.sqrt(norm)
        return mo

    if kpts.shape == chk_kpts.shape and np.allclose(kpts, chk_kpts):
        def makedm(mos, occs):
            moa, mob = mos
            mos =([fproj(mo, None) for mo in moa],
                  [fproj(mo, None) for mo in mob])
            return make_rdm1(mos, occs)
    else:
        def makedm(mos, occs):
            where = [np.argmin(lib.norm(chk_kpts-kpt, axis=1)) for kpt in kpts]
            moa, mob = mos
            occa, occb = occs
            dkpts = [chk_kpts[w]-kpts[i] for i,w in enumerate(where)]
            mos = (fproj([moa[w] for w in where], dkpts),
                   fproj([mob[w] for w in where], dkpts))
            occs = ([occa[i] for i in where], [occb[i] for i in where])
            return make_rdm1(mos, occs)

    if getattr(mo[0], 'ndim', None) == 2:  # KRHF
        mo_occa = [(occ>1e-8).astype(np.double) for occ in mo_occ]
        mo_occb = [occ-mo_occa[k] for k,occ in enumerate(mo_occ)]
        dm = makedm((mo, mo), (mo_occa, mo_occb))
    else:  # KUHF
        dm = makedm(mo, mo_occ)

    # Real DM for gamma point
    if np.allclose(kpts, 0):
        dm = dm.real
    return dm


def dip_moment(cell, dm_kpts, unit='Debye', verbose=logger.NOTE,
               grids=None, rho=None, kpts=np.zeros((1,3))):
    ''' Dipole moment in the unit cell.

    Args:
         cell : an instance of :class:`Cell`

         dm_kpts (two lists of ndarrays) : KUHF density matrices of k-points

    Return:
        A list: the dipole moment on x, y and z components
    '''
    dm_kpts = dm_kpts[0] + dm_kpts[1]
    return khf.dip_moment(cell, dm_kpts, unit, verbose, grids, rho, kpts)

get_rho = khf.get_rho


class KUHF(pbcuhf.UHF, khf.KSCF):
    '''UHF class with k-point sampling.
    '''
    conv_tol = getattr(__config__, 'pbc_scf_KSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'pbc_scf_KSCF_conv_tol_grad', None)
    direct_scf = getattr(__config__, 'pbc_scf_SCF_direct_scf', False)

    def __init__(self, cell, kpts=np.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        khf.KSCF.__init__(self, cell, kpts, exxdiv)
        self.nelec = None

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            cell = self.cell
            nkpts = len(self.kpts)
            ne = cell.tot_electrons(nkpts)
            nalpha = (ne + cell.spin) // 2
            nbeta = nalpha - cell.spin
            if nalpha + nbeta != ne:
                raise RuntimeError('Electron number %d and spin %d are not consistent\n'
                                   'Note cell.spin = 2S = Nalpha - Nbeta, not 2S+1' %
                                   (ne, cell.spin))
            return nalpha, nbeta
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    def dump_flags(self, verbose=None):
        khf.KSCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per unit cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    build = khf.KSCF.build
    check_sanity = khf.KSCF.check_sanity

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        dm_kpts = None
        key = key.lower()
        if key == '1e' or key == 'hcore':
            dm_kpts = self.init_guess_by_1e(cell)
        elif getattr(cell, 'natm', 0) == 0:
            logger.info(self, 'No atom found in cell. Use 1e initial guess')
            dm_kpts = self.init_guess_by_1e(cell)
        elif key == 'atom':
            dm = self.init_guess_by_atom(cell)
        elif key[:3] == 'chk':
            try:
                dm_kpts = self.from_chk()
            except (IOError, KeyError):
                logger.warn(self, 'Fail to read %s. Use MINAO initial guess',
                            self.chkfile)
                dm = self.init_guess_by_minao(cell)
        else:
            dm = self.init_guess_by_minao(cell)

        if dm_kpts is None:
            nao = dm[0].shape[-1]
            nkpts = len(self.kpts)
            # dm[spin,nao,nao] at gamma point -> dm_kpts[spin,nkpts,nao,nao]
            dm_kpts = np.repeat(dm[:,None,:,:], nkpts, axis=1)
            dm_kpts[0,:] *= 1.01
            dm_kpts[1,:] *= 0.99  # To slightly break spin symmetry
            assert dm_kpts.shape[0]==2

        ne = np.einsum('xkij,kji->x', dm_kpts, self.get_ovlp(cell)).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relates to the charged system.
        nkpts = len(self.kpts)
        nelec = np.asarray(self.nelec)
        if np.any(abs(ne - nelec) > 1e-7*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                        'of initial guess density matrix (Ne/cell = %g)!\n'
                        '  This can cause huge error in Fock matrix and '
                        'lead to instability in SCF for low-dimensional '
                        'systems.\n  DM is normalized wrt the number '
                        'of electrons %s', ne.mean()/nkpts, nelec/nkpts)
            dm_kpts *= (nelec / ne).reshape(2,-1,1,1)
        return dm_kpts

    get_hcore = khf.KSCF.get_hcore
    get_ovlp = khf.KSCF.get_ovlp
    get_jk = khf.KSCF.get_jk
    get_j = khf.KSCF.get_j
    get_k = khf.KSCF.get_k
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec

    get_rho = khf.KSCF.get_rho

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = vj[0] + vj[1] - vk
        return vhf


    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        return khf.analyze(self, verbose, with_meta_lowdin, **kwargs)


    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        def grad(mo, mo_occ, fock):
            occidx = mo_occ > 0
            viridx = ~occidx
            g = reduce(np.dot, (mo[:,viridx].T.conj(), fock, mo[:,occidx]))
            return g.ravel()

        nkpts = len(self.kpts)
        grad_kpts = [grad(mo_coeff_kpts[0][k], mo_occ_kpts[0][k], fock[0][k])
                     for k in range(nkpts)]
        grad_kpts+= [grad(mo_coeff_kpts[1][k], mo_occ_kpts[1][k], fock[1][k])
                     for k in range(nkpts)]
        return np.hstack(grad_kpts)

    def eig(self, h_kpts, s_kpts):
        e_a, c_a = khf.KSCF.eig(self, h_kpts[0], s_kpts)
        e_b, c_b = khf.KSCF.eig(self, h_kpts[1], s_kpts)
        return (e_a,e_b), (c_a,c_b)

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None, **kwargs):
        if mo_coeff_kpts is None: mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts, **kwargs)

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
        (e_a,e_b), (c_a,c_b) = self.eig(fock, s1e)
        if single_kpt_band:
            e_a = e_a[0]
            e_b = e_b[0]
            c_a = c_a[0]
            c_b = c_b[0]
        return (e_a,e_b), (c_a,c_b)

    def init_guess_by_chkfile(self, chk=None, project=True, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)

    @lib.with_doc(mulliken_meta.__doc__)
    def mulliken_meta(self, cell=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    def mulliken_pop(self):
        raise NotImplementedError

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, cell=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        rho = kwargs.pop('rho', None)
        if rho is None:
            rho = self.get_rho(dm)
        return dip_moment(cell, dm, unit, verbose, rho=rho, kpts=self.kpts, **kwargs)

    @lib.with_doc(mol_uhf.spin_square.__doc__)
    def spin_square(self, mo_coeff=None, s=None):
        '''Treating the k-point sampling wfn as a giant Slater determinant,
        the spin_square value is the <S^2> of the giant determinant.
        '''
        nkpts = len(self.kpts)
        if mo_coeff is None:
            mo_a = [self.mo_coeff[0][k][:,self.mo_occ[0][k]>0] for k in range(nkpts)]
            mo_b = [self.mo_coeff[1][k][:,self.mo_occ[1][k]>0] for k in range(nkpts)]
        else:
            mo_a, mo_b = mo_coeff
        if s is None:
            s = self.get_ovlp()

        nelec_a = sum([mo_a[k].shape[1] for k in range(nkpts)])
        nelec_b = sum([mo_b[k].shape[1] for k in range(nkpts)])
        ssxy = (nelec_a + nelec_b) * .5
        for k in range(nkpts):
            sij = reduce(np.dot, (mo_a[k].T.conj(), s[k], mo_b[k]))
            ssxy -= np.einsum('ij,ij->', sij.conj(), sij).real
        ssz = (nelec_b-nelec_a)**2 * .25
        ss = ssxy + ssz
        s = np.sqrt(ss+.25) - .5
        return ss, s*2+1

    canonicalize = canonicalize

    dump_chk = khf.KSCF.dump_chk

    density_fit = khf.KSCF.density_fit
    # mix_density_fit inherits from khf.KSCF.mix_density_fit

    newton = khf.KSCF.newton
    x2c = x2c1e = sfx2c1e = khf.KSCF.sfx2c1e

    def stability(self,
                  internal=getattr(__config__, 'pbc_scf_KSCF_stability_internal', True),
                  external=getattr(__config__, 'pbc_scf_KSCF_stability_external', False),
                  verbose=None):
        from pyscf.pbc.scf.stability import uhf_stability
        return uhf_stability(self, internal, external, verbose)

    def convert_from_(self, mf):
        '''Convert given mean-field object to KUHF'''
        addons.convert_to_uhf(mf, self)
        return self

del(WITH_META_LOWDIN, PRE_ORTH_METHOD)


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 3
    cell.mesh = [11] * 3
    cell.verbose = 5
    cell.build()
    mf = KUHF(cell, [2,1,1])
    mf.kernel()
    mf.analyze()

