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
from pyscf.pbc.scf import chkfile  # noqa
from pyscf import __config__

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
    dm = lib.asarray(dm_kpts).reshape(2,nkpts,nao,nao)
    return lib.tag_array(dm, mo_coeff=mo_coeff_kpts, mo_occ=mo_occ_kpts)

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

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4 and fock_last is not None:
        f_a = []
        f_b = []
        for k in range(len(s_kpts)):
            f_a.append(mol_hf.damping(f_kpts[0][k], fock_last[0][k], dampa))
            f_b.append(mol_hf.damping(f_kpts[1][k], fock_last[1][k], dampa))
        f_kpts = [f_a, f_b]
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts, f_prev=fock_last)
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
    assert (mo_energy_kpts[0][0].ndim == 1)
    assert (mo_occ_kpts[0][0].ndim == 1)

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
    else:
        for mo_e in mo_energy_kpts[1]:
            mo_occ_kpts[1].append(np.zeros_like(mo_e))

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  alpha mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            if (np.count_nonzero(mo_occ_kpts[0][k]) > 0 and
                np.count_nonzero(mo_occ_kpts[0][k] == 0) > 0):
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                             k, kpt[0], kpt[1], kpt[2],
                             mo_energy_kpts[0][k][mo_occ_kpts[0][k]> 0],
                             mo_energy_kpts[0][k][mo_occ_kpts[0][k]==0])
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[0][k])
        logger.debug(mf, '     k-point                  beta  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            if (np.count_nonzero(mo_occ_kpts[1][k]) > 0 and
                np.count_nonzero(mo_occ_kpts[1][k] == 0) > 0):
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                             k, kpt[0], kpt[1], kpt[2],
                             mo_energy_kpts[1][k][mo_occ_kpts[1][k]> 0],
                             mo_energy_kpts[1][k][mo_occ_kpts[1][k]==0])
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[1][k])
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
    if CHECK_COULOMB_IMAG and abs(e_coul.imag) > mf.cell.precision*10:
        logger.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real


def _make_rdm1_meta(cell, dm_ao_kpts, kpts, pre_orth_method, s):
    from pyscf.lo import orth
    from pyscf.pbc.tools import k2gamma

    kmesh = k2gamma.kpts_to_kmesh(cell, kpts-kpts[0])
    nkpts, nao = dm_ao_kpts[0].shape[:2]
    scell, phase = k2gamma.get_phase(cell, kpts, kmesh)
    s_sc = k2gamma.to_supercell_ao_integrals(cell, kpts, s, kmesh=kmesh, force_real=False)
    orth_coeff = orth.orth_ao(scell, 'meta_lowdin', pre_orth_method, s=s_sc)[:,:nao] # cell 0 only
    c_inv = np.dot(orth_coeff.T.conj(), s_sc)
    c_inv = lib.einsum('aRp,Rk->kap', c_inv.reshape(nao,nkpts,nao), phase)
    dm_a = lib.einsum('kap,kpq,kbq->ab', c_inv, dm_ao_kpts[0], c_inv.conj())
    dm_b = lib.einsum('kap,kpq,kbq->ab', c_inv, dm_ao_kpts[1], c_inv.conj())

    return (dm_a, dm_b)


def mulliken_meta(cell, dm_ao_kpts, kpts, verbose=logger.DEBUG,
                  pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''A modified Mulliken population analysis, based on meta-Lowdin AOs.
    The results are equivalent to the corresponding supercell calculation.
    '''
    log = logger.new_logger(cell, verbose)

    if s is None:
        s = khf.get_ovlp(None, cell=cell, kpts=kpts)

    dm_a, dm_b = _make_rdm1_meta(cell, dm_ao_kpts, kpts, pre_orth_method, s)

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mol_uhf.mulliken_pop(cell, (dm_a,dm_b), np.eye(dm_a.shape[0]), log)


def mulliken_meta_spin(cell, dm_ao_kpts, kpts, verbose=logger.DEBUG,
                       pre_orth_method=PRE_ORTH_METHOD, s=None):
    '''A modified Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    log = logger.new_logger(cell, verbose)

    if s is None:
        s = khf.get_ovlp(None, cell=cell, kpts=kpts)

    dm_a, dm_b = _make_rdm1_meta(cell, dm_ao_kpts, kpts, pre_orth_method, s)

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return mol_uhf.mulliken_spin_pop(cell, (dm_a,dm_b), np.eye(dm_a.shape[0]), log)


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

    def makedm(mos, occs):
        moa, mob = mos
        mos = (fproj(moa, chk_kpts), fproj(mob, chk_kpts))
        dm = make_rdm1(mos, occs)
        if kpts.shape != chk_kpts.shape or not np.allclose(kpts, chk_kpts):
            dm = [addons.project_dm_k2k(cell, dm[0], chk_kpts, kpts),
                  addons.project_dm_k2k(cell, dm[1], chk_kpts, kpts)]
        return np.asarray(dm)

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
    ''' Dipole moment in the cell.

    Args:
         cell : an instance of :class:`Cell`

         dm_kpts (two lists of ndarrays) : KUHF density matrices of k-points

    Return:
        A list: the dipole moment on x, y and z components
    '''
    dm_kpts = dm_kpts[0] + dm_kpts[1]
    return khf.dip_moment(cell, dm_kpts, unit, verbose, grids, rho, kpts)

get_rho = khf.get_rho

def gen_response(mf, mo_coeff=None, mo_occ=None,
                 with_j=True, hermi=0, max_memory=None, with_nlc=True):
    from pyscf.pbc.scf._response_functions import _get_jk, _get_k
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if with_j:
        def vind(dm1, kshift=0):
            vj, vk = _get_jk(mf, cell, dm1, hermi, kpts, kshift)
            v1 = vj[0] + vj[1] - vk
            return v1
    else:
        def vind(dm1, kshift=0):
            return -_get_k(mf, cell, dm1, hermi, kpts, kshift)
    return vind

class KUHF(khf.KSCF):
    '''UHF class with k-point sampling (default: gamma point).
    '''
    conv_tol_grad = getattr(__config__, 'pbc_scf_KSCF_conv_tol_grad', None)
    init_guess_breaksym = getattr(__config__, 'scf_uhf_init_guess_breaksym', 1)

    _keys = {"init_guess_breaksym"}

    init_guess_by_1e     = pbcuhf.UHF.init_guess_by_1e
    init_guess_by_minao  = pbcuhf.UHF.init_guess_by_minao
    init_guess_by_atom   = pbcuhf.UHF.init_guess_by_atom
    init_guess_by_huckel = pbcuhf.UHF.init_guess_by_huckel
    init_guess_by_mod_huckel = pbcuhf.UHF.init_guess_by_mod_huckel
    get_fock = get_fock
    get_fermi = get_fermi
    get_occ = get_occ
    energy_elec = energy_elec
    get_rho = get_rho
    analyze = khf.analyze
    canonicalize = canonicalize
    gen_response = gen_response
    to_gpu = lib.to_gpu

    def __init__(self, cell, kpts=None,
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
        logger.info(self, 'number of electrons per cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm_kpts = mol_hf.SCF.get_init_guess(self, cell, key)
        assert dm_kpts.shape[0] == 2
        nkpts = len(self.kpts)
        if dm_kpts.ndim != 4:
            # dm[spin,nao,nao] at gamma point -> dm_kpts[spin,nkpts,nao,nao]
            dm_kpts = np.repeat(dm_kpts[:,None,:,:], nkpts, axis=1)

        ne = lib.einsum('xkij,kji->x', dm_kpts, s1e).real
        nelec = np.asarray(self.nelec)
        if np.any(abs(ne - nelec) > 0.01*nkpts):
            logger.debug(self, 'Big error detected in the electron number '
                         'of initial guess density matrix (Ne/cell = %g)!\n'
                         '  This can cause huge error in Fock matrix and '
                         'lead to instability in SCF for low-dimensional '
                         'systems.\n  DM is normalized wrt the number '
                         'of electrons %s', ne.mean()/nkpts, nelec/nkpts)
            dm_kpts *= (nelec / ne).reshape(2,-1,1,1)
        return dm_kpts

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = vj[0] + vj[1] - vk
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        def grad(mo, mo_occ, fock):
            occidx = mo_occ > 0
            viridx = ~occidx
            g = reduce(np.dot, (mo[:,viridx].T.conj(), fock, mo[:,occidx]))
            return g.ravel()

        nkpts = len(mo_occ_kpts[0])
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
    def mulliken_meta(self, cell=None, dm=None, kpts=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpts is None: kpts = self.kpts
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, kpts, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    @lib.with_doc(mulliken_meta_spin.__doc__)
    def mulliken_meta_spin(self, cell=None, dm=None, kpts=None, verbose=logger.DEBUG,
                           pre_orth_method=PRE_ORTH_METHOD, s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if kpts is None: kpts = self.kpts
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta_spin(cell, dm, kpts, s=s, verbose=verbose,
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

    def stability(self,
                  internal=getattr(__config__, 'pbc_scf_KSCF_stability_internal', True),
                  external=getattr(__config__, 'pbc_scf_KSCF_stability_external', False),
                  verbose=None):
        from pyscf.pbc.scf.stability import uhf_stability
        return uhf_stability(self, internal, external, verbose)

    def Gradients(self):
        from pyscf.pbc.grad import kuhf
        return kuhf.Gradients(self)

    def to_ks(self, xc='HF'):
        '''Convert to RKS object.
        '''
        from pyscf.pbc import dft
        return self._transfer_attrs_(dft.KUKS(self.cell, self.kpts, xc=xc))

    def convert_from_(self, mf):
        '''Convert given mean-field object to KUHF'''
        addons.convert_to_uhf(mf, self)
        return self
