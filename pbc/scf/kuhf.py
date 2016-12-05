#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Hartree-Fock for periodic systems with k-point sampling

See Also:
    hf.py : Hartree-Fock for periodic systems at a single k-point
'''

import time
import numpy as np
import scipy.linalg
import h5py
from pyscf.scf import hf
from pyscf.scf import uhf
from pyscf.pbc.scf import khf
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import addons


def make_rdm1(mo_coeff_kpts, mo_occ_kpts):
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

def get_fock(mf, h1e_kpts, s_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor

    f_kpts = h1e_kpts + vhf_kpts
    if adiis and cycle >= diis_start_cycle:
        f_kpts = adiis.update(s_kpts, dm_kpts, f_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts =([hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta)
                  for k, s in enumerate(s_kpts)] +
                 [hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb)
                  for k, s in enumerate(s_kpts)])
    return lib.asarray(f_kpts)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    mo_occ_kpts = np.zeros_like(mo_energy_kpts)

    nkpts = len(mo_energy_kpts[0])
    nocc = mf.cell.nelectron * nkpts

    # TODO: implement Fermi smearing and print mo_energy kpt by kpt
    mo_energy = np.sort(mo_energy_kpts.ravel())
    fermi = mo_energy[nocc-1]
    mo_occ_kpts[mo_energy_kpts <= fermi] = 1

    if nocc < mo_energy.size:
        logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                    mo_energy[nocc-1], mo_energy[nocc])
        if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
            logger.warn(mf, '!! HOMO %.12g == LUMO %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
    else:
        logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(mo_energy))
        logger.debug(mf, '     k-point                  alpha mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[0,k,mo_occ_kpts[0,k]> 0],
                         mo_energy_kpts[0,k,mo_occ_kpts[0,k]==0])
        logger.debug(mf, '     k-point                  beta  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[1,k,mo_occ_kpts[1,k]> 0],
                         mo_energy_kpts[1,k,mo_occ_kpts[1,k]==0])
        np.set_printoptions()

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
    if abs(e_coul.imag > 1.e-10):
        raise RuntimeError("Coulomb energy has imaginary part, "
                           "something is wrong!", e_coul.imag)
    e1 = e1.real
    e_coul = e_coul.real
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul

def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    mo_occ_kpts = np.asarray(mo_occ_kpts)
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_hcore() + mf.get_jk(mol, dm)
    occidx = mo_occ_kpts == 2
    viridx = ~occidx
    mo_coeff_kpts = mo_coeff_kpts.copy()
    mo_e = np.empty_like(mo_occ_kpts)

    def eig_(fock, mo_coeff_kpts, idx, es, cs):
        if np.count_nonzero(idx) > 0:
            orb = mo_coeff_kpts[:,idx]
            f1 = reduce(np.dot, (orb.T.conj(), fock, orb))
            e, c = scipy.linalg.eigh(f1)
            es[idx] = e
            cs[:,idx] = np.dot(orb, c)

    for k, mo in enumerate(mo_coeff_kpts[0]):
        occidxa = mo_occ_kpts[0][k] == 1
        viridxa = ~occidxa
        eig_(fock[0][k], mo, occidxa, mo_e[0,k], mo)
        eig_(fock[0][k], mo, viridxa, mo_e[0,k], mo)
    for k, mo in enumerate(mo_coeff_kpts[1]):
        occidxb = mo_occ_kpts[1][k] == 1
        viridxb = ~occidxb
        eig_(fock[1][k], mo, occidxb, mo_e[1,k], mo)
        eig_(fock[1][k], mo, viridxb, mo_e[1,k], mo)
    return mo_e, mo

def init_guess_by_chkfile(cell, chkfile_name, project=True, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    chk_cell, scf_rec = pyscf.pbc.scf.chkfile.load_scf(chkfile_name)

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
            mo = mo.reshape((1,)+mo.shape)
            mo_occ = mo_occ.reshape((1,)+mo_occ.shape)
        else:  # UHF
            mo = mo.reshape((2,1)+mo.shape[1:])
            mo_occ = mo_occ.reshape((2,1)+mo_occ.shape[1:])

    def fproj(mo, kpt):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell, kpt)
        else:
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
            mos = ([fproj(moa[w], chk_kpts[w]-kpts[i]) for i,w in enumerate(where)],
                   [fproj(mob[w], chk_kpts[w]-kpts[i]) for i,w in enumerate(where)])
            occs = (occa[where],occb[where])
            return make_rdm1(mos, occs)

    if mo.ndim == 3:  # KRHF
        dm = makedm((mo, mo), (mo_occ*.5, mo_occ*.5))
    else:  # KUHF
        dm = makedm(mo, mo_occ)

    # Real DM for gamma point
    if np.allclose(kpts, 0):
        dm = dm.real
    return dm


class KUHF(uhf.UHF, khf.KRHF):
    '''UHF class with k-point sampling.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), exxdiv='ewald'):
        from pyscf.pbc import df
        self.cell = cell
        uhf.UHF.__init__(self, cell)

        self.with_df = df.FFTDF(cell)
        self.exxdiv = exxdiv
        self.kpts = kpts
        self.direct_scf = False

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv', 'with_df'])

    @property
    def kpts(self):
        return self.with_df.kpts
    @kpts.setter
    def kpts(self, x):
        self.with_df.kpts = np.reshape(x, (-1,3))

    def dump_flags(self):
        uhf.UHF.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC SCF flags ********')
        logger.info(self, 'N kpts = %d', len(self.kpts))
        logger.debug(self, 'kpts = %s', self.kpts)
        logger.info(self, 'DF object = %s', self.with_df)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s', self.exxdiv)
        #if self.exxdiv == 'vcut_ws':
        #    if self.exx_built is False:
        #        self.precompute_exx()
        #    logger.info(self, 'WS alpha = %s', self.exx_alpha)

    def build(self, cell=None):
        uhf.UHF.build(self, cell)
        #if self.exxdiv == 'vcut_ws':
        #    self.precompute_exx()

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = uhf.UHF.get_init_guess(self, cell, key)
        if key.lower() == 'chkfile':
            dm_kpts = dm
        else:
            nao = dm.shape[1]
            nkpts = len(self.kpts)
            dm_kpts = lib.asarray([dm]*nkpts).reshape(nkpts,2,nao,nao)
            dm_kpts = dm_kpts.transpose(1,0,2,3)
            dm[1,:] *= .98  # To break spin symmetry
        return dm_kpts

    get_hcore = khf.KRHF.get_hcore
    get_ovlp = khf.KRHF.get_ovlp
    get_jk = khf.KRHF.get_jk
    get_j = khf.KRHF.get_j
    get_k = khf.KRHF.get_k

    get_fock = get_fock
    get_occ = get_occ
    energy_elec = energy_elec

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpt_band=None):
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpt_band)
        vhf = uhf._makevhf(vj, vk)
        return vhf

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts = len(self.kpts)
        grad_kpts = [uhf.get_grad(mo_coeff_kpts[:,k],
                                            mo_occ_kpts[:,k], fock[:,k])
                     for k in range(nkpts)]
        return np.hstack(grad_kpts)

    def eig(self, h_kpts, s_kpts):
        e_a, c_a = khf.KRHF.eig(self, h_kpts[0], s_kpts)
        e_b, c_b = khf.KRHF.eig(self, h_kpts[1], s_kpts)
        return lib.asarray((e_a,e_b)), lib.asarray((c_a,c_b))

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        if mo_coeff_kpts is None: mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts)

    def get_bands(self, kpt_band, cell=None, dm_kpts=None, kpts=None):
        '''Get energy bands at a given (arbitrary) 'band' k-point.

        Returns:
            mo_energy : (nao,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nao) ndarray
                Band orbitals psi_n(k)
        '''
        if cell is None: cell = self.cell
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        if kpts is None: kpts = self.kpts

        fock = self.get_hcore(cell, kpt_band)
        fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpt_band=kpt_band)
        s1e = self.get_ovlp(cell, kpt_band)
        mo_energy, mo_coeff = uhf.eig(fock, s1e)
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=True, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)
    def from_chk(self, chk=None, project=True, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        uhf.UHF.dump_chk(self, envs)
        if self.chkfile:
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

    canonicalize = canonicalize

