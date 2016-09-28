#!/usr/bin/env python
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

import time
import numpy as np
import h5py
from pyscf.pbc.scf import hf as pbchf
from pyscf import lib
from pyscf.scf import hf
from pyscf.lib import logger
from pyscf.pbc.gto import ecp
from pyscf.pbc.scf import addons
from pyscf.pbc.scf import chkfile


def get_ovlp(mf, cell=None, kpts=None):
    '''Get the overlap AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        ovlp_kpts : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    return lib.asarray(cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts))


def get_hcore(mf, cell=None, kpts=None):
    '''Get the core Hamiltonian AO matrices at sampled k-points.

    Args:
        kpts : (nkpts, 3) ndarray

    Returns:
        hcore : (nkpts, nao, nao) ndarray
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    return lib.asarray([pbchf.get_hcore(cell, k) for k in kpts])


def get_j(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    from pyscf.pbc import df
    return df.DF(cell).get_jk(dm_kpts, kpts, kpt_band, with_k=False)[0]


def get_jk(mf, cell, dm_kpts, kpts, kpt_band=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpt_band : (3,) ndarray
            An arbitrary "band" k-point at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    from pyscf.pbc import df
    return df.DF(cell).get_jk(dm_kpts, kpts, kpt_band, with_j=False,
                              exxdiv=mf.exxdiv)[0]

def get_fock(mf, h1e_kpts, s_kpts, vhf_kpts, dm_kpts, cycle=-1, adiis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    f_kpts = h1e_kpts + vhf_kpts
    if adiis and cycle >= diis_start_cycle:
        f_kpts = adiis.update(s_kpts, dm_kpts, f_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts = [hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                  for k, s in enumerate(s_kpts)]
    return lib.asarray(f_kpts)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''
    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
    mo_occ_kpts = np.zeros_like(mo_energy_kpts)

    nkpts = mo_energy_kpts.shape[0]
    nocc = (mf.cell.nelectron * nkpts) // 2

    # TODO: implement Fermi smearing and print mo_energy kpt by kpt
    mo_energy = np.sort(mo_energy_kpts.ravel())
    fermi = mo_energy[nocc-1]
    mo_occ_kpts[mo_energy_kpts <= fermi] = 2

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
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.kpts):
            kstr = '(%6.3f %6.3f %6.3f)' % tuple(mf.cell.get_scaled_kpts(kpt))
            logger.debug(mf, '  %2d %s   %s %s', k, kstr,
                         mo_energy_kpts[k,mo_occ_kpts[k]> 0],
                         mo_energy_kpts[k,mo_occ_kpts[k]==0])
        np.set_printoptions()

    return mo_occ_kpts


def make_rdm1(mo_coeff_kpts, mo_occ_kpts):
    '''One particle density matrices for all k-points.

    Returns:
        dm_kpts : (nkpts, nao, nao) ndarray
    '''
    nkpts = len(mo_occ_kpts)
    dm_kpts = [hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k])
               for k in range(nkpts)]
    return lib.asarray(dm_kpts)


def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''Following pyscf.scf.hf.energy_elec()
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    nkpts = len(dm_kpts)
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts, h1e_kpts)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
    if abs(e_coul.imag > 1.e-7):
        raise RuntimeError("Coulomb energy has imaginary part, "
                           "something is wrong!", e_coul.imag)
    e1 = e1.real
    e_coul = e_coul.real
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul


def init_guess_by_chkfile(cell, chkfile_name, project=True, kpts=None):
    '''Read the KHF results from checkpoint file, then project it to the
    basis defined by ``cell``

    Returns:
        Density matrix, 3D ndarray
    '''
    chk_cell, scf_rec = chkfile.load_scf(chkfile_name)

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
            mos = [fproj(mo, None) for mo in mos]
            return make_rdm1(mos, occs)
    else:
        where = [np.argmin(lib.norm(chk_kpts-kpt, axis=1)) for kpt in kpts]
        def makedm(mos, occs):
            mos = [fproj(mos[w], chk_kpts[w]-kpts[i]) for i,w in enumerate(where)]
            return make_rdm1(mos, occs[where])

    if mo.ndim == 3:  # KRHF
        dm = makedm(mo, mo_occ)
    else:  # KUHF
        dm = makedm(mo[0], mo_occ[0]) + makedm(mo[1], mo_occ[1])

    # Real DM for gamma point
    if np.allclose(kpts, 0):
        dm = dm.real
    return dm


class KRHF(hf.RHF):
    '''RHF class with k-point sampling.

    Compared to molecular SCF, some members such as mo_coeff, mo_occ
    now have an additional first dimension for the k-points,
    e.g. mo_coeff is (nkpts, nao, nao) ndarray

    Attributes:
        kpts : (nks,3) ndarray
            The sampling k-points in Cartesian coordinates, in units of 1/Bohr.
    '''
    def __init__(self, cell, kpts=np.zeros((1,3)), exxdiv='ewald'):
        from pyscf.pbc import df
        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()
        self.cell = cell
        hf.RHF.__init__(self, cell)

        self.with_df = df.DF(cell)
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

    @property
    def mo_energy_kpts(self):
        return self.mo_energy

    @property
    def mo_coeff_kpts(self):
        return self.mo_coeff

    @property
    def mo_occ_kpts(self):
        return self.mo_occ

    def dump_flags(self):
        hf.RHF.dump_flags(self)
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
        hf.RHF.build(self, cell)
        #if self.exxdiv == 'vcut_ws':
        #    self.precompute_exx()

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None: cell = self.cell
        dm = hf.RHF.get_init_guess(self, cell, key)
        if key.lower() == 'chkfile':
            dm_kpts = dm
        else:
            dm_kpts = lib.asarray([dm]*len(self.kpts))
        return dm_kpts

    def get_hcore(self, cell=None, kpts=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if cell.pseudo is None:
            nuc = self.with_df.get_nuc(kpts)
        else:
            nuc = self.with_df.get_pp(kpts)
            if cell._ecp:
                nuc += ecp.int_ecp(cell, kpts)
        t = cell.pbc_intor('cint1e_kin_sph', 1, 1, kpts)
        return lib.asarray(nuc) + lib.asarray(t)

    get_ovlp = get_ovlp
    get_fock = get_fock
    get_occ = get_occ
    energy_elec = energy_elec

    def get_j(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpt_band=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj = self.with_df.get_jk(dm_kpts, hermi, kpts, kpt_band,
                                 with_k=False)[0]
        logger.timer(self, 'vj', *cpu0)
        return vj

    def get_k(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpt_band=None):
        return self.get_jk(cell, dm_kpts, hermi, kpts, kpt_band)[1]

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpt_band=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpt_band,
                                     exxdiv=self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpt_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpt_band)
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

        nkpts = len(self.kpts)
        grad_kpts = [hf.get_grad(mo_coeff_kpts[k], mo_occ_kpts[k], fock[k])
                     for k in range(nkpts)]
        return np.hstack(grad_kpts)

    def eig(self, h_kpts, s_kpts):
        nkpts = len(h_kpts)
        eig_kpts = []
        mo_coeff_kpts = []

        for k in range(nkpts):
            e, c = hf.RHF.eig(self, h_kpts[k], s_kpts[k])
            eig_kpts.append(e)
            mo_coeff_kpts.append(c)
        return lib.asarray(eig_kpts), lib.asarray(mo_coeff_kpts)

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

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
        mo_energy, mo_coeff = hf.eig(fock, s1e)
        return mo_energy, mo_coeff

    def init_guess_by_chkfile(self, chk=None, project=True, kpts=None):
        if chk is None: chk = self.chkfile
        if kpts is None: kpts = self.kpts
        return init_guess_by_chkfile(self.cell, chk, project, kpts)
    def from_chk(self, chk=None, project=True, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        hf.RHF.dump_chk(self, envs)
        if self.chkfile:
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

