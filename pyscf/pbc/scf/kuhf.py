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
from pyscf.pbc.scf import chkfile
from functools import reduce


canonical_occ = canonical_occ_ = addons.canonical_occ_


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

def get_fock(mf, h1e_kpts, s_kpts, vhf_kpts, dm_kpts, cycle=-1, diis=None,
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
    if diis and cycle >= diis_start_cycle:
        f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, mf, h1e_kpts, vhf_kpts)
    if abs(level_shift_factor) > 1e-4:
        f_kpts =([hf.level_shift(s, dm_kpts[0,k], f_kpts[0,k], shifta)
                  for k, s in enumerate(s_kpts)],
                 [hf.level_shift(s, dm_kpts[1,k], f_kpts[1,k], shiftb)
                  for k, s in enumerate(s_kpts)])
    return lib.asarray(f_kpts)

def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    '''Label the occupancies for each orbital for sampled k-points.

    This is a k-point version of scf.hf.SCF.get_occ
    '''

    if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

    nkpts = len(mo_energy_kpts[0])

    nocc_a = mf.nelec[0] * nkpts
    mo_energy = np.sort(np.hstack(mo_energy_kpts[0]))
    fermi_a = mo_energy[nocc_a-1]
    mo_occ_kpts = [[], []]
    for mo_e in mo_energy_kpts[0]:
        mo_occ_kpts[0].append((mo_e <= fermi_a).astype(np.double))
    if nocc_a < len(mo_energy):
        logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', fermi_a, mo_energy[nocc_a])
    else:
        logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', fermi_a)

    if mf.nelec[1] > 0:
        nocc_b = mf.nelec[1] * nkpts
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
    if abs(e_coul.imag > 1.e-10):
        raise RuntimeError("Coulomb energy has imaginary part, "
                           "something is wrong!", e_coul.imag)
    e1 = e1.real
    e_coul = e_coul.real
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul


def mulliken_meta(cell, dm_ao, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if s is None:
        s = khf.get_ovlp(cell)
    log = logger.new_logger(cell, verbose)
    log.note('Analyze output for the gamma point')
    log.note("KUHF mulliken_meta")
    dm_ao_gamma = dm_ao[:,0,:,:].real
    s_gamma = s[0,:,:].real
    c = orth.restore_ao_character(cell, pre_orth_method)
    orth_coeff = orth.orth_ao(cell, 'meta_lowdin', pre_orth_ao=c, s=s_gamma)
    c_inv = np.dot(orth_coeff.T, s_gamma)
    dm_a = reduce(np.dot, (c_inv, dm_ao_gamma[0], c_inv.T.conj()))
    dm_b = reduce(np.dot, (c_inv, dm_ao_gamma[1], c_inv.T.conj()))

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return uhf.mulliken_pop(cell, (dm_a,dm_b), np.eye(orth_coeff.shape[0]), log)


def canonicalize(mf, mo_coeff_kpts, mo_occ_kpts, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix within occupied,
    virtual subspaces separatedly (without change occupancy).
    '''
    mo_coeff_kpts = np.asarray(mo_coeff_kpts)
    mo_occ_kpts = np.asarray(mo_occ_kpts)
    if fock is None:
        dm = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
        fock = mf.get_hcore() + mf.get_jk(mf.cell, dm)

    def eig_(fock, mo_coeff_kpts, idx, es, cs):
        if np.count_nonzero(idx) > 0:
            orb = mo_coeff_kpts[:,idx]
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
            mo = np.expand_dims(mo, axis=0)
            mo_occ = np.expand_dims(mo_occ, axis=0)
        else:  # UHF
            mo = [np.expand_dims(mo[0], axis=0),
                  np.expand_dims(mo[1], axis=0)]
            mo_occ = [np.expand_dims(mo_occ[0], axis=0),
                      np.expand_dims(mo_occ[1], axis=0)]

    def fproj(mo, kpts):
        if project:
            return addons.project_mo_nr2nr(chk_cell, mo, cell, kpts)
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
            dkpts = [chk_kpts[w]-kpts[i] for i,w in enumerate(where)]
            mos = (fproj([moa[w] for w in where], dkpts),
                   fproj([mob[w] for w in where], dkpts))
            occs = ([occa[i] for i in where], [occb[i] for i in where])
            return make_rdm1(mos, occs)

    if mo[0].ndim == 2:  # KRHF
        mo_occa = [(occ>1e-8).astype(np.double) for occ in mo_occ]
        mo_occb = [occ-mo_occa[k] for k,occ in enumerate(mo_occ)]
        dm = makedm((mo, mo), (mo_occa, mo_occb))
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
        khf.KRHF.dump_flags(self)
        return self

    def build(self, cell=None):
        uhf.UHF.build(self, cell)
        #if self.exxdiv == 'vcut_ws':
        #    self.precompute_exx()

    def get_init_guess(self, cell=None, key='minao'):
        if cell is None:
            cell = self.cell
        dm_kpts = None
        if key.lower() == '1e':
            dm = self.init_guess_by_1e(cell)
        elif getattr(cell, 'natm', 0) == 0:
            logger.info(self, 'No atom found in cell. Use 1e initial guess')
            dm = self.init_guess_by_1e(cell)
        elif key.lower() == 'atom':
            dm = self.init_guess_by_atom(cell)
        elif key.lower().startswith('chk'):
            try:
                dm_kpts = self.from_chk()
            except (IOError, KeyError):
                logger.warn(self, 'Fail in reading %s. Use MINAO initial guess',
                            self.chkfile)
                dm = self.init_guess_by_minao(cell)
        else:
            dm = self.init_guess_by_minao(cell)

        if dm_kpts is None:
            assert(dm.ndim == 3)
            nao = dm.shape[-1]
            nkpts = len(self.kpts)
            dm_kpts = lib.asarray([dm]*nkpts).reshape(nkpts,2,nao,nao)
            dm_kpts = dm_kpts.transpose(1,0,2,3)
            dm_kpts[0,:] *= 1.01
            dm_kpts[1,:] *= 0.99  # To break spin symmetry
            assert dm_kpts.shape[0]==2

        if cell.dimension < 3:
            ne = np.einsum('xkij,kji->k', dm_kpts, self.get_ovlp(cell))
            if np.any(abs(ne - cell.nelectron).sum() > 1e-7):
                logger.warn(self, 'Big error detected in the electron number '
                            'of initial guess density matrix (Ne/cell = %g)!\n'
                            '  This can cause huge error in Fock matrix and '
                            'lead to instability in SCF for low-dimensional '
                            'systems.\n  DM is normalized to correct number '
                            'of electrons', ne.mean())
                dm_kpts *= cell.nelectron / ne.reshape(2,-1,1,1)
        return dm_kpts

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return uhf.UHF.init_guess_by_1e(cell)

    get_hcore = khf.KRHF.get_hcore
    get_ovlp = khf.KRHF.get_ovlp
    get_jk = khf.KRHF.get_jk
    get_j = khf.KRHF.get_j
    get_k = khf.KRHF.get_k

    get_fock = get_fock
    get_occ = get_occ
    energy_elec = energy_elec

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = uhf._makevhf(vj, vk)
        return vhf


    def analyze(self, verbose=None, **kwargs):
        if verbose is None: verbose = self.verbose
        return khf.analyze(self, verbose, **kwargs)


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
        e_a, c_a = khf.KRHF.eig(self, h_kpts[0], s_kpts)
        e_b, c_b = khf.KRHF.eig(self, h_kpts[1], s_kpts)
        return (e_a,e_b), (c_a,c_b)

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        if mo_coeff_kpts is None: mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None: mo_occ_kpts = self.mo_occ
        return make_rdm1(mo_coeff_kpts, mo_occ_kpts)

    def get_bands(self, kpts_band, cell=None, dm_kpts=None, kpts=None):
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

        kpts_band = np.asarray(kpts_band)
        single_kpt_band = (kpts_band.ndim == 1)
        kpts_band = kpts_band.reshape(-1,3)

        fock = self.get_hcore(cell, kpts_band)
        fock = fock + self.get_veff(cell, dm_kpts, kpts=kpts, kpts_band=kpts_band)
        s1e = self.get_ovlp(cell, kpts_band)
        e_a, c_a = khf.KRHF.eig(self, fock[0], s1e)
        e_b, c_b = khf.KRHF.eig(self, fock[1], s1e)
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
    def from_chk(self, chk=None, project=True, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        uhf.UHF.dump_chk(self, envs)
        if self.chkfile:
            with h5py.File(self.chkfile) as fh5:
                fh5['scf/kpts'] = self.kpts
        return self

    def mulliken_meta(self, cell=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method='ANO', s=None):
        if cell is None: cell = self.cell
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(cell)
        return mulliken_meta(cell, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    @lib.with_doc(uhf.spin_square.__doc__)
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

    def density_fit(self, auxbasis=None, gs=None):
        return khf.KRHF.density_fit(self, auxbasis, gs)

    # mix_density_fit inherits from khf.KRHF.mix_density_fit


if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = '321g'
    cell.a = np.eye(3) * 3
    cell.gs = [5] * 3
    cell.verbose = 5
    cell.build()
    mf = KUHF(cell, [2,1,1])
    mf.kernel()
    mf.analyze()

