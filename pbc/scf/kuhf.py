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


def canonical_occ_(mf):
    '''Label the occupancies for each orbital for sampled k-points.
    This is for KUHF objects. 
    Each k-point has a fixed number of up and down electrons in this, 
    which results in a finite size error for metallic systems
    but can accelerate convergence '''
    assert(isinstance(mf,KUHF))

    def get_occ(mo_energy_kpts=None,mo_coeff=None):
        if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy
        #print(mo_energy_kpts)
        #mo_occ_kpts = [[],[]] #np.zeros_like(np.array(mo_energy_kpts))
        mo_occ_kpts=np.zeros_like(mo_energy_kpts)
        print("shape",mo_occ_kpts.shape)

        nkpts = np.array(mo_energy_kpts).shape[1]
        homo=[-1e8,-1e8]
        lumo=[1e8,1e8]

        for k in range(nkpts):
            for s in [0,1]:
                occ=np.zeros_like(mo_energy_kpts[s,k])
                e_idx=np.argsort(mo_energy_kpts[s,k])
                e_sort=mo_energy_kpts[s,k][e_idx]
                n=mf.nelec[s]
                mo_occ_kpts[s,k][e_idx[:n]]=1
                homo[s]=max(homo[s],e_sort[n-1])
                lumo[s]=min(lumo[s],e_sort[n])

        for nm,s in zip(['alpha','beta'],[0,1]):
            logger.info(mf, nm+' HOMO = %.12g  LUMO = %.12g',
                    homo[s],lumo[s])
            if homo[s] > lumo[s]:
                logger.info(mf,"WARNING! HOMO is greater than LUMO! This may result in errors with canonical occupation.")

        return mo_occ_kpts

    mf.get_occ=get_occ
    return mf
canonical_occ=canonical_occ_



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
    mo_occ_kpts = np.zeros_like(mo_energy_kpts)

    nkpts = len(mo_energy_kpts[0])

    nocc_a = mf.nelec[0] * nkpts
    mo_energy = np.sort(mo_energy_kpts[0].ravel())
    fermi_a = mo_energy[nocc_a-1]
    mo_occ_kpts[0,mo_energy_kpts[0]<=fermi_a] = 1
    if nocc_a < len(mo_energy):
      logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', fermi_a, mo_energy[nocc_a])    
    else:
       logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', fermi_a)

    nocc_b = mf.nelec[1] * nkpts
    mo_energy = np.sort(mo_energy_kpts[1].ravel())
    fermi_b = mo_energy[nocc_b-1]
    mo_occ_kpts[1,mo_energy_kpts[1]<=fermi_b] = 1
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
                         mo_energy_kpts[0,k,mo_occ_kpts[0,k]> 0],
                         mo_energy_kpts[0,k,mo_occ_kpts[0,k]==0])
        logger.debug(mf, '     k-point                  beta  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                         k, kpt[0], kpt[1], kpt[2],
                         mo_energy_kpts[1,k,mo_occ_kpts[1,k]> 0],
                         mo_energy_kpts[1,k,mo_occ_kpts[1,k]==0])
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

def analyze(mf, verbose=logger.DEBUG, **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, verbose)
    log.note('Analyze output for the gamma point')
    #log.note('**** MO energy ****')
    #log.note('                             alpha | beta                alpha | beta')
    #for i in range(mo_occ.shape[-1]):
    #   log.note('MO #%-3d energy= %-18.15g | %-18.15g occ= %g | %g',
    #             i+1, mo_energy[0][i], mo_energy[1][i],
    #             mo_occ[0][i], mo_occ[1][i])
    ovlp_ao = mf.get_ovlp()
    #if verbose >= logger.DEBUG:
    #    log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) for alpha spin **')
    #    label = mf.mol.spheric_labels(True)
    #    orth_coeff = orth.orth_ao(mf.mol, 'meta_lowdin', s=ovlp_ao)
    #    c_inv = numpy.dot(orth_coeff.T, ovlp_ao)
    #    dump_mat.dump_rec(mf.stdout, c_inv.dot(mo_coeff[0]), label, start=1,
    #                      **kwargs)
    #    log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) for beta spin **')
    #    dump_mat.dump_rec(mf.stdout, c_inv.dot(mo_coeff[1]), label, start=1,
    #                      **kwargs)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return (mf.mulliken_meta(mf.mol, dm, s=ovlp_ao, verbose=log))
#            mf.dip_moment(mf.mol, dm, verbose=log))


def mulliken_meta(mol, dm_ao, verbose=logger.DEBUG, pre_orth_method='ANO',
                  s=None):
    '''Mulliken population analysis, based on meta-Lowdin AOs.
    '''
    from pyscf.lo import orth
    if s is None:
        s = hf.get_ovlp(mol)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    log.note("KUHF mulliken_meta")
    dm_ao_gamma=dm_ao[:,0,:,:].real.copy()
    s_gamma=s[0,:,:].real.copy()
    c = orth.pre_orth_ao(mol, pre_orth_method)
    orth_coeff = orth.orth_ao(mol, 'meta_lowdin', pre_orth_ao=c, s=s_gamma)
    c_inv = np.dot(orth_coeff.T, s_gamma)
    dm_a = reduce(np.dot, (c_inv, dm_ao_gamma[0], c_inv.T.conj()))
    dm_b = reduce(np.dot, (c_inv, dm_ao_gamma[1], c_inv.T.conj()))

    log.note(' ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **')
    return uhf.mulliken_pop(mol, (dm_a,dm_b), np.eye(orth_coeff.shape[0]), log)


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
    return mo_e, mo_coeff_kpts

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
            nao = dm.shape[-1]
            nkpts = len(self.kpts)
            if len(dm.shape)==3:
                dm_kpts = lib.asarray([dm]*nkpts).reshape(nkpts,2,nao,nao)
                dm_kpts = dm_kpts.transpose(1,0,2,3)
            else:
                dm_kpts=dm
            dm_kpts[1,:] *= .98  # To break spin symmetry
            assert dm_kpts.shape[0]==2
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
                 kpts=None, kpts_band=None):
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        vhf = uhf._makevhf(vj, vk)
        return vhf


    def analyze(self, verbose=None, **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, **kwargs)


    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore(self.cell, self.kpts) + self.get_veff(self.cell, dm1)

        nkpts = len(self.kpts)
        grad_kpts = [uhf.get_grad(mo_coeff_kpts[:,k], mo_occ_kpts[:,k], fock[:,k])
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
        return lib.asarray((e_a,e_b)), lib.asarray((c_a,c_b))

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

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method='ANO', s=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return mulliken_meta(mol, dm, s=s, verbose=verbose,
                             pre_orth_method=pre_orth_method)

    @lib.with_doc(uhf.spin_square.__doc__)
    def spin_square(self, mo_coeff=None, s=None):
        '''Treating the k-point sampling wfn as a giant Slater determinant,
        the spin_square value is the <S^2> of the giant determinant.
        '''
        nkpts = len(self.kpts)
        if mo_coeff is None:
            mo_a = [self.mo_coeff[0,k][:,self.mo_occ[0,k]>0] for k in range(nkpts)]
            mo_b = [self.mo_coeff[1,k][:,self.mo_occ[1,k]>0] for k in range(nkpts)]
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

