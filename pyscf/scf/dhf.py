#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Dirac Hartree-Fock
'''

import time
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import _vhf
from pyscf.scf import chkfile


def kernel(mf, conv_tol=1e-9, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True):
    '''the modified SCF kernel for Dirac-Hartree-Fock.  In this kernel, the
    SCF is carried out in three steps.  First the 2-electron part is
    approximated by large component integrals (LL|LL); Next, (SS|LL) the
    interaction between large and small components are added; Finally,
    converge the SCF with the small component contributions (SS|SS)
    '''
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)
    if dm0 is None:
        dm = mf.get_init_guess()
    else:
        dm = dm0

    mf._coulomb_now = 'LLLL'
    if dm0 is None and mf._coulomb_now.upper() == 'LLLL':
        scf_conv, e_tot, mo_energy, mo_coeff, mo_occ \
                = hf.kernel(mf, 1e-2, 1e-1,
                            dump_chk, dm0=dm, callback=callback,
                            conv_check=False)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        mf._coulomb_now = 'SSLL'

    if dm0 is None and (mf._coulomb_now.upper() == 'SSLL' or
                        mf._coulomb_now.upper() == 'LLSS'):
        scf_conv, e_tot, mo_energy, mo_coeff, mo_occ \
                = hf.kernel(mf, 1e-3, 1e-1,
                            dump_chk, dm0=dm, callback=callback,
                            conv_check=False)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        mf._coulomb_now = 'SSSS'

    if mf.with_ssss:
        mf._coulomb_now = 'SSSS'
    else:
        mf._coulomb_now = 'SSLL'

    return hf.kernel(mf, conv_tol, conv_tol_grad, dump_chk, dm0=dm,
                     callback=callback, conv_check=conv_check)

def get_jk_coulomb(mol, dm, hermi=1, coulomb_allow='SSSS',
                   opt_llll=None, opt_ssll=None, opt_ssss=None):
    if coulomb_allow.upper() == 'LLLL':
        logger.info(mol, 'Coulomb integral: (LL|LL)')
        j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
        n2c = j1.shape[1]
        vj = numpy.zeros_like(dm)
        vk = numpy.zeros_like(dm)
        vj[...,:n2c,:n2c] = j1
        vk[...,:n2c,:n2c] = k1
    elif coulomb_allow.upper() == 'SSLL' \
      or coulomb_allow.upper() == 'LLSS':
        logger.info(mol, 'Coulomb integral: (LL|LL) + (SS|LL)')
        vj, vk = _call_veff_ssll(mol, dm, hermi, opt_ssll)
        j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
        n2c = j1.shape[1]
        vj[...,:n2c,:n2c] += j1
        vk[...,:n2c,:n2c] += k1
    else: # coulomb_allow == 'SSSS'
        logger.info(mol, 'Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
        vj, vk = _call_veff_ssll(mol, dm, hermi, opt_ssll)
        j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
        n2c = j1.shape[1]
        vj[...,:n2c,:n2c] += j1
        vk[...,:n2c,:n2c] += k1
        j1, k1 = _call_veff_ssss(mol, dm, hermi, opt_ssss)
        vj[...,n2c:,n2c:] += j1
        vk[...,n2c:,n2c:] += k1
    return vj, vk

def get_jk(mol, dm, hermi=1, coulomb_allow='SSSS'):
    return get_jk_coulomb(mol, dm, hermi=hermi, coulomb_allow=coulomb_allow)


def get_hcore(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    t  = mol.intor_symmetric('int1e_spsp_spinor') * .5
    vn = mol.intor_symmetric('int1e_nuc_spinor')
    wn = mol.intor_symmetric('int1e_spnucsp_spinor')
    h1e = numpy.empty((n4c, n4c), numpy.complex)
    h1e[:n2c,:n2c] = vn
    h1e[n2c:,:n2c] = t
    h1e[:n2c,n2c:] = t
    h1e[n2c:,n2c:] = wn * (.25/c**2) - t
    return h1e

def get_ovlp(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    s = mol.intor_symmetric('int1e_ovlp_spinor')
    t = mol.intor_symmetric('int1e_spsp_spinor')
    s1e = numpy.zeros((n4c, n4c), numpy.complex)
    s1e[:n2c,:n2c] = s
    s1e[n2c:,n2c:] = t * (.5/c)**2
    return s1e

def make_rdm1(mo_coeff, mo_occ):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    dm = hf.init_guess_by_minao(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    mf = UHF(mol)
    return mf.init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_chkfile(mol, chkfile_name, project=True):
    from pyscf.scf import addons
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if numpy.iscomplexobj(mo[0]):  # DHF
#TODO: check if mo is GHF orbital
        if project:
            dm = make_rdm1(addons.project_mo_r2r(chk_mol, mo, mol), mo_occ)
        else:
            dm = make_rdm1(mo, mo_occ)
    else:
        if mo[0].ndim == 1: # nr-RHF
            dm = reduce(numpy.dot, (mo*mo_occ, mo.T))
        else: # nr-UHF
            dm = reduce(numpy.dot, (mo[0]*mo_occ[0], mo[0].T)) \
               + reduce(numpy.dot, (mo[1]*mo_occ[1], mo[1].T))
        dm = _proj_dmll(chk_mol, dm, mol)
    return dm

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    elif key.lower() == '1e':
        return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.hf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)

def time_reversal_matrix(mol, mat):
    ''' T(A_ij) = A[T(i),T(j)]^*
    '''
    n2c = mol.nao_2c()
    tao = numpy.asarray(mol.time_reversal_map())
    # tao(i) = -j  means  T(f_i) = -f_j
    # tao(i) =  j  means  T(f_i) =  f_j
    idx = abs(tao)-1 # -1 for C indexing convention
    #:signL = [(1 if x>0 else -1) for x in tao]
    #:sign = numpy.hstack((signL, signL))

    #:tmat = numpy.empty_like(mat)
    #:for j in range(mat.__len__()):
    #:    for i in range(mat.__len__()):
    #:        tmat[idx[i],idx[j]] = mat[i,j] * sign[i]*sign[j]
    #:return tmat.conjugate()
    sign_mask = tao<0
    if mat.shape[0] == n2c*2:
        idx = numpy.hstack((idx, idx+n2c))
        sign_mask = numpy.hstack((sign_mask, sign_mask))

    tmat = mat.take(idx,axis=0).take(idx,axis=1)
    tmat[sign_mask,:] *= -1
    tmat[:,sign_mask] *= -1
    return tmat.T

def analyze(mf, verbose=logger.DEBUG, **kwargs):
    #from pyscf.tools import dump_mat
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mf.stdout, verbose)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    #mo_coeff = mf.mo_coeff

    log.info('**** MO energy ****')
    for i in range(len(mo_energy)):
        if mo_occ[i] > 0:
            log.info('occupied MO #%d energy= %.15g occ= %g', \
                     i+1, mo_energy[i], mo_occ[i])
        else:
            log.info('virtual MO #%d energy= %.15g occ= %g', \
                     i+1, mo_energy[i], mo_occ[i])
#TODO    if mf.verbose >= logger.DEBUG:
#TODO        log.debug(' ** MO coefficients **')
#TODO        label = mf.mol.spinor_labels(True)
#TODO        dump_mat.dump_rec(mf.stdout, mo_coeff, label, start=1)
#TODO    dm = mf.make_rdm1(mo_coeff, mo_occ)
#TODO    return mf.mulliken_pop(mf.mol, dm, mf.get_ovlp(), log)

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''DHF Gradients'''
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(numpy.dot, (mo_coeff[:,viridx].T.conj(), fock_ao,
                           mo_coeff[:,occidx]))
    return g.ravel()


class UHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for Dirac-Hartree-Fock
        with_ssss : bool, for Dirac-Hartree-Fock only
            If False, ignore small component integrals (SS|SS).  Default is True.
        with_gaunt : bool, for Dirac-Hartree-Fock only
            Default is False.
        with_breit : bool, for Dirac-Hartree-Fock only
            Gaunt + gauge term.  Default is False.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> e0 = mf.scf()
    >>> mf = scf.DHF(mol)
    >>> e1 = mf.scf()
    >>> print('Relativistic effects = %.12f' % (e1-e0))
    Relativistic effects = -0.000008854205
    '''
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.conv_tol = 1e-8
        self.with_ssss = True
        self._coulomb_now = 'SSSS' # 'SSSS' ~ LLLL+LLSS+SSSS
        self.with_gaunt = False
        self.with_breit = False

        self.opt = (None, None, None, None) # (opt_llll, opt_ssll, opt_ssss, opt_gaunt)
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        logger.info(self, 'with_ssss %s, with_gaunt %s, with_breit %s',
                    self.with_ssss, self.with_gaunt, self.with_breit)
        logger.info(self, 'light speed = %s', lib.param.LIGHT_SPEED)
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_chkfile(self, chkfile=None, project=True):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.direct_scf:
            self.opt = self.init_direct_scf(self.mol)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        c = lib.param.LIGHT_SPEED
        n4c = len(mo_energy)
        n2c = n4c // 2
        mo_occ = numpy.zeros(n2c * 2)
        if mo_energy[n2c] > -1.999 * c**2:
            mo_occ[n2c:n2c+mol.nelectron] = 1
        else:
            n = 0
            for i, e in enumerate(mo_energy):
                if e > -1.999 * c**2 and n < mol.nelectron:
                    mo_occ[i] = 1
                    n += 1
        if self.verbose >= logger.INFO:
            logger.info(self, 'HOMO %d = %.12g  LUMO %d = %.12g',
                        n2c+mol.nelectron, mo_energy[n2c+mol.nelectron-1],
                        n2c+mol.nelectron+1, mo_energy[n2c+mol.nelectron])
            logger.debug(self, 'NES  mo_energy = %s', mo_energy[:n2c])
            logger.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])
        return mo_occ

    # full density matrix for UHF
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ)

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        def set_vkscreen(opt, name):
            opt._this.contents.r_vkscreen = _vhf._fpointer(name)
        opt_llll = _vhf.VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                               'CVHFrkbllll_direct_scf',
                               'CVHFrkbllll_direct_scf_dm')
        opt_llll.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt_llll, 'CVHFrkbllll_vkscreen')
        opt_ssss = _vhf.VHFOpt(mol, 'int2e_spsp1spsp2_spinor',
                               'CVHFrkbllll_prescreen',
                               'CVHFrkbssss_direct_scf',
                               'CVHFrkbssss_direct_scf_dm')
        opt_ssss.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt_ssss, 'CVHFrkbllll_vkscreen')
        opt_ssll = _vhf.VHFOpt(mol, 'int2e_spsp1_spinor',
                               'CVHFrkbssll_prescreen',
                               'CVHFrkbssll_direct_scf',
                               'CVHFrkbssll_direct_scf_dm')
        opt_ssll.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt_ssll, 'CVHFrkbssll_vkscreen')
#TODO: prescreen for gaunt
        opt_gaunt = None
        return opt_llll, opt_ssll, opt_ssss, opt_gaunt

    def get_jk(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        verbose_bak, mol.verbose = mol.verbose, self.verbose
        stdout_bak,  mol.stdout  = mol.stdout , self.stdout
        if self.direct_scf and self.opt[0] is None:
            self.opt = self.init_direct_scf(mol)
        opt_llll, opt_ssll, opt_ssss, opt_gaunt = self.opt

        vj, vk = get_jk_coulomb(mol, dm, hermi, self._coulomb_now,
                                opt_llll, opt_ssll, opt_ssss)

        if self.with_breit:
            if 'SSSS' in self._coulomb_now.upper() or not self.with_ssss:
                vj1, vk1 = _call_veff_gaunt_breit(mol, dm, hermi, opt_gaunt, True)
                logger.info(self, 'Add Breit term')
                vj += vj1
                vk += vk1
        elif self.with_gaunt and 'SS' in self._coulomb_now.upper():
            logger.info(self, 'Add Gaunt term')
            vj1, vk1 = _call_veff_gaunt_breit(mol, dm, hermi, opt_gaunt, False)
            vj += vj1
            vk += vk1

        mol.verbose = verbose_bak
        mol.stdout  = stdout_bak
        logger.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Dirac-Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())

        self.build()
        self.dump_flags()
        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ \
                = kernel(self, self.conv_tol, self.conv_tol_grad,
                         dm0=dm0, callback=self.callback,
                         conv_check=self.conv_check)

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose)

    def x2c(self):
        import pyscf.scf.x2c
        x2chf = pyscf.scf.x2c.UHF(self.mol)
        x2chf.__dict__.update(self.__dict__)
        return x2chf
DHF = UHF


class HF1e(UHF):
    def scf(self, *args):
        logger.info(self, '\n')
        logger.info(self, '******** 1 electron system ********')
        self.converged = True
        h1e = self.get_hcore(self.mol)
        s1e = self.get_ovlp(self.mol)
        self.mo_energy, self.mo_coeff = self.eig(h1e, s1e)
        self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
        self.e_tot = self.mo_energy[self.mo_occ>0][0] + self.mol.energy_nuc()
        self._finalize()
        return self.e_tot

class RHF(UHF):
    '''Dirac-RHF'''
    def __init__(self, mol):
        if mol.nelectron.__mod__(2) != 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
        UHF.__init__(self, mol)

    # full density matrix for RHF
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        r'''D/2 = \psi_i^\dag\psi_i = \psi_{Ti}^\dag\psi_{Ti}
        D(UHF) = \psi_i^\dag\psi_i + \psi_{Ti}^\dag\psi_{Ti}
        RHF average the density of spin up and spin down:
        D(RHF) = (D(UHF) + T[D(UHF)])/2
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm = make_rdm1(mo_coeff, mo_occ)
        return (dm + time_reversal_matrix(self.mol, dm)) * .5

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        c = lib.param.LIGHT_SPEED
        n4c = len(mo_energy)
        n2c = n4c // 2
        mo_occ = numpy.zeros(n2c * 2)
        if mo_energy[n2c] > -1.999 * c**2:
            mo_occ[n2c:n2c+mol.nelectron] = 1
        else:
            n = 0
            for i, e in enumerate(mo_energy):
                if e > -1.999 * c**2 and n < mol.nelectron:
                    mo_occ[i] = 1
                    n += 1
        if self.verbose >= logger.INFO:
            logger.info(self, 'HOMO %d = %.12g, LUMO %d = %.12g,',
                        (n2c+mol.nelectron)//2, mo_energy[n2c+mol.nelectron-1],
                        (n2c+mol.nelectron)//2+1, mo_energy[n2c+mol.nelectron])
            logger.debug(self, 'NES  mo_energy = %s', mo_energy[:n2c])
            logger.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])
        return mo_occ


def _jk_triu_(vj, vk, hermi):
    if hermi == 0:
        if vj.ndim == 2:
            vj = lib.hermi_triu(vj, 1)
        else:
            for i in range(vj.shape[0]):
                vj[i] = lib.hermi_triu(vj[i], 1)
    else:
        if vj.ndim == 2:
            vj = lib.hermi_triu(vj, hermi)
            vk = lib.hermi_triu(vk, hermi)
        else:
            for i in range(vj.shape[0]):
                vj[i] = lib.hermi_triu(vj[i], hermi)
                vk[i] = lib.hermi_triu(vk[i], hermi)
    return vj, vk


def _call_veff_llll(mol, dm, hermi=1, mf_opt=None):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] // 2
        dms = dm[:n2c,:n2c].copy()
    else:
        n2c = dm[0].shape[0] // 2
        dms = []
        for dmi in dm:
            dms.append(dmi[:n2c,:n2c].copy())
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    return _jk_triu_(vj, vk, hermi)

def _call_veff_ssll(mol, dm, hermi=1, mf_opt=None):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] // 2
        dmll = dm[:n2c,:n2c].copy()
        dmsl = dm[n2c:,:n2c].copy()
        dmss = dm[n2c:,n2c:].copy()
        dms = (dmll, dmss, dmsl)
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] // 2
        dms = [dmi[:n2c,:n2c].copy() for dmi in dm] \
            + [dmi[n2c:,n2c:].copy() for dmi in dm] \
            + [dmi[n2c:,:n2c].copy() for dmi in dm]
    jks = ('lk->s2ij',) * n_dm \
        + ('ji->s2kl',) * n_dm \
        + ('jk->s1il',) * n_dm
    c1 = .5 / lib.param.LIGHT_SPEED
    vx = _vhf.rdirect_bindm('int2e_spsp1_spinor', 's4', jks, dms, 1,
                            mol._atm, mol._bas, mol._env, mf_opt) * c1**2
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,n2c:,n2c:] = vx[      :n_dm  ,:,:]
    vj[:,:n2c,:n2c] = vx[n_dm  :n_dm*2,:,:]
    vk[:,n2c:,:n2c] = vx[n_dm*2:      ,:,:]
    if n_dm == 1:
        vj = vj.reshape(vj.shape[1:])
        vk = vk.reshape(vk.shape[1:])
    return _jk_triu_(vj, vk, hermi)

def _call_veff_ssss(mol, dm, hermi=1, mf_opt=None):
    c1 = .5 / lib.param.LIGHT_SPEED
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] // 2
        dms = dm[n2c:,n2c:].copy()
    else:
        n2c = dm[0].shape[0] // 2
        dms = []
        for dmi in dm:
            dms.append(dmi[n2c:,n2c:].copy())
    vj, vk = _vhf.rdirect_mapdm('int2e_spsp1spsp2_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, mf_opt) * c1**4
    return _jk_triu_(vj, vk, hermi)

def _call_veff_gaunt_breit(mol, dm, hermi=1, mf_opt=None, with_breit=False):
    if with_breit:
        intor_prefix = 'int2e_breit_'
    else:
        intor_prefix = 'int2e_'
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] // 2
        dmls = dm[:n2c,n2c:].copy()
        dmsl = dm[n2c:,:n2c].copy()
        dmll = dm[:n2c,:n2c].copy()
        dmss = dm[n2c:,n2c:].copy()
        dms = [dmsl, dmsl, dmls, dmll, dmss]
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] // 2
        dmll = [dmi[:n2c,:n2c].copy() for dmi in dm]
        dmls = [dmi[:n2c,n2c:].copy() for dmi in dm]
        dmsl = [dmi[n2c:,:n2c].copy() for dmi in dm]
        dmss = [dmi[n2c:,n2c:].copy() for dmi in dm]
        dms = dmsl + dmsl + dmls + dmll + dmss
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)

    jks = ('lk->s1ij',) * n_dm \
        + ('jk->s1il',) * n_dm
    vx = _vhf.rdirect_bindm(intor_prefix+'ssp1ssp2_spinor', 's1', jks, dms[:n_dm*2], 1,
                            mol._atm, mol._bas, mol._env, mf_opt)
    vj[:,:n2c,n2c:] = vx[:n_dm,:,:]
    vk[:,:n2c,n2c:] = vx[n_dm:,:,:]

    jks = ('lk->s1ij',) * n_dm \
        + ('li->s1kj',) * n_dm \
        + ('jk->s1il',) * n_dm
    vx = _vhf.rdirect_bindm(intor_prefix+'ssp1sps2_spinor', 's1', jks, dms[n_dm*2:], 1,
                            mol._atm, mol._bas, mol._env, mf_opt)
    vj[:,:n2c,n2c:]+= vx[      :n_dm  ,:,:]
    vk[:,n2c:,n2c:] = vx[n_dm  :n_dm*2,:,:]
    vk[:,:n2c,:n2c] = vx[n_dm*2:      ,:,:]

    if hermi == 1:
        vj[:,n2c:,:n2c] = vj[:,:n2c,n2c:].transpose(0,2,1).conj()
        vk[:,n2c:,:n2c] = vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    elif hermi == 2:
        vj[:,n2c:,:n2c] = -vj[:,:n2c,n2c:].transpose(0,2,1).conj()
        vk[:,n2c:,:n2c] = -vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    else:
        raise NotImplementedError
    if n_dm == 1:
        vj = vj.reshape(n2c*2,n2c*2)
        vk = vk.reshape(n2c*2,n2c*2)
    c1 = .5 / lib.param.LIGHT_SPEED
    if with_breit:
        return vj*c1**2, vk*c1**2
    else:
        return -vj*c1**2, -vk*c1**2

def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, 1, mol)

    n2c = proj.shape[0]
    n4c = n2c * 2
    dm = numpy.zeros((n4c,n4c), dtype=complex)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
    return dm


if __name__ == '__main__':
    import pyscf.gto
    mol = pyscf.gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dhf'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = {
        'He': [(0, 0, (1, 1)),
               (0, 0, (3, 1)),
               (1, 0, (1, 1)), ]}
    mol.build()

##############
# SCF result
    method = UHF(mol)
    energy = method.scf() #-2.38146942868
    print(energy)
    method.with_gaunt = True
    print(method.scf()) # -2.38138339005
    method.with_breit = True
    print(method.scf()) # -2.38138339005
