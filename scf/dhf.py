#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Dirac Hartree-Fock
'''

import ctypes
import time
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import hf
import addons
import diis
import chkfile
import _vhf

__doc__ = '''Options:
self.chkfile = '/dev/shm/...'
self.stdout = '...'
self.diis_space = 6
self.diis_start_cycle = 1
self.damp_factor = 1
self.level_shift_factor = 0
self.conv_threshold = 1e-10
self.max_cycle = 50
self.oob = 0                    # operator oriented basis level
                                # 1 sp|f> -> |f>
                                # 2 sp|f> -> sr|f>


self.init_guess = method        # method = one of 'atom', '1e', 'chkfile'
self.set_potential(method, oob) # method = one of 'coulomb', 'gaunt'
                                # oob = operator oriented basis level
                                #       1 sp|f> -> |f>
                                #       2 sp|f> -> sr|f>
self.with_ssss = False   # pass (SS|SS) integral
'''


class UHF(hf.SCF):
    __doc__ = 'Dirac-UHF'
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self.conv_threshold = 1e-8
        self.oob = 0
        self.with_ssss = True
        self.init_guess = 'minao'
        self._coulomb_now = 'LLLL' # 'SSSS' ~ LLLL+LLSS+SSSS
        self.with_gaunt = False

        self.opt_llll = None
        self.opt_ssll = None
        self.opt_ssss = None
        self._keys = set(self.__dict__.keys() + ['_keys'])

    def _init_guess_by_atom(self, mol=None):
        '''Initial guess from occupancy-averaged atomic NR-RHF'''
        if mol is None:
            mol = self.mol
        ehf, dm0 = hf.init_guess_by_atom(self, mol)

        s0 = mol.intor_symmetric('cint1e_ovlp_sph')
        ua, ub = symm.cg.real2spinor_whole(mol)
        s = numpy.dot(ua.T.conj(), s0) + numpy.dot(ub.T.conj(), s0) # (*)
        proj = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp'), s)

        n2c = ua.shape[1]
        n4c = n2c * 2
        dm = numpy.zeros((n4c,n4c), dtype=complex)
        # *.5 because alpha and beta are summed in Eq. (*)
        dm_ll = reduce(numpy.dot, (proj, dm0*.5, proj.T.conj()))
        dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
        return ehf, dm

    def _init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        from pyscf import symm
        if mol is None:
            mol = self.mol
        ehf, dm0 = hf.init_guess_by_minao(self, mol)
        proj = addons.project_mo_nr2r(mol, 1, mol)

        n2c = proj.shape[0]
        n4c = n2c * 2
        dm = numpy.zeros((n4c,n4c), dtype=complex)
        # *.5 because alpha and beta are summed in project_mo_nr2r
        dm_ll = reduce(numpy.dot, (proj, dm0*.5, proj.T.conj()))
        dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
        return ehf, dm

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        log.info(self, 'OOB = %d', self.oob)

    def init_diis(self):
        diis_a = diis.SCF_DIIS(self)
        diis_a.diis_space = self.diis_space
        #diis_a.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                f = diis_a.update(s, d, f)
            if cycle < self.diis_start_cycle-1:
                f = hf.damping(s, d, f, self.damp_factor)
                f = hf.level_shift(s, d, f, self.level_shift_factor)
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = hf.level_shift(s, d, f, fac)
            return f
        return scf_diis

    @lib.omnimethod
    def get_hcore(self, mol):
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        s  = mol.intor_symmetric('cint1e_ovlp')
        t  = mol.intor_symmetric('cint1e_spsp') * .5
        vn = mol.intor_symmetric('cint1e_nuc')
        wn = mol.intor_symmetric('cint1e_spnucsp')
        h1e = numpy.empty((n4c, n4c), numpy.complex)
        h1e[:n2c,:n2c] = vn
        h1e[n2c:,:n2c] = t
        h1e[:n2c,n2c:] = t
        h1e[n2c:,n2c:] = wn * (.25/c**2) - t
        return h1e

    @lib.omnimethod
    def get_ovlp(self, mol):
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        s = mol.intor_symmetric('cint1e_ovlp')
        t = mol.intor_symmetric('cint1e_spsp') * .5
        s1e = numpy.zeros((n4c, n4c), numpy.complex)
        s1e[:n2c,:n2c] = s
        s1e[n2c:,n2c:] = t * (.5/c**2)
        return s1e

    def build(self, mol=None):
        self.build_(mol)
    def build_(self, mol=None):
        if mol is None:
            mol = self.mol
        mol.check_sanity(self)

        if self.direct_scf:
            def set_vkscreen(opt, name):
                opt._this.contents.r_vkscreen = \
                    ctypes.c_void_p(_ctypes.dlsym(_vhf.libcvhf._handle, name))
            self.opt_llll = _vhf.VHFOpt(mol, 'cint2e', 'CVHFrkbllll_prescreen',
                                        'CVHFrkbllll_direct_scf',
                                        'CVHFrkbllll_direct_scf_dm')
            self.opt_llll.direct_scf_threshold = self.direct_scf_threshold
            set_vkscreen(self.opt_llll, 'CVHFrkbllll_vkscreen')
            self.opt_ssss = _vhf.VHFOpt(mol, 'cint2e_spsp1spsp2',
                                        'CVHFrkbllll_prescreen',
                                        'CVHFrkbssss_direct_scf',
                                        'CVHFrkbssss_direct_scf_dm')
            self.opt_ssss.direct_scf_threshold = self.direct_scf_threshold
            set_vkscreen(self.opt_ssss, 'CVHFrkbllll_vkscreen')
            self.opt_ssll = _vhf.VHFOpt(mol, 'cint2e_spsp1',
                                        'CVHFrkbssll_prescreen',
                                        'CVHFrkbssll_direct_scf',
                                        'CVHFrkbssll_direct_scf_dm')
            self.opt_ssll.direct_scf_threshold = self.direct_scf_threshold
            set_vkscreen(self.opt_ssll, 'CVHFrkbssll_vkscreen')

    def set_occ(self, mo_energy, mo_coeff=None):
        mol = self.mol
        n4c = mo_energy.size
        n2c = n4c / 2
        c = mol.light_speed
        mo_occ = numpy.zeros(n2c * 2)
        if mo_energy[n2c] > -1.999 * mol.light_speed**2:
            mo_occ[n2c:n2c+mol.nelectron] = 1
        else:
            n = 0
            for i, e in enumerate(mo_energy):
                if e > -1.999 * mol.light_speed**2 and n < mol.nelectron:
                    mo_occ[i] = 1
                    n += 1
        if self.verbose >= log.INFO:
            self.dump_occ(mol, mo_occ, mo_energy)
        return mo_occ

    # full density matrix for UHF
    @lib.omnimethod
    def make_rdm1(self, mo_coeff, mo_occ):
        mo = mo_coeff[:,mo_occ>0]
        return numpy.dot(mo*mo_occ[mo_occ>0], mo.T.conj())

    def dump_occ(self, mol, mo_occ, mo_energy):
        n4c = mo_energy.size
        n2c = n4c / 2
        log.info(self, 'HOMO %d = %.12g, LUMO %d = %.12g,', \
                 n2c+mol.nelectron, mo_energy[n2c+mol.nelectron-1], \
                 n2c+mol.nelectron+1, mo_energy[n2c+mol.nelectron])
        log.debug(self, 'NES  mo_energy = %s', mo_energy[:n2c])
        log.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])

    def calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ):
        e_tmp = hf.SCF.calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ)
        return e_tmp


    def get_coulomb_vj_vk(self, mol, dm, coulomb_allow='SSSS', hermi=1):
        if coulomb_allow.upper() == 'LLLL':
            log.info(self, 'Coulomb integral: (LL|LL)')
            j1, k1 = _call_veff_llll(self, dm, hermi)
            n2c = j1.shape[1]
            vj = numpy.zeros_like(dm)
            vk = numpy.zeros_like(dm)
            vj[...,:n2c,:n2c] = j1
            vk[...,:n2c,:n2c] = k1
        elif coulomb_allow.upper() == 'SSLL' \
          or coulomb_allow.upper() == 'LLSS':
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL)')
            vj, vk = _call_veff_ssll(self, dm, hermi)
            j1, k1 = _call_veff_llll(self, dm, hermi)
            n2c = j1.shape[1]
            vj[...,:n2c,:n2c] += j1
            vk[...,:n2c,:n2c] += k1
        else: # coulomb_allow == 'SSSS'
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
            vj, vk = _call_veff_ssll(self, dm, hermi)
            j1, k1 = _call_veff_llll(self, dm, hermi)
            n2c = j1.shape[1]
            vj[...,:n2c,:n2c] += j1
            vk[...,:n2c,:n2c] += k1
            j1, k1 = _call_veff_ssss(self, dm, hermi)
            vj[...,n2c:,n2c:] += j1
            vk[...,n2c:,n2c:] += k1
        return vj, vk

    def get_gaunt_vj_vk(self, mol, dm):
        '''Dirac-Coulomb-Gaunt'''
        log.info(self, 'integral for Gaunt term')
        vj, vk = hf.get_vj_vk(pycint.rkb_vhf_gaunt, mol, dm)
        return -vj, -vk

    def get_gaunt_vj_vk_screen(self, mol, dm):
        '''Dirac-Coulomb-Gaunt'''
        log.info(self, 'integral for Gaunt term')
        vj, vk = hf.get_vj_vk(pycint.rkb_vhf_gaunt_direct, mol, dm)
        return -vj, -vk

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        '''Dirac-Coulomb'''
        t0 = (time.clock(), time.time())
        if self.direct_scf:
            ddm = dm - dm_last
            vj, vk = self.get_coulomb_vj_vk(mol, ddm, self._coulomb_now)
            vhf = vhf_last + vj - vk
        else:
            vj, vk = self.get_coulomb_vj_vk(mol, dm, self._coulomb_now)
            vhf = vj - vk
        log.timer(self, 'vj and vk', *t0)
        return vhf

    def get_vhf_with_gaunt(self, mol, dm, dm_last=0, vhf_last=0):
        if self.direct_scf:
            ddm = dm - dm_last
            vj, vk = self.get_coulomb_vj_vk(mol, ddm, self._coulomb_now)
            vj1, vk1 = self.get_gaunt_vj_vk_screen(mol, ddm)
            return vhf_last + vj0 + vj1 - vk0 - vk1
        else:
            vj, vk = self.get_coulomb_vj_vk(mol, dm, self._coulomb_now)
            vj1, vk1 = self.get_gaunt_vj_vk(mol, dm)
            return vj0 + vj1 - vk0 - vk1

    def set_potential(self, v='coulomb', oob=0, ssss=1):
        if v.lower() == 'coulomb':
            if oob > 0:
                self.get_veff = self.coulomb_oob
            else:
                try:
                    del(self.get_veff)
                except:
                    pass
            #if 0 <= ssss <= 1:
            #    self.with_ssss = ssss
            #else:
            #    raise KeyError('Incorrect (SS|SS) approx.')
        elif v.lower() == 'gaunt':
            self.with_gaunt = True
        else:
            raise KeyError('Unknown potential.')

        if 0 <= oob <=2:
            self.oob = oob
        else:
            raise KeyError('Incorrect OOB level.')

    def scf_cycle(self, mol, conv_threshold=1e-9, dump_chk=True, init_dm=None):
        if init_dm is None:
            hf_energy, dm = self.make_init_guess(mol)
        else:
            hf_energy = 0
            dm = init_dm

        if self.oob > 0:
            return hf.scf_cycle(mol, self, conv_threshold, dump_chk, \
                                init_dm=dm)

        if init_dm is None and self._coulomb_now.upper() == 'LLLL':
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
                    = hf.scf_cycle(mol, self, 4e-3, dump_chk, init_dm=dm)
            dm = self.make_rdm1(mo_coeff, mo_occ)
            self._coulomb_now = 'SSLL'

        if init_dm is None and (self._coulomb_now.upper() == 'SSLL' \
                             or self._coulomb_now.upper() == 'LLSS'):
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
                    = hf.scf_cycle(mol, self, 4e-4, dump_chk, init_dm=dm)
            dm = self.make_rdm1(mo_coeff, mo_occ)
            self._coulomb_now = 'SSSS'

        if self.with_ssss:
            self._coulomb_now = 'SSSS'
        else:
            self._coulomb_now = 'SSLL'

        if self.with_gaunt:
            self.get_veff = self.get_vhf_with_gaunt

        return hf.scf_cycle(mol, self, conv_threshold, dump_chk, init_dm=dm)


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


def _call_veff_llll(mf, dm, hermi=1):
    mol = mf.mol
    n2c = mol.nao_2c()
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] / 2
        dms = dm[:n2c,:n2c].copy()
    else:
        n2c = dm[0].shape[0] / 2
        dms = []
        for dmi in dm:
            dms.append(dmi[:n2c,:n2c].copy())
    if mf.direct_scf:
        opt = mf.opt_llll
    else:
        opt = None
    vj, vk = _vhf.rdirect_mapdm('cint2e', 'CVHFdot_rs8',
                                ('CVHFrs8_ji_s2kl', 'CVHFrs8_jk_s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, opt)
    return _jk_triu_(vj, vk, hermi)

def _call_veff_ssll(mf, dm, hermi=1):
    mol = mf.mol
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] / 2
        dmll = dm[:n2c,:n2c].copy()
        dmsl = dm[n2c:,:n2c].copy()
        dmss = dm[n2c:,n2c:].copy()
        dms = (dmll, dmss, dmsl)
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] / 2
        dms = []
        for dmi in dm:
            dms.append(dmi[:n2c,:n2c].copy())
        for dmi in dm:
            dms.append(dmi[n2c:,n2c:].copy())
        for dmi in dm:
            dms.append(dmi[n2c:,:n2c].copy())
    jks = ('CVHFrs4_lk_s2ij',) * n_dm \
        + ('CVHFrs4_ji_s2kl',) * n_dm \
        + ('CVHFrs4_jk_s1il',) * n_dm
    if mf.direct_scf:
        opt = mf.opt_ssll
    else:
        opt = None
    c1 = .5/mol.light_speed
    vx = _vhf.rdirect_bindm('cint2e_spsp1', 'CVHFdot_rs4', jks, dms, 1,
                            mol._atm, mol._bas, mol._env, opt) * c1**2
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex)
    vj[:,n2c:,n2c:] = vx[      :n_dm  ,:,:]
    vj[:,:n2c,:n2c] = vx[n_dm  :n_dm*2,:,:]
    vk[:,n2c:,:n2c] = vx[n_dm*2:      ,:,:]
    if n_dm == 1:
        vj = vj.reshape(vj.shape[1:])
        vk = vk.reshape(vk.shape[1:])
    return _jk_triu_(vj, vk, hermi)

def _call_veff_ssss(mf, dm, hermi=1):
    mol = mf.mol
    c1 = .5/mol.light_speed
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] / 2
        dms = dm[n2c:,n2c:].copy()
    else:
        n2c = dm[0].shape[0] / 2
        dms = []
        for dmi in dm:
            dms.append(dmi[n2c:,n2c:].copy())
    if mf.direct_scf:
        opt = mf.opt_ssss
    else:
        opt = None
    vj, vk = _vhf.rdirect_mapdm('cint2e_spsp1spsp2', 'CVHFdot_rs8',
                                ('CVHFrs8_ji_s2kl', 'CVHFrs8_jk_s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, opt) * c1**4
    return _jk_triu_(vj, vk, hermi)


def time_reversal_ao_idx(mol):
    n2c = mol.num_2C_function()
    tao = mol.time_reversal_map()
    # tao(i) = -j  means  T(f_i) = -f_j
    # tao(i) =  j  means  T(f_i) =  f_j
    taoL = numpy.array(map(lambda x: abs(x)-1, tao)) # -1 to fit C-array
    idx = numpy.hstack((taoL, taoL+n2c))
    signL = map(lambda x: 1 if x>0 else -1, tao)
    sign = numpy.hstack((signL, signL))
    return idx, sign

def time_reversal_matrix(mol, mat):
    tao, sign = time_reversal_ao_idx(mol)
    tmat = numpy.empty_like(mat)
    for j in range(mat.__len__()):
        for i in range(mat.__len__()):
            tmat[tao[i],tao[j]] = mat[i,j] * sign[i]*sign[j]
    return tmat.conjugate()

class RHF(UHF):
    __doc__ = 'Dirac-RHF'
    def __init__(self, mol):
        if mol.nelectron.__mod__(2) is not 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
        UHF.__init__(self, mol)

    # full density matrix for RHF
    @lib.omnimethod
    def make_rdm1(self, mo_coeff, mo_occ):
        '''D/2 = \psi_i^\dag\psi_i = \psi_{Ti}^\dag\psi_{Ti}
        D(UHF) = \psi_i^\dag\psi_i + \psi_{Ti}^\dag\psi_{Ti}
        RHF average the density of spin up and spin down:
        D(RHF) = (D(UHF) + T[D(UHF)])/2
        '''
        dm = UHF.make_rdm1(mo_coeff, mo_occ)
        return (dm + time_reversal_matrix(self.mol, dm)) * .5

    def dump_occ(self, mol, mo_occ, mo_energy):
        n4c = mo_energy.size
        n2c = n4c / 2
        log.info(self, 'HOMO %d = %.12g, LUMO %d = %.12g,', \
                 (n2c+mol.nelectron)/2, mo_energy[n2c+mol.nelectron-1], \
                 (n2c+mol.nelectron)/2+1, mo_energy[n2c+mol.nelectron])
        log.debug(self, 'NES  mo_energy = %s', mo_energy[:n2c])
        log.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])



if __name__ == '__main__':
    mol = gto.Mole()
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
