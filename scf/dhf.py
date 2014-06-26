#!/usr/bin/env python
# -*- coding: utf-8
#
# File: dhf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Dirac Hartree-Fock
'''

__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'
__version__ = '$ 0.1 $'

import ctypes
import numpy
import scipy.linalg.flapack as lapack
import gto
import lib.logger as log
import lib
import lib.parameters as param
import lib.pycint as pycint
import hf

_cint = hf._cint

__doc__ = '''Options:
self.chkfile = '/dev/shm/...'
self.fout = '...'
self.diis_space = 6
self.diis_start_cycle = 1
self.damp_factor = 1
self.level_shift_factor = 0
self.scf_threshold = 1e-10
self.max_scf_cycle = 50
self.oob = 0                    # operator oriented basis level
                                # 1 sp|f> -> |f>
                                # 2 sp|f> -> sr|f>


self.init_guess(method)         # method = one of 'atom', '1e', 'chkfile'
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
        self.scf_threshold = 1e-8
        self.oob = 0
        self.with_ssss = True
        self.set_init_guess('1e')
        self._coulomb_now = 'LLLL' # 'SSSS' ~ LLLL+LLSS+SSSS
        self.with_gaunt = False

    def _init_guess_by_chkfile(self, mol):
        '''Read initial guess from chkfile.'''
        #try:
        #    chk_mol, scf_rec = chkfile.read_scf(self.chkfile)
        #except IOError:
        #    log.warn(mol, 'Fail in reading from %s. Use 1e initial guess', \
        #             self.chkfile)
        #    return self._init_guess_by_1e(mol)
        chk_mol, scf_rec = chkfile.read_scf(self.chkfile)

        if not mol.is_same_mol(chk_mol):
            #raise RuntimeError('input moleinfo is incompatible with chkfile')
            log.warn(mol, 'input moleinfo is incompatible with chkfile. ' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

        log.info(self, '\n')
        log.info(self, 'Read initial guess from file %s.', self.chkfile)

        n2c = mol.num_2C_function()
        n4c = n2c * 2
        c = mol.light_speed

        nbas_chk = chk_mol.nbas
        for ia in range(mol.natm):
            basis_add = mol.basis[mol.symbol_of_atm(ia)]
            chk_mol._bas.extend(chk_mol.make_bas_env_by_atm_id(ia, basis_add))
        chk_mol.nbas = chk_mol._bas.__len__()
        bras = range(nbas_chk, chk_mol.nbas)
        kets = range(nbas_chk)

        if hf.chk_scf_type(scf_rec['mo_coeff'])[0] == 'R':
            s = chk_mol.intor_cross('cint1e_ovlp', bras, kets)
            t = chk_mol.intor_cross('cint1e_spsp', bras, kets)
            n1 = s.shape[1]
            proj = numpy.zeros((n4c, n1*2), numpy.complex)
            proj[:n2c,:n1] = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp'),s)
            proj[n2c:,n1:] = numpy.linalg.solve(mol.intor_symmetric('cint1e_spsp'),t)

            mo_coeff = numpy.dot(proj, scf_rec['mo_coeff'])
            mo_occ = scf_rec['mo_occ']
            dm = numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())
            self._coulomb_now = 'SSSS'
        else:
            import symm
            log.debug(self, 'Convert NR MO coeff from  %s', self.chkfile)
            if hf.chk_scf_type(scf_rec['mo_coeff']) == 'NR-RHF':
                c = scf_rec['mo_coeff']
            else:
                c = scf_rec['mo_coeff'][0]
            s0 = chk_mol.intor_cross('cint1e_ovlp_sph', bras, kets)
            ua, ub = symm.cg.real2spinor_whole(mol)
            s = numpy.dot(ua.T.conj(), s0) + numpy.dot(ub.T.conj(), s0) # (*)
            proj = numpy.linalg.solve(mol.intor_symmetric('cint1e_ovlp'), s)

            # alpha, beta are summed in Eq. (*)
            nocc = mol.nelectron / 2
            mo_coeff = numpy.dot(proj, c)[:,:nocc]
            dm = numpy.zeros((n4c,n4c), dtype=complex)
            dm_ll = numpy.dot(mo_coeff, mo_coeff.T.conj())
            # NR alpha and beta MO does not have time reversal symmetry
            dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
            self._coulomb_now = 'LLLL'
        return scf_rec['hf_energy'], dm

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
        import symm
        if mol is None:
            mol = self.mol
        try:
            ehf, dm0 = hf.init_guess_by_minao(self, mol)
        except KeyError:
            log.warn(self, 'Fail in generating initial guess from MINAO. ' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

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

    def dump_scf_option(self):
        hf.SCF.dump_scf_option(self)
        log.info(self, 'OOB = %d', self.oob)

    def eig(self, h, s):
        try:
            import lib.jacobi
            return lib.jacobi.zgeeigen(h, s)
        except ImportError:
            c, e, info = lapack.zhegv(h, s)
            print e
            return e, c, info

    def init_diis(self):
        diis_a = diis.SCF_DIIS(self)
        diis_a.diis_space = self.diis_space
        #diis_a.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                f = diis_a.update(s, d, f)
            if cycle < self.diis_start_cycle-1:
                f = damping(s, d, f, self.damp_factor)
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
        h1e = numpy.zeros((n4c, n4c), numpy.complex)
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

    def init_direct_scf(self, mol):
        if self.direct_scf:
            natm = lib.c_int_p(ctypes.c_int(mol._atm.__len__()))
            nbas = lib.c_int_p(ctypes.c_int(mol._bas.__len__()))
            atm = lib.c_int_arr(mol._atm)
            bas = lib.c_int_arr(mol._bas)
            env = lib.c_double_arr(mol._env)
            _cint.init_rkb_direct_scf_(atm, natm, bas, nbas, env)
            if self.with_gaunt:
                _cint.init_rkb_gaunt_direct_scf_(atm, natm, bas, nbas, env)
            self.set_direct_scf_threshold(self.direct_scf_threshold)
        else:
            _cint.turnoff_direct_scf_()

    def del_direct_scf(self):
        _cint.del_rkb_direct_scf_()

    def set_mo_occ(self, mo_energy, mo_coeff=None):
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
    def calc_den_mat(self, mo_coeff, mo_occ):
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
        e_tmp = hf.SCF.calc_tot_elec_energy(vhf, dm, mo_energy, mo_occ)
        return e_tmp

    def coulomb_oob(self, mol, dm, dm_last=0, vhf_last=0):
        if self.direct_scf:
            dm = dm - dm_last

        n2c = mol.num_2C_function()
        n4c = n2c * 2
        c = mol.light_speed
        vhf = numpy.zeros((n4c, n4c), numpy.complex)

        if self.oob == 1:
            log.info(self, 'Coulomb integral: OOB, |sp f> = |f> C')

            s = mol.intor_symmetric('cint1e_ovlp')
            p = mol.intor_symmetric('cint1e_sp') * (.5/c)
            u = numpy.dot(numpy.linalg.inv(s), p)
            dmll = dm[:n2c,:n2c]
            dmsl = numpy.dot(u, dm[n2c:,:n2c])
            dmss = reduce(numpy.dot, (u, dm[n2c:,n2c:], u.T.conj()))
            dmjk = numpy.rollaxis(numpy.array((dmll, dmsl, dmss)), 0, 3)
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_direct_o3, mol, dmjk)

            vhf[:n2c,:n2c] += vj[0] + vj[2]
            vhf[n2c:,n2c:] += reduce(numpy.dot, (u.T.conj(), vj[0]+vj[2], u))
            vhf[:n2c,:n2c] -= vk[0]
            vhf[n2c:,:n2c] -= numpy.dot(u.T.conj(), vk[1])
            vhf[n2c:,n2c:] -= reduce(numpy.dot, (u.T.conj(), vk[2], u))
            vhf[:n2c,n2c:] = vhf[n2c:,:n2c].T.conj()
        elif self.oob == 2:
# rp = 1/2c <sr|sp>
# rr = <sr|sr>
# rr_rp = rr^{-1} rp
# (SS|LL) = (pr_rr.LL.rp|LL) + (pr.LL.rr_rp|LL)
# (SS|SS) = (pr_rr.LL.rp|pr_rr.LL.rp) + (pr.LL.rr_rp|pr_rr.LL.rp)
#         + (pr_rr.LL.rp|pr.LL.rr_rp) + (pr.LL.rr_rp|pr.LL.rr_rp)
# =>
#       JLL = (LL|LL) * dmLL -> JSS = (SS|LL) * dmLL, 2 type J
#       JLL = (SS|LL) * dmSS, 2 type dm -> JSS = (SS|SS) * dmSS, 2 type J
#       KLL = (LL|LL) * dmLL
#       KSL = (SS|LL) * dmSL, 2 type dm
#       KSS = (SS|SS) * dmSS, 4 type dm
            log.info(self, 'Coulomb integral: OOB, |sp f> = |sr f> C')

            s  = mol.intor_symmetric('cint1e_ovlp')
            rr = mol.intor_symmetric('cint1e_srsr')
            rp = mol.intor('cint1e_srsp') * (.5/c)
            s_rp = numpy.dot(numpy.linalg.inv(s), rp)
            rr_rp = numpy.dot(numpy.linalg.inv(rr), rp)
            dmll = dm[:n2c,:n2c]
            dmrl = numpy.dot(rr_rp, dm[n2c:,:n2c])
            dmsl = numpy.dot(s_rp, dm[n2c:,:n2c])
            dmss = reduce(numpy.dot, (s_rp, dm[n2c:,n2c:], s_rp.T.conj()))
            dmsr = reduce(numpy.dot, (s_rp, dm[n2c:,n2c:], rr_rp.T.conj()))
            dmrs = reduce(numpy.dot, (rr_rp, dm[n2c:,n2c:], s_rp.T.conj()))
            dmrr = reduce(numpy.dot, (rr_rp, dm[n2c:,n2c:], rr_rp.T.conj()))
            #dmjk = numpy.rollaxis(numpy.array((dmll, dmsl, dmrl,
            #                                   dmss, dmsr, dmrs, dmrr)), 0, 3)
            #vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_direct_o3, mol, dmjk)

            #jll = vj[0] + (vj[4] + vj[5]) * .5
            #jss = (rr_rp.H * jll * s_rp + s_rp.H * jll * rr_rp) * .5
            #ksl = (rr_rp.H * vk[1] + s_rp.H * vk[2]) * .5
            #kss = (rr_rp.H * vk[3] * rr_rp + rr_rp.H * vk[4] * s_rp \
            #       + s_rp.H * vk[5] * rr_rp + s_rp.H * vk[6] * s_rp) * .25
#            dmjk = numpy.rollaxis(numpy.array((dmll, dmsl, dmrl,
#                                               dmss, dmsr, dmrr)), 0, 3)
#            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_direct_o3, mol, dmjk)

#            jll = vj[0] + vj[4]
#            tmp = rr_rp.H * jll * s_rp
#            jss = (tmp + tmp.H) * .5
#            ksl = (rr_rp.H * vk[1] + s_rp.H * vk[2]) * .5
#            tmp = rr_rp.H * vk[4] * s_rp
#            kss = (rr_rp.H * vk[3] * rr_rp + tmp + tmp.T.conj() \
#                   + s_rp.H * vk[5] * s_rp) * .25
            dmjk = numpy.rollaxis(numpy.array((dmll, dmsl, dmrl,
                                               dmss, dmsr, dmrr)), 0, 3)
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_direct_o3, mol, dmjk)

            jll = vj[0] + vj[4]
            tmp = reduce(numpy.dot, (rr_rp.T.conj(), jll, s_rp))
            jss = (tmp + tmp.T.conj()) * .5
            ksl = (numpy.dot(rr_rp.T.conj(), vk[1]) \
                   + numpy.dot(s_rp.T.conj(), vk[2])) * .5
            tmp = reduce(numpy.dot, (rr_rp.T.conj(), vk[4], s_rp))
            kss = (reduce(numpy.dot, (rr_rp.T.conj(), vk[3], rr_rp)) \
                   + tmp + tmp.T.conj() \
                   + reduce(numpy.dot, (s_rp.T.conj(), vk[5], s_rp))) * .25

            vhf[:n2c,:n2c] += jll
            vhf[n2c:,n2c:] += jss
            vhf[:n2c,:n2c] -= vk[0]
            vhf[n2c:,:n2c] -= ksl
            vhf[n2c:,n2c:] -= kss
            vhf[:n2c,n2c:] = vhf[n2c:,:n2c].T.conj()
#ABORT        elif self.oob == 10:
# test sp r_{12}^{-1} sp ~= spsp r_{12}^{-1}
#ABORT            log.info(self, 'Coulomb integral: approx. of p 1/r12 ~= 1/r12 p')
#ABORT
#ABORT            s  = mol.intor_symmetric('cint1e_ovlp')
#ABORT            pp = mol.intor_symmetric('cint1e_spsp') * (.5/c)
#ABORT            s_pp = numpy.mat(s).I * pp
#ABORT            dmll = dm[:n2c,:n2c]
#ABORT            dmsl = (.5/c) * dm[n2c:,:n2c]
#ABORT            dmtl = s_pp * dm[n2c:,:n2c]
#ABORT            dmtt = s_pp * dm[n2c:,n2c:] * s_pp.H
#ABORT            dmts = s_pp * dm[n2c:,n2c:] * (.5/c)
#ABORT            dmst = (.5/c) * dm[n2c:,n2c:] * s_pp.H
#ABORT            dmss = (.5/c) * dm[n2c:,n2c:] * (.5/c)
#ABORT            dmjk = numpy.rollaxis(numpy.array((dmll, dmsl, dmtl,
#ABORT                                               dmtt, dmts, dmst, dmss)), 0, 3)
#ABORT            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_o3, mol, dmjk)
#ABORT
#ABORT            jll = vj[0] + (vj[4] + vj[5]) * .5
#ABORT            jss = ((.5/c) * jll * s_pp + s_pp.H * jll * (.5/c)) * .5
#ABORT            ksl = ((.5/c) * vk[1] + s_pp.H * vk[2]) * .5
#ABORT            kss = ((.5/c) * vk[3] * (.5/c) + (.5/c) * vk[4] * s_pp \
#ABORT                   + s_pp.H * vk[5] * (.5/c) + s_pp.H * vk[6] * s_pp) * .25
#ABORT            vhf[:n2c,:n2c] += jll
#ABORT            vhf[n2c:,n2c:] += jss
#ABORT            vhf[:n2c,:n2c] -= vk[0]
#ABORT            vhf[n2c:,:n2c] -= ksl
#ABORT            vhf[n2c:,n2c:] -= kss
#ABORT            vhf[:n2c,n2c:] = vhf[n2c:,:n2c].T.conj()
        if self.direct_scf:
            vhf = vhf_last + vhf
        return vhf

    def gaunt_oob(self, mol, dm):
        pass
        #return vhf

    def get_coulomb_vj_vk(self, mol, dm, coulomb_allow='SSSS'):
        if coulomb_allow.upper() == 'LLLL':
            log.info(self, 'Coulomb integral: (LL|LL)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_o3, mol, dm)
        elif coulomb_allow.upper() == 'SSLL' \
          or coulomb_allow.upper() == 'LLSS':
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_sl_o3, mol, dm)
        else: # coulomb_allow == 'SSSS'
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_coul_o3, mol, dm)
        return vj, vk

    def get_coulomb_vj_vk_screen(self, mol, dm, coulomb_allow='SSSS'):
        if coulomb_allow.upper() == 'LLLL':
            log.info(self, 'Coulomb integral: (LL|LL)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_ll_direct_o3, mol, dm)
        elif coulomb_allow.upper() == 'SSLL' \
          or coulomb_allow.upper() == 'LLSS':
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_sl_direct_o3, mol, dm)
        else: # coulomb_allow == 'SSSS'
            log.info(self, 'Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
            vj, vk = hf.get_vj_vk(pycint.rkb_vhf_coul_direct_o3, mol, dm)
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

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        '''Dirac-Coulomb'''
        if self.direct_scf:
            vj, vk = self.get_coulomb_vj_vk_screen(mol, dm-dm_last, \
                                                   self._coulomb_now)
            return vhf_last + vj - vk
        else:
            vj, vk = self.get_coulomb_vj_vk(mol, dm, self._coulomb_now)
            return vj - vk

    def get_vhf_with_gaunt(self, mol, dm, dm_last=0, vhf_last=0):
        if self.direct_scf:
            ddm = dm - dm_last
            vj0, vk0 = self.get_coulomb_vj_vk_screen(mol, ddm, \
                                                     self._coulomb_now)
            vj1, vk1 = self.get_gaunt_vj_vk_screen(mol, ddm)
            return vhf_last + vj0 + vj1 - vk0 - vk1
        else:
            vj0, vk0 = self.get_coulomb_vj_vk(mol, dm, self._coulomb_now)
            vj1, vk1 = self.get_gaunt_vj_vk(mol, dm)
            return vj0 + vj1 - vk0 - vk1

    def set_potential(self, v='coulomb', oob=0, ssss=1):
        if v.lower() == 'coulomb':
            if oob > 0:
                self.get_eff_potential = self.coulomb_oob
            else:
                try:
                    del(self.get_eff_potential)
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

    def scf_cycle(self, mol, scf_threshold=1e-9, dump_chk=True, init_dm=None):
        if init_dm is None:
            hf_energy, dm = self.init_guess_method(mol)
        else:
            hf_energy = 0
            dm = init_dm

        if self.oob > 0:
            return hf.scf_cycle(mol, self, scf_threshold, dump_chk, \
                                init_dm=dm)

        if init_dm is None and self._coulomb_now.upper() is 'LLLL':
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
                    = hf.scf_cycle(mol, self, 4e-3, dump_chk, init_dm=dm)
            dm = self.calc_den_mat(mo_coeff, mo_occ)
            self._coulomb_now = 'SSLL'

        if init_dm is None and self._coulomb_now.upper() is 'SSLL' \
                            or self._coulomb_now.upper() is 'LLSS':
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
                    = hf.scf_cycle(mol, self, 4e-4, dump_chk, init_dm=dm)
            dm = self.calc_den_mat(mo_coeff, mo_occ)
            self._coulomb_now = 'SSSS'

        if self.with_ssss:
            self._coulomb_now = 'SSSS'
        else:
            self._coulomb_now = 'SSLL'

        if self.with_gaunt:
            self.get_eff_potential = self.get_vhf_with_gaunt

        return hf.scf_cycle(mol, self, scf_threshold, dump_chk, init_dm=dm)


def time_reversal_ao_idx(mol):
    n2c = mol.num_2C_function()
    tao = mol.time_reversal_spinor()
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
    def calc_den_mat(self, mo_coeff, mo_occ):
        '''D/2 = \psi_i^\dag\psi_i = \psi_{Ti}^\dag\psi_{Ti}
        D(UHF) = \psi_i^\dag\psi_i + \psi_{Ti}^\dag\psi_{Ti}
        RHF average the density of spin up and spin down:
        D(RHF) = (D(UHF) + T[D(UHF)])/2
        '''
        dm = UHF.calc_den_mat(mo_coeff, mo_occ)
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
    import gto.basis as basis
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dhf'

    mol.atom.extend([['He', (0.,0.,0.)], ])
# input even-tempered basis
    mol.etb = {
        'He': { 'max_l' : 1           # for even-tempered basis
              , 's'     : (4, 1, 1.8) # for etb:(num_basis, alpha, beta)
              , 'p'     : (1, 1, 1.8) # for etb: eta = alpha*beta**i
              , 'd'     : (0, 1, 1.8) #           for i in range num_basis
              , 'f'     : (0,0,0)
              , 'g'     : (0,0,0)}, }
# or input basis information directly
#    mol.basis = {
#        'He': [(0, 0, (1, 1)),
#               (0, 0, (3, 1)),
#               (1, 0, (1, 1)), ]}
    mol.build()

##############
# SCF result
    method = UHF(mol)
    method.init_guess('1e')
#TODO:    if restart:
#TODO:        method.init_guess('chkfile')
#TODO:    if by_atom:
#TODO:        method.init_guess('atom')
    method.set_potential('coulomb')
    #method.set_direct_scf_threshold(1e-18)
    energy = method.scf(mol) #=-2.63133406544043
    print energy
