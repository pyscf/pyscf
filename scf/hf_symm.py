#!/usr/bin/env python
# -*- coding: utf-8
#
# File: hf_symm.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Hartree-Fock
'''

__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'
__version__ = '$ 0.2 $'

import os
import cPickle as pickle
import ctypes
import time

import numpy
import scipy.linalg.flapack as lapack
import gto
import lib.logger as log
import symm
import diis
import lib
import lib.parameters as param
import lib.pycint as pycint
import hf

alib = '/'.join((os.environ['HOME'], 'code/lib/libvhf.so'))
_cint = ctypes.cdll.LoadLibrary(alib)


def dump_mo_coeff(mol, mo_coeff, e_ir_idx, argsort, title='   '):
    log.debug(mol, ' **** %s MO coefficients ****', title)
    nmo = mo_coeff.shape[1]
    mo_coeff = mo_coeff[:,argsort]
    for k in range(0, nmo, 5):
        lbl = []
        for i1 in range(k, min(k+5,nmo)):
            e,ir,i = e_ir_idx[argsort[i1]]
            lbl.append('#%d(%s %d)' % (i1+1, mol.irrep_name[ir], i+1))
        log.debug(mol, ('%s MO_id+1 ' % title) + ' '.join(lbl))
        hf.dump_orbital_coeff(mol, mo_coeff[:,k:k+5])

def dump_mo_energy(mol, mo_energy, nocc, ehomo, elumo, title=''):
    nirrep = mol.symm_orb.__len__()
    for ir in range(nirrep):
        if nocc[ir] == 0:
            log.debug(mol, '%s%s nocc = 0', title, mol.irrep_name[ir])
        elif nocc[ir] == mo_energy[ir].__len__():
            log.debug(mol, '%s%s nocc = %d, HOMO = %.12g,', \
                      title, mol.irrep_name[ir], \
                      nocc[ir], mo_energy[ir][nocc[ir]-1])
        else:
            log.debug(mol, '%s%s nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                      title, mol.irrep_name[ir], \
                      nocc[ir], mo_energy[ir][nocc[ir]-1],
                      mo_energy[ir][nocc[ir]])
            if mo_energy[ir][nocc[ir]-1] > elumo:
                log.warn(mol, '!! %s%s HOMO > system LUMO', \
                         title, mol.irrep_name[ir])
            if mo_energy[ir][nocc[ir]] < ehomo:
                log.warn(mol, '!! %s%s LUMO < system HOMO', \
                         title, mol.irrep_name[ir])
        log.debug(mol, '   mo_energy = %s', mo_energy[ir])

def argsort_mo_energy(mol, mo_energy):
    nirrep = mol.symm_orb.__len__()
    mo_e = []
    for ir in range(nirrep):
        for i,e in enumerate(mo_energy[ir]):
            mo_e.append((e,ir,i))
    return mo_e, sorted(range(len(mo_e)), key=mo_e.__getitem__)

def so2ao_mo_coeff(so, mo_coeff):
    return numpy.hstack([numpy.dot(so[ir],mo_coeff[ir]) \
                         for ir in range(so.__len__())])


class RHF(hf.RHF):
    def __init__(self, mol):
        hf.RHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc = {}

    def dump_scf_option(self):
        hf.RHF.dump_scf_option(self)
        log.info(self, 'RHF with symmetry adpated basis')
        float_ir = []
        fix_ne = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if self.irrep_nocc.has_key(irname):
                fix_ne += self.irrep_nocc[irname]
            else:
                float_ir.append(irname)
        if fix_ne > 0:
            log.info(self, 'fix %d electrons in irreps %s', \
                     fix_ne, self.irrep_nocc.items())
            if fix_ne > self.mol.nelectron:
                log.error(self, 'number of electrons error in irrep_nocc %s', \
                          self.irrep_nocc.items())
                raise ValueError('irrep_nocc')
        if float_ir:
            log.info(self, '%d free electrons in irreps %s', \
                     self.mol.nelectron-fix_ne, float_ir)
        elif fix_ne != self.mol.nelectron:
            log.error(self, 'number of electrons error in irrep_nocc %s', \
                      self.irrep_nocc.items())
            raise ValueError('irrep_nocc')

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        cs = []
        es = []
        for ir in range(nirrep):
            c, e, info = lapack.dsygv(h[ir], s[ir])
            cs.append(c)
            es.append(e)
        return es, cs, info

    def damping(self, s, d, f, factor):
        if factor < 1e-3:
            return f
        else:
            return [hf.RHF.damping(self, s[ir], d[ir], f[ir], factor) \
                    for ir in range(self.mol.symm_orb.__len__())]

    def level_shift(self, s, d, f, factor):
        if factor < 1e-3:
            return f
        else:
            return [hf.RHF.level_shift(self, s[ir], d[ir], f[ir], factor) \
                    for ir in range(self.mol.symm_orb.__len__())]

    def init_diis(self):
        diis_a = diis.SCF_DIIS(self)
        diis_a.diis_space = self.diis_space
        #diis_a.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                nirrep = self.mol.symm_orb.__len__()
                errvec = []
                for ir in range(nirrep):
                    sdf = reduce(numpy.dot, (s[ir], d[ir], f[ir]))
                    errvec.append((sdf.T.conj()-sdf).flatten())
                errvec = numpy.hstack(errvec)
                diis_a.err_vec_stack.append(errvec)
                log.debug(self, 'diis-norm(errvec) = %g', \
                          numpy.linalg.norm(errvec))
                if diis_a.err_vec_stack.__len__() > diis_a.diis_space:
                    diis_a.err_vec_stack.pop(0)
                f1 = numpy.hstack([fi.flatten() for fi in f])
                fnew = diis.DIIS.update(diis_a, f1)
                p0 = 0
                f = []
                for si in s:
                    n = si.shape[0]
                    f.append(fnew[p0:p0+n*n].reshape(n,n))
                    p0 += n*n
            if cycle < self.diis_start_cycle-1:
                f = self.damping(s, d, f, self.damp_factor)
                f = self.level_shift(s, d, f, self.level_shift_factor)
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = self.level_shift(s, d, f, fac)
            return f
        return scf_diis

    @lib.omnimethod
    def get_hcore(self, mol):
        h = mol.intor_symmetric('cint1e_kin_sph') \
                + mol.intor_symmetric('cint1e_nuc_sph')
        return symm.symmetrize_matrix(h, mol.symm_orb)

    @lib.omnimethod
    def get_ovlp(self, mol):
        s = mol.intor_symmetric('cint1e_ovlp_sph')
        return symm.symmetrize_matrix(s, mol.symm_orb)

    def make_fock(self, h1e, vhf):
        f = []
        nirrep = self.mol.symm_orb.__len__()
        for ir in range(nirrep):
            f.append(h1e[ir] + vhf[ir])
        return f

    def dump_scf_to_chkfile(self, hf_energy, mo_energy, mo_occ, mo_coeff):
        hf.dump_scf_to_chkfile(self.mol, self.chkfile, hf_energy, \
                               numpy.hstack(mo_energy), \
                               numpy.hstack(mo_occ), \
                               so2ao_mo_coeff(self.mol.symm_orb, mo_coeff))

    def symmetrize_den_mat(self, dm_ao):
        s0 = self.mol.intor_symmetric('cint1e_ovlp_sph')
        s = symm.symmetrize_matrix(s0, self.mol.symm_orb)
        nirrep = self.mol.symm_orb.__len__()
        dm = []
        for ir in range(nirrep):
            sinv = numpy.linalg.inv(s[ir])
            so = reduce(numpy.dot, (s0, self.mol.symm_orb[ir], sinv))
            dm.append(reduce(numpy.dot, (so.T, dm_ao, so)))
        return dm

    def _init_guess_by_minao(self, mol):
        try:
            e, dm = hf.init_guess_by_minao(self, mol)
            return e, self.symmetrize_den_mat(dm)
        except:
            log.warn(self, 'Fail in generating initial guess from MINAO. ' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

    def _init_guess_by_chkfile(self, mol):
        e, dm = hf.RHF._init_guess_by_chkfile(self, mol)
        if isinstance(dm,numpy.ndarray):
            dm = self.symmetrize_den_mat(dm)
        return e, dm

    def _init_guess_by_atom(self, mol):
        e, dm = hf.RHF._init_guess_by_atom(self, mol)
        if isinstance(dm,numpy.ndarray):
            dm = self.symmetrize_den_mat(dm)
        return e, dm

    def set_mo_occ(self, mo_energy):
        mol = self.mol
        nirrep = mol.symm_orb.__len__()
        mo_e_plain = []
        nocc = []
        nocc_fix = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            if self.irrep_nocc.has_key(irname):
                n = self.irrep_nocc[irname] / 2
                nocc.append(n)
                nocc_fix += n
            else:
                nocc.append(-1)
                mo_e_plain.append(mo_energy[ir])
        nocc_float = mol.nelectron / 2 - nocc_fix
        assert(nocc_float >= 0)
        if nocc_float > 0:
            mo_e_plain = sorted(numpy.hstack(mo_e_plain))
            elumo = mo_e_plain[nocc_float]

        mo_occ = []
        ehomos = []
        elumos = []
        for ir in range(nirrep):
            occ = numpy.zeros_like(mo_energy[ir])
            if nocc[ir] < 0:
                if nocc_float > 0:
                    occ[mo_energy[ir]<elumo] = 2
                    nocc[ir] = int(occ.sum()) / 2
                else:
                    nocc[ir] = 0
            else:
                occ[:nocc[ir]] = 2
            if nocc[ir] > 0:
                ehomos.append(mo_energy[ir][nocc[ir]-1])
            if nocc[ir] < len(mo_energy[ir]):
                elumos.append(mo_energy[ir][nocc[ir]])
            mo_occ.append(occ)
        ehomo = max(ehomos)
        elumo = min(elumos)
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'system HOMO = %.15g, LUMO = %.15g', ehomo, elumo)
            log.debug(self, 'irrep_nocc = %s', nocc)
            dump_mo_energy(mol, mo_energy, nocc, ehomo, elumo)
        return mo_occ

    # full density matrix
    def calc_den_mat(self, mo_coeff, mo_occ):
        nirrep = self.mol.symm_orb.__len__()
        dm = []
        for ir in range(nirrep):
            mo = mo_coeff[ir][:,mo_occ[ir]>0]
            occ = mo_occ[ir][mo_occ[ir]>0]
            dm.append(numpy.dot(mo*occ, mo.T.conj()))
        return dm

    def calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ):
        nirrep = self.mol.symm_orb.__len__()
        sum_mo_energy = 0
        coul_dup = 0
        for ir in range(nirrep):
            sum_mo_energy += numpy.dot(mo_energy[ir], mo_occ[ir])
            coul_dup += lib.trace_ab(dm[ir], vhf[ir])
        log.debug(self, 'E_coul = %.15g', (coul_dup.real * .5))
        e = sum_mo_energy - coul_dup * .5
        return e.real, coul_dup * .5

    def check_dm_converge(self, dm, dm_last, scf_threshold):
        if dm_last is 0:
            return False
        nirrep = self.mol.symm_orb.__len__()
        delta_dm = 0
        dm_tot = 0
        for ir in range(nirrep):
            delta_dm += abs(dm[ir]-dm_last[ir]).sum()
            dm_tot += abs(dm_last[ir]).sum()
        dm_change = delta_dm/dm_tot
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < scf_threshold*1e2

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        import _vhf
        t0 = time.clock()
        nirrep = mol.symm_orb.__len__()
        nao = mol.symm_orb[0].shape[0]
        def dm_so2ao():
            dm_ao = numpy.zeros((nao,nao))
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                dm_ao += reduce(numpy.dot, (so, dm[ir], so.T))
            return dm_ao
        def dm_so2ao_diff():
            if dm_last is 0:
                return dm_so2ao()
            dm_ao = numpy.zeros((nao,nao))
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                dm_ao += reduce(numpy.dot, (so, dm[ir]-dm_last[ir], so.T))
            return dm_ao

        def vhf_ao2so(vhf_ao):
            return symm.symmetrize_matrix(vhf_ao, mol.symm_orb)
        def vhf_ao2so_diff(vhf_ao):
            if vhf_last is 0:
                return vhf_ao2so(vhf_ao)
            vhf = []
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                vhf.append(vhf_last[ir]+reduce(numpy.dot, (so.T, vhf_ao, so)))
            return vhf

        if self.eri_in_memory:
            if self._eri is None:
                self._eri = hf.gen_8fold_eri_sph(mol)
            vj, vk = hf.dot_eri_dm(self._eri, dm_so2ao())
            vhf = vhf_ao2so(vj-vk*.5)
        elif self.direct_scf:
            if dm[0].ndim == 2:
                vj, vk = _vhf.vhf_jk_direct_o2(dm_so2ao_diff(), mol._atm, \
                                                  mol._bas, mol._env, self.opt)
            else:
                vj, vk = hf.get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm_so2ao_diff())
            vhf = vhf_ao2so_diff(vj-vk*.5)
        else:
            if dm[0].ndim == 2:
                vj, vk = _vhf.vhf_jk_direct_o2(dm_so2ao(), mol._atm, \
                                                  mol._bas, mol._env)
            else:
                vj, vk = hf.get_vj_vk(pycint.nr_vhf_o3, mol, dm_so2ao())
            vhf = vhf_ao2so(vj-vk*.5)
        log.debug(self, 'CPU time for vj and vk %.8g sec', (time.clock()-t0))
        return vhf

    def scf(self):
        self.dump_scf_option()
        self.init_direct_scf(self.mol)
        self.scf_conv, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(self.mol, self, self.scf_threshold)
        self.del_direct_scf()

        log.info(self, 'CPU time: %12.2f', time.clock())
        e_nuc = self.mol.nuclear_repulsion()
        log.log(self, 'nuclear repulsion = %.15g', e_nuc)
        if self.scf_conv:
            log.log(self, 'converged electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_scf_cycle)
        log.log(self, 'total molecular energy = %.15g', \
                self.hf_energy + e_nuc)
        self.analyze_scf_result(self.mol, self.mo_energy, self.mo_occ, \
                                self.mo_coeff)

        # transform them to ensure compatible with old code.
        # these transformation should be done last.
        self.mo_occ = numpy.hstack(self.mo_occ)
        mo_energy = numpy.hstack(self.mo_energy)
        o_sort = numpy.argsort(mo_energy[self.mo_occ>0])
        v_sort = numpy.argsort(mo_energy[self.mo_occ==0])
        self.mo_energy = numpy.hstack((mo_energy[self.mo_occ>0][o_sort], \
                                       mo_energy[self.mo_occ==0][v_sort]))
        mo_coeff = so2ao_mo_coeff(self.mol.symm_orb, self.mo_coeff)
        self.mo_coeff = numpy.hstack((mo_coeff[:,self.mo_occ>0][:,o_sort], \
                                      mo_coeff[:,self.mo_occ==0][:,v_sort]))
        nocc = int(self.mo_occ.sum()) / 2
        self.mo_occ[:nocc] = 2
        self.mo_occ[nocc:] = 0
        return self.hf_energy + e_nuc

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        nirrep = mol.symm_orb.__len__()
        if self.verbose >= param.VERBOSE_INFO:
            s = 0
            for ir in range(nirrep):
                if int(mo_occ[ir].sum()) % 2:
                    s ^= mol.irrep_id[ir]
            log.info(self, 'total symmetry = %s', \
                     symm.irrep_name(mol.pgname,s))
            log.info(self, 'occupancy for each irrep:  ' + (' %4s'*nirrep), \
                     *mol.irrep_name)
            noccs = [mo_occ[ir].sum() for ir in range(nirrep)]
            log.info(self, '                           ' + (' %4d'*nirrep), \
                     *noccs)
            log.info(self, '**** MO energy ****')
            e_ir_idx,argsort = argsort_mo_energy(mol, mo_energy)
            for k,j in enumerate(argsort):
                e,ir,i = e_ir_idx[j]
                occ = mo_occ[ir][i]
                log.info(self, 'MO #%d (%s %d), energy= %.15g occ= %g', \
                         k+1, mol.irrep_name[ir], i+1, e, occ)

        c = so2ao_mo_coeff(mol.symm_orb, self.mo_coeff)
        if self.verbose >= param.VERBOSE_DEBUG:
            dump_mo_coeff(mol, c, e_ir_idx, argsort)
        dm = hf.RHF.calc_den_mat(c, numpy.hstack(self.mo_occ))
        self.mulliken_pop(mol, dm, mol.intor_symmetric('cint1e_ovlp_sph'))


class UHF(hf.UHF):
    __doc__ = 'UHF'
    def __init__(self, mol):
        hf.UHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc_alpha = {}
        self.irrep_nocc_beta = {}

    def dump_scf_option(self):
        hf.SCF.dump_scf_option(self)
        log.info(self, 'UHF with symmetry adpated basis')
        float_ir = []
        fix_na = 0
        fix_nb = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if self.irrep_nocc_alpha.has_key(irname):
                fix_na += self.irrep_nocc_alpha[irname]
            else:
                float_ir.append(irname)
            if self.irrep_nocc_beta.has_key(irname):
                fix_nb += self.irrep_nocc_beta[irname]
            else:
                float_ir.append(irname)
        if fix_na+fix_nb > 0:
            log.info(self, 'fix %d electrons in irreps:\n' \
                     '   alpha %s,\n   beta  %s', \
                     fix_na+fix_nb, self.irrep_nocc_alpha.items(), \
                     self.irrep_nocc_beta.items())
            if fix_na+fix_nb > self.mol.nelectron \
               or (self.fix_nelectron_alpha > 0 and \
                   ((fix_na>self.fix_nelectron_alpha) or \
                    (fix_nb+self.fix_nelectron_alpha>self.mol.nelectron))):
                log.error(self, 'number of electrons error in irrep_nocc\n' \
                        '   alpha %s,\n   beta  %s', \
                        self.irrep_nocc_alpha.items(), \
                        self.irrep_nocc_beta.items())
                raise ValueError('irrep_nocc')
        if float_ir:
            if self.fix_nelectron_alpha > 0:
                log.info(self, '%d free alpha electrons ' \
                         '%d free beta electrons in irreps %s', \
                         self.fix_nelectron_alpha-fix_na, \
                         self.mol.nelectron-self.fix_nelectron_alpha-fix_nb, \
                         float_ir)
            else:
                log.info(self, '%d free electrons in irreps %s', \
                         self.mol.nelectron-fix_na-fix_nb, float_ir)
        elif fix_na+fix_nb != self.mol.nelectron:
            log.error(self, 'number of electrons error in irrep_nocc \n' \
                    '   alpha %s,\n   beta  %s', \
                    self.irrep_nocc_alpha.items(), \
                    self.irrep_nocc_beta.items())
            raise ValueError('irrep_nocc')

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        cs_a = []
        es_a = []
        cs_b = []
        es_b = []
        for ir in range(nirrep):
            c, e, info = lapack.dsygv(h[0][ir], s[ir])
            cs_a.append(c)
            es_a.append(e)
            c, e, info = lapack.dsygv(h[1][ir], s[ir])
            cs_b.append(c)
            es_b.append(e)
        return (es_a,es_b), (cs_a,cs_b), info

    def damping(self, s, d, f, factor):
        if factor < 1e-3:
            return f
        else:
            return [hf.UHF.damping(self, s[ir], d[ir], f[ir], factor) \
                    for ir in range(self.mol.symm_orb.__len__())]

    def level_shift(self, s, d, f, factor):
        if factor < 1e-3:
            return f
        else:
            return [hf.UHF.level_shift(self, s[ir], d[ir], f[ir], factor) \
                    for ir in range(self.mol.symm_orb.__len__())]

    def init_diis(self):
        udiis = diis.SCF_DIIS(self)
        udiis.diis_space = self.diis_space
        #udiis.diis_start_cycle = self.diis_start_cycle
        self.old_f = 0
        def scf_diis(cycle, s, d, f):
            nirrep = self.mol.symm_orb.__len__()
            if cycle >= self.diis_start_cycle:
                errvec = []
                ff = []
                for ir in range(nirrep):
                    sdf = reduce(numpy.dot, (s[ir], d[0][ir], f[0][ir]))
                    errvec.append((sdf.T.conj()-sdf).flatten())
                    sdf = reduce(numpy.dot, (s[ir], d[1][ir], f[1][ir]))
                    errvec.append((sdf.T.conj()-sdf).flatten())
                    #sd = numpy.dot(s[ir],d[0][ir])
                    #sdf = numpy.dot(sd,f[0][ir])
                    #errvec.append((sdf.T-numpy.dot(sdf,sd.T)).flatten())
                    #sd = numpy.dot(s[ir],d[1][ir])
                    #sdf = numpy.dot(sd,f[1][ir])
                    #errvec.append((sdf.T-numpy.dot(sdf,sd.T)).flatten())
                    ff.append(f[0][ir].flatten())
                    ff.append(f[1][ir].flatten())
                errvec = numpy.hstack(errvec)
                udiis.err_vec_stack.append(errvec)
                log.debug(self, 'diis-norm(errvec) = %g', \
                          numpy.linalg.norm(errvec))
                if udiis.err_vec_stack.__len__() > udiis.diis_space:
                    udiis.err_vec_stack.pop(0)
                fnew = diis.DIIS.update(udiis, numpy.hstack(ff))
                p0 = 0
                fa = []
                fb = []
                for si in s:
                    n = si.shape[0]
                    fa.append(fnew[p0:p0+n*n].reshape(n,n))
                    p0 += n*n
                    fb.append(fnew[p0:p0+n*n].reshape(n,n))
                    p0 += n*n
                f = (fa,fb)
            if cycle < self.diis_start_cycle-1:
                f = (self.damping(s, d[0], f[0], self.damp_factor), \
                     self.damping(s, d[1], f[1], self.damp_factor))
                f = (self.level_shift(s,d[0],f[0],self.level_shift_factor), \
                     self.level_shift(s,d[1],f[1],self.level_shift_factor))
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = (self.level_shift(s, d[0], f[0], fac), \
                     self.level_shift(s, d[1], f[1], fac))
            return f
        return scf_diis

    def get_hcore(self, mol):
        h = mol.intor_symmetric('cint1e_kin_sph') \
                + mol.intor_symmetric('cint1e_nuc_sph')
        h = symm.symmetrize_matrix(h, mol.symm_orb)
        return (h,h)

    def get_ovlp(self, mol):
        s = mol.intor_symmetric('cint1e_ovlp_sph')
        return symm.symmetrize_matrix(s, mol.symm_orb)

    def make_fock(self, h1e, vhf):
        f_a = []
        f_b = []
        nirrep = self.mol.symm_orb.__len__()
        for ir in range(nirrep):
            c = self.mol.symm_orb[ir]
            f_a.append(h1e[0][ir] + vhf[0][ir])
            f_b.append(h1e[1][ir] + vhf[1][ir])
        return (f_a,f_b)

    def dump_scf_to_chkfile(self, hf_energy, mo_energy, mo_occ, mo_coeff):
        ca = so2ao_mo_coeff(self.mol.symm_orb, mo_coeff[0])
        cb = so2ao_mo_coeff(self.mol.symm_orb, mo_coeff[1])
        hf.dump_scf_to_chkfile(self.mol, self.chkfile, hf_energy, \
                               (numpy.hstack(mo_energy[0]), \
                                numpy.hstack(mo_energy[1])), \
                               (numpy.hstack(mo_occ[0]), \
                                numpy.hstack(mo_occ[1])), \
                               (ca, cb))

    def symmetrize_den_mat(self, dm_ao):
        s0 = self.mol.intor_symmetric('cint1e_ovlp_sph')
        s = symm.symmetrize_matrix(s0, self.mol.symm_orb)
        nirrep = self.mol.symm_orb.__len__()
        dm_a = []
        dm_b = []
        for ir in range(nirrep):
            sinv = numpy.linalg.inv(s[ir])
            so = reduce(numpy.dot, (s0, self.mol.symm_orb[ir], sinv))
            dm_a.append(reduce(numpy.dot, (so.T, dm_ao[0], so)))
            dm_b.append(reduce(numpy.dot, (so.T, dm_ao[1], so)))
        return (dm_a, dm_b)

    def _init_guess_by_minao(self, mol):
        log.info(self, 'initial guess from MINAO')
        try:
            dm = self._init_minao_uhf_dm(mol)
            return 0, self.symmetrize_den_mat(dm)
        except:
            log.warn(self, 'Fail in generating initial guess from MINAO.' \
                     'Use 1e initial guess')
            return self._init_guess_by_1e(mol)

    def _init_guess_by_chkfile(self, mol):
        e, dm = hf.UHF._init_guess_by_chkfile(self, mol)
        if isinstance(dm[0],numpy.ndarray):
            dm = self.symmetrize_den_mat(dm)
        return e, dm

    def _init_guess_by_atom(self, mol):
        e, dm = hf.UHF._init_guess_by_atom(self, mol)
        if isinstance(dm[0], numpy.ndarray):
            dm = self.symmetrize_den_mat(dm)
        return e, dm

    def set_mo_occ(self, mo_energy):
        mol = self.mol
        nirrep = mol.symm_orb.__len__()
        mo_e_plain = [[],[]]
        nocc = [[],[]]
        nocc_fix = [0,0]
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            if self.irrep_nocc_alpha.has_key(irname):
                n = self.irrep_nocc_alpha[irname]
                nocc[0].append(n)
                nocc_fix[0] += n
            else:
                nocc[0].append(-1)
                mo_e_plain[0].append(mo_energy[0][ir])
            if self.irrep_nocc_beta.has_key(irname):
                n = self.irrep_nocc_beta[irname]
                nocc[1].append(n)
                nocc_fix[1] += n
            else:
                nocc[1].append(-1)
                mo_e_plain[1].append(mo_energy[1][ir])

        nocc_float = mol.nelectron - nocc_fix[0] - nocc_fix[1]
        assert(nocc_float >= 0)
        if len(mo_e_plain[0]) > 0:
            ea = sorted(numpy.hstack(mo_e_plain[0]))
        if len(mo_e_plain[1]) > 0:
            eb = sorted(numpy.hstack(mo_e_plain[1]))
        fermia = 0
        fermib = 0
        n_a = 0
        n_b = 0
        if nocc_float > 0:
            if self.fix_nelectron_alpha > 0:
                n_a = self.fix_nelectron_alpha-nocc_fix[0]
                n_b = mol.nelectron - self.fix_nelectron_alpha - nocc_fix[1]
                assert(n_a >= 0)
                assert(n_b >= 0)
                if len(mo_e_plain[0]) > 0:
                    fermia = ea[n_a]
                if len(mo_e_plain[1]) > 0:
                    fermib = eb[n_b]
            else:
                ee = sorted(numpy.hstack((ea,eb)))
                fermia = fermib = ee[nocc_float]
                if len(mo_e_plain[0]) > 0:
                    n_a = int((ea<fermia).sum())
                if len(mo_e_plain[1]) > 0:
                    n_b = int((eb<fermib).sum())
                if n_a+nocc_fix[0] != self.nelectron_alpha:
                    log.info(self, 'change num. alpha/beta electrons' \
                             '%d / %d -> %d / %d', \
                             self.nelectron_alpha,
                             mol.nelectron-self.nelectron_alpha,
                             n_a+nocc_fix[0], n_b+nocc_fix[1])
        self.nelectron_alpha = n_a + nocc_fix[0]
        def get_mocc(nocc, mo_energy, fermi0, nfloat):
            mo_occ = []
            ehomos = []
            elumos = []
            for ir in range(nirrep):
                occ = numpy.zeros_like(mo_energy[ir])
                if nocc[ir] < 0:
                    if nfloat > 0:
                        occ[mo_energy[ir]<fermi0] = 1
                        nocc[ir] = int(occ.sum())
                    else:
                        nocc[ir] = 0
                else:
                    occ[:nocc[ir]] = 1
                mo_occ.append(occ)
                if nocc[ir] > 0:
                    ehomos.append(mo_energy[ir][nocc[ir]-1])
                if nocc[ir] < len(mo_energy[ir]):
                    elumos.append(mo_energy[ir][nocc[ir]])
            return mo_occ, max(ehomos), min(elumos)
        occa, ehomoa, elumoa = get_mocc(nocc[0], mo_energy[0], fermia, n_a)
        occb, ehomob, elumob = get_mocc(nocc[1], mo_energy[1], fermib, n_b)
        if self.verbose >= param.VERBOSE_DEBUG:
            ehomo = max(ehomoa, ehomob)
            elumo = min(elumoa, elumob)
            log.debug(self, 'system HOMO = %.15g, LUMO = %.15g', \
                      ehomo, elumo)
            if ehomo > elumo:
                log.debug(self, '!! alpha/beta HOMO > alpha/beta LUMO')
                log.debug(self, '   %.9g / %.9g > %.9g / %.9g', \
                          ehomoa, ehomob, elumoa, elumob)
            log.debug(self, 'alpha irrep_nocc = %s', nocc[0])
            log.debug(self, 'beta  irrep_nocc = %s', nocc[1])
            dump_mo_energy(mol, mo_energy[0], nocc[0], ehomo, elumo, 'alpha-')
            dump_mo_energy(mol, mo_energy[1], nocc[1], ehomo, elumo, 'beta-')
        return (occa, occb)

    # full density matrix for RHF
    def calc_den_mat(self, mo_coeff, mo_occ):
        nirrep = self.mol.symm_orb.__len__()
        nao = self.mol.symm_orb[0].shape[0]
        dm_a = []
        dm_b = []
        for ir in range(nirrep):
            mo = mo_coeff[0][ir][:,mo_occ[0][ir]>0]
            occ = mo_occ[0][ir][mo_occ[0][ir]>0]
            dm_a.append(numpy.dot(mo*occ, mo.T.conj()))
            mo = mo_coeff[1][ir][:,mo_occ[1][ir]>0]
            occ = mo_occ[1][ir][mo_occ[1][ir]>0]
            dm_b.append(numpy.dot(mo*occ, mo.T.conj()))
        return (dm_a,dm_b)

    def calc_tot_elec_energy(self, vhf, dm, mo_energy, mo_occ):
        nirrep = self.mol.symm_orb.__len__()
        sum_mo_energy = 0
        coul_dup = 0
        for ir in range(nirrep):
            sum_mo_energy += numpy.dot(mo_energy[0][ir], mo_occ[0][ir])
            sum_mo_energy += numpy.dot(mo_energy[1][ir], mo_occ[1][ir])
            coul_dup += lib.trace_ab(dm[0][ir], vhf[0][ir]) \
                      + lib.trace_ab(dm[1][ir], vhf[1][ir])
        log.debug(self, 'E_coul = %.15g', (coul_dup.real * .5))
        e = sum_mo_energy - coul_dup * .5
        return e.real, coul_dup * .5

    def check_dm_converge(self, dm, dm_last, scf_threshold):
        if dm_last is 0:
            return False
        nirrep = self.mol.symm_orb.__len__()
        delta_dm = 0
        dm_tot = 0
        for ir in range(nirrep):
            delta_dm += abs(dm[0][ir]-dm_last[0][ir]).sum() \
                      + abs(dm[1][ir]-dm_last[1][ir]).sum()
            dm_tot += abs(dm_last[0][ir]).sum() + abs(dm_last[1][ir]).sum()
        dm_change = delta_dm/dm_tot
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < scf_threshold*1e2

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        import _vhf
        t0 = time.clock()
        nirrep = mol.symm_orb.__len__()
        nao = mol.symm_orb[0].shape[0]
        def dm_so2ao():
            dm_ao = numpy.zeros((2,nao,nao))
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                dm_ao[0] += reduce(numpy.dot, (so, dm[0][ir], so.T))
                dm_ao[1] += reduce(numpy.dot, (so, dm[1][ir], so.T))
            return dm_ao
        def dm_so2ao_diff():
            if dm_last is 0:
                return dm_so2ao()
            dm_ao = numpy.zeros((2,nao,nao))
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                dm_ao[0] +=reduce(numpy.dot,(so,dm[0][ir]-dm_last[0][ir],so.T))
                dm_ao[1] +=reduce(numpy.dot,(so,dm[1][ir]-dm_last[1][ir],so.T))
            return dm_ao

        def vhf_ao2so(vhf_ao):
            return symm.symmetrize_matrix(vhf_ao, mol.symm_orb)
        def vhf_ao2so_diff(vhf_ao):
            if vhf_last is 0:
                return vhf_ao2so(vhf_ao)
            vhf = []
            for ir in range(nirrep):
                so = mol.symm_orb[ir]
                vhf.append(vhf_last[ir]+reduce(numpy.dot, (so.T, vhf_ao, so)))
            return vhf

        if self.eri_in_memory:
            if self._eri is None:
                self._eri = hf.gen_8fold_eri_sph(mol)
            dm_ao = dm_so2ao()
            vj0, vk0 = hf.dot_eri_dm(self._eri, dm_ao[0])
            vj1, vk1 = hf.dot_eri_dm(self._eri, dm_ao[1])
            vhf = (vhf_ao2so(vj0+vj1-vk0), vhf_ao2so(vj0+vj1-vk1))
        elif self.direct_scf:
            vj, vk = hf.get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm_so2ao_diff())
            vhf = (vhf_ao2so_diff(vj[0]+vj[1]-vk[0]), \
                   vhf_ao2so_diff(vj[0]+vj[1]-vk[1]))
        else:
            vj, vk = hf.get_vj_vk(pycint.nr_vhf_direct_o3, mol, dm_so2ao_diff())
            vhf = (vhf_ao2so(vj[0]+vj[1]-vk[0]), vhf_ao2so(vj[0]+vj[1]-vk[1]))
        log.debug(self, 'CPU time for vj and vk %.8g sec', (time.clock()-t0))
        return vhf

    def break_spin_sym(self, mol, mo_coeff):
        if self.break_symmetry:
            mo_coeff[1][0][:,0] = 0
        return mo_coeff

    def scf(self):
        self.dump_scf_option()
        self.init_direct_scf(self.mol)
        self.scf_conv, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(self.mol, self, self.scf_threshold)
        self.del_direct_scf()
        if self.nelectron_alpha * 2 < self.mol.nelectron:
            self.mo_coeff = (self.mo_coeff[1], self.mo_coeff[0])
            self.mo_occ = (self.mo_occ[1], self.mo_occ[0])
            self.mo_energy = (self.mo_energy[1], self.mo_energy[0])

        log.info(self, 'CPU time: %12.2f', time.clock())
        e_nuc = self.mol.nuclear_repulsion()
        log.log(self, 'nuclear repulsion = %.15g', e_nuc)
        if self.scf_conv:
            log.log(self, 'converged electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_scf_cycle)
        log.log(self, 'total molecular energy = %.15g', \
                self.hf_energy + e_nuc)
        self.analyze_scf_result(self.mol, self.mo_energy, self.mo_occ, \
                                self.mo_coeff)

        self.mo_occ = (numpy.hstack(self.mo_occ[0]), \
                       numpy.hstack(self.mo_occ[1]))
        ea = numpy.hstack(self.mo_energy[0])
        eb = numpy.hstack(self.mo_energy[0])
        oa_sort = numpy.argsort(ea[self.mo_occ[0]>0])
        va_sort = numpy.argsort(ea[self.mo_occ[0]==0])
        ob_sort = numpy.argsort(eb[self.mo_occ[1]>0])
        vb_sort = numpy.argsort(eb[self.mo_occ[1]==0])
        self.mo_energy = (numpy.hstack((ea[self.mo_occ[0]>0][oa_sort], \
                                        ea[self.mo_occ[0]==0][va_sort])), \
                          numpy.hstack((eb[self.mo_occ[1]>0][ob_sort], \
                                        eb[self.mo_occ[1]==0][vb_sort])))
        ca = so2ao_mo_coeff(self.mol.symm_orb,self.mo_coeff[0])
        cb = so2ao_mo_coeff(self.mol.symm_orb,self.mo_coeff[1])
        self.mo_coeff = (numpy.hstack((ca[:,self.mo_occ[0]>0][:,oa_sort], \
                                       ca[:,self.mo_occ[0]==0][:,va_sort])), \
                         numpy.hstack((cb[:,self.mo_occ[1]>0][:,ob_sort], \
                                       cb[:,self.mo_occ[1]==0][:,vb_sort])))
        nocc_a = int(self.mo_occ[0].sum())
        nocc_b = int(self.mo_occ[1].sum())
        self.mo_occ[0][:nocc_a] = 1
        self.mo_occ[0][nocc_a:] = 0
        self.mo_occ[1][:nocc_b] = 1
        self.mo_occ[1][nocc_b:] = 0
        return self.hf_energy + e_nuc

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        nirrep = mol.symm_orb.__len__()
        if self.verbose >= param.VERBOSE_INFO:
            s = 0
            for ir in range(nirrep):
                if int(mo_occ[0][ir].sum()+mo_occ[1][ir].sum()) % 2:
                    s ^= mol.irrep_id[ir]
            log.info(self, 'total symmetry = %s', \
                     symm.irrep_name(mol.pgname,s))
            log.info(self, 'alpha occupancy for each irrep:  '+(' %4s'*nirrep), \
                     *mol.irrep_name)
            noccs = [self.mo_occ[0][ir].sum() for ir in range(nirrep)]
            log.info(self, '                                 '+(' %4d'*nirrep), \
                     *noccs)
            log.info(self, 'beta  occupancy for each irrep:  '+(' %4s'*nirrep), \
                     *mol.irrep_name)
            noccs = [self.mo_occ[1][ir].sum() for ir in range(nirrep)]
            log.info(self, '                                 '+(' %4d'*nirrep), \
                     *noccs)

        occa = numpy.hstack(mo_occ[0])
        occb = numpy.hstack(mo_occ[1])
        ca = so2ao_mo_coeff(mol.symm_orb,mo_coeff[0])
        cb = so2ao_mo_coeff(mol.symm_orb,mo_coeff[1])
        ss, s = hf.spin_square(mol, ca[:,occa>0], cb[:,occb>0])
        log.info(self, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

        if self.verbose >= param.VERBOSE_INFO:
            log.info(self, '**** MO energy ****')
            e_ir_a,sorta = argsort_mo_energy(mol, mo_energy[0])
            for k,j in enumerate(sorta):
                e,ir,i = e_ir_a[j]
                occ = mo_occ[0][ir][i]
                log.info(self, 'alpha MO #%d (%s %d), energy= %.15g occ= %g', \
                         k+1, mol.irrep_name[ir], i+1, e, occ)
            e_ir_b,sortb = argsort_mo_energy(mol, self.mo_energy[1])
            for k,j in enumerate(sortb):
                e,ir,i = e_ir_b[j]
                occ = self.mo_occ[1][ir][i]
                log.info(self, 'beta MO #%d (%s %d), energy= %.15g occ= %g', \
                         k+1, mol.irrep_name[ir], i+1, e, occ)

        if self.verbose >= param.VERBOSE_DEBUG:
            dump_mo_coeff(mol, ca, e_ir_a, sorta, 'alpha')
            dump_mo_coeff(mol, cb, e_ir_b, sortb, 'beta')

        dm = hf.UHF.calc_den_mat((ca,cb), (occa, occb))
        self.mulliken_pop(mol, dm, mol.intor_symmetric('cint1e_ovlp_sph'))

def map_rhf_to_uhf(mol, rhf):
    assert(isinstance(rhf, RHF))
    uhf = UHF(mol)
    uhf.verbose               = rhf.verbose
    uhf.mo_energy             = numpy.array((rhf.mo_energy,rhf.mo_energy))
    uhf.mo_coeff              = numpy.array((rhf.mo_coeff,rhf.mo_coeff))
    uhf.mo_occ                = numpy.array((rhf.mo_occ,rhf.mo_occ))
    uhf.hf_energy             = rhf.hf_energy
    uhf.diis_space            = rhf.diis_space
    uhf.diis_start_cycle      = rhf.diis_start_cycle
    uhf.damp_factor           = rhf.damp_factor
    uhf.level_shift_factor    = rhf.level_shift_factor
    uhf.scf_conv              = rhf.scf_conv
    uhf.direct_scf            = rhf.direct_scf
    uhf.direct_scf_threshold  = rhf.direct_scf_threshold

    uhf.chkfile               = rhf.chkfile
    self.fout                 = rhf.fout
    self.scf_threshold        = rhf.scf_threshold
    self.max_scf_cycle        = rhf.max_scf_cycle
    return uhf

def spin_square(mol, mo_a, mo_b):
    # S^2 = S+ * S- + S- * S+ + Sz * Sz
    # S+ = \sum_i S_i+ ~ effective for all beta occupied orbitals
    # S- = \sum_i S_i- ~ effective for all alpha occupied orbitals
    # S+ * S- ~ sum of nocc_a * nocc_b couplings
    # Sz = Msz^2
    nocc_a = mo_a.shape[1]
    nocc_b = mo_b.shape[1]
    ovlp = mol.intor_symmetric('cint1e_ovlp_sph')
    s = reduce(numpy.dot, (mo_a.T, ovlp, mo_b))
    ssx = ssy = (nocc_a+nocc_b)*.25 - 2*(s**2).sum()*.25
    ssz = (nocc_b-nocc_a)**2 * .25
    ss = ssx + ssy + ssz
    #log.debug(mol, 's_x^2 = %.9g, s_y^2 = %.9g, s_z^2 = %.9g', ssx,ssy,ssz)
    s = numpy.sqrt(ss+.25) - .5
    multip = s*2+1
    return ss, multip
