#!/usr/bin/env python
#
# File: dhf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
NMR shielding for Dirac Hartree-Fock
'''

import cPickle as pickle
import time

import numpy
from pyscf import gto
from pyscf.lib import logger as log
from pyscf.lib import parameters as param
from pyscf.lib import pycint
from pyscf import lib
from pyscf import scf
import hf

class MSC(hf.MSC):
    __doc__ = 'magnetic shielding constants'
    def __init__(self, scf_method, restart=False):
        hf.MSC.__init__(self, scf_method, restart)
        self.is_giao = True
        self.is_cpscf = True
        self.MB = self.rmb

    def dump_flags(self):
        hf.MSC.dump_flags(self)
        log.info(self, 'MB basis = %s', self.MB.__doc__)

    def msc(self):
        cput0 = (time.clock(), time.time())
        if self.verbose >= param.VERBOSE_INFO:
            self.dump_flags()

        if not self.is_giao:
            self.mol.set_common_origin(self.gauge_orig)

        res = self.para(self.mol, self.scf)
        msc_para, para_pos, para_neg, para_occ = [i*1e6/param.LIGHTSPEED**2 for i in res]
        msc_dia = self.dia(self.mol, self.scf) * 1e6/param.LIGHTSPEED**2
        e11 = msc_para + msc_dia

        log.timer(self, 'NMR', *cput0)
        if self.verbose > param.VERBOSE_QUITE:
            fout = self.stdout
            for tot,d,p,p0,ppos,pneg,atom in \
                    zip(e11, msc_dia, msc_para, para_occ, \
                    para_pos, para_neg, self.shielding_nuc):
                fout.write('total MSC of atom %d %s\n' \
                           % (atom, self.mol.symbol_of_atm(atom-1)))
                fout.write('B_x %s\n' % str(tot[0]))
                fout.write('B_y %s\n' % str(tot[1]))
                fout.write('B_z %s\n' % str(tot[2]))
                fout.write('dia-magnetism\n')
                fout.write('B_x %s\n' % str(d[0]))
                fout.write('B_y %s\n' % str(d[1]))
                fout.write('B_z %s\n' % str(d[2]))
                fout.write('para-magnetism\n')
                fout.write('B_x %s\n' % str(p[0]))
                fout.write('B_y %s\n' % str(p[1]))
                fout.write('B_z %s\n' % str(p[2]))
                if self.verbose >= param.VERBOSE_INFO:
                    fout.write('INFO: occ part of para-magnetism\n')
                    fout.write('INFO: B_x %s\n' % str(p0[0]))
                    fout.write('INFO: B_y %s\n' % str(p0[1]))
                    fout.write('INFO: B_z %s\n' % str(p0[2]))
                    fout.write('INFO: vir-pos part of para-magnetism\n')
                    fout.write('INFO: B_x %s\n' % str(ppos[0]))
                    fout.write('INFO: B_y %s\n' % str(ppos[1]))
                    fout.write('INFO: B_z %s\n' % str(ppos[2]))
                    fout.write('INFO: vir-neg part of para-magnetism\n')
                    fout.write('INFO: B_x %s\n' % str(pneg[0]))
                    fout.write('INFO: B_y %s\n' % str(pneg[1]))
                    fout.write('INFO: B_z %s\n' % str(pneg[2]))
        self.stdout.flush()
        return e11

    def dia(self, mol, scf0):
        '''Dia-magnetic'''
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        msc_dia = []
        dm0 = scf0.calc_den_mat(scf0.mo_coeff, scf0.mo_occ)
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            if self.MB == self.rmb and self.is_giao:
                t11 = mol.intor('cint1e_giao_sa10sa01', 9)
                t11 += mol.intor('cint1e_spgsa01', 9)
            elif self.MB == self.rmb and not self.is_giao:
                t11 = mol.intor('cint1e_cg_sa10sa01', 9)
            elif self.is_giao:
                t11 = mol.intor('cint1e_spgsa01', 9)
            else:
                t11 = 0
            h11 = numpy.zeros((9, n4c, n4c), complex)
            for i in range(9):
                h11[i,n2c:,:n2c] = t11[i] * .5
                h11[i,:n2c,n2c:] = t11[i].conj().T * .5
            a11 = [numpy.real(lib.trace_ab(dm0, x)) for x in h11]
            # param.MI_POS XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
            #           => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
            msc_dia.append(a11)
        return numpy.array(msc_dia).reshape(self.shielding_nuc.__len__(), 3, 3)

    def para(self, mol, scf0):
        '''Para-magnetism'''
        t0 = (time.clock(), time.time())
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        h1, s1 = self.get_h10_s10(mol, scf0)
        t0 = log.timer(self, 'h10', *t0)
        if self.is_cpscf:
            self.mo_e10, self.mo10 = hf.solve_cpscf(mol, scf0, self.v_ind, h1,
                                                    s1, self.max_cycle,
                                                    self.threshold)
        else:
            self.mo_e10, self.mo10 = hf.solve_ucpscf(mol, scf0, h1, s1)
        t0 = log.timer(self, 'solving CPSCF/UCPSCF', *t0)

        msc_para = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_neg = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        para_occ = numpy.zeros((self.shielding_nuc.__len__(),3,3))
        h01 = numpy.zeros((3, n4c, n4c), complex)
        for n,nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            t01 = mol.intor('cint1e_sa01sp', 3)
            for m in range(3):
                h01[m,:n2c,n2c:] = .5 * t01[m]
                h01[m,n2c:,:n2c] = .5 * t01[m].conj().T
            h01_mo = self._mat_ao2mo(h01, scf0.mo_coeff, scf0.mo_occ)
            for b in range(3):
                for m in range(3):
                    # + c.c.
                    p = 2 * numpy.real(self._ab_diagonal(h01_mo[m],
                                                         self.mo10[b].conj().T))
                    msc_para[n,b,m] = p.sum()
                    para_neg[n,b,m] = p[scf0.mo_energy<-1.5*c**2].sum()
                    para_occ[n,b,m] = p[scf0.mo_occ>0].sum()
        para_pos = msc_para - para_neg - para_occ
        t0 = log.timer(self, 'h01', *t0)
        return msc_para, para_pos, para_neg, para_occ

    @lib.omnimethod
    def calc_den_mat_1(self, mo1, mo0, occ):
        m = mo0[:,occ>0]
        dm1 = []
        for i in range(3):
            mo1_ao = numpy.dot(mo0, mo1[i])
            tmp = numpy.dot(mo1_ao, m.T.conj())
            dm1.append(tmp + tmp.T.conj())
        #print 'dd',abs(dm1[0].diagonal()).sum(),dm1[0].sum()
        #print abs(dm1[0] + dm1[1]*1j).sum()
        #numpy.set_printoptions(3)
        #print dm1
        #print mo1[0]
        return numpy.array(dm1)

    def get_h10_s10(self, mol, scf0):
        '''Fock_{MB}+Fock_{GIAO}, S_{MB}+S_{GIAO}'''
#TODO the uncouupled force
        h1_ao, s1_ao = self.MB(mol, scf0)
        if self.is_giao:
            hg_ao, sg_ao = self.add_giao(mol, scf0)
            h1_ao += hg_ao
            s1_ao += sg_ao

        h1 = self._mat_ao2mo(h1_ao, scf0.mo_coeff, scf0.mo_occ)
        s1 = self._mat_ao2mo(s1_ao, scf0.mo_coeff, scf0.mo_occ)
        return h1, s1

    def add_giao(self, mol, scf0):
        '''GIAO'''
        t0 = (time.clock(), time.time())
        log.info(self, 'first order Fock matrix / GIAOs')
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed

        sg = mol.intor('cint1e_govlp', 3)
        tg = mol.intor('cint1e_spgsp', 3)
        vg = mol.intor('cint1e_gnuc', 3)
        wg = mol.intor('cint1e_spgnucsp', 3)

        s1 = numpy.zeros((3, n4c, n4c), complex)
        if self.restart:
            h1 = scf.load_chkfile_key(self.scf.chkfile, 'vhf_GIAO')
            log.info(self, 'restore vhf_GIAO from chkfile')
        else:
            dm0 = scf0.calc_den_mat(scf0.mo_coeff, scf0.mo_occ)
            vj, vk = scf.hf.get_vj_vk(pycint.rkb_giao_vhf_coul, mol, dm0)
            h1 = vj - vk
            if self.scf.with_gaunt:
                vj, vk = scf.hf.get_vj_vk(pycint.rkb_giao_vhf_gaunt, mol, dm0)
                h1 += vj - vk
            scf.dump_chkfile_key(self.scf.chkfile, 'vhf_GIAO', h1)

        for i in range(3):
            h1[i,:n2c,:n2c] += vg[i]
            h1[i,n2c:,:n2c] += tg[i] * .5
            h1[i,:n2c,n2c:] += tg[i].conj().T * .5
            h1[i,n2c:,n2c:] += wg[i]*(.25/c**2) - tg[i]*.5
            s1[i,:n2c,:n2c] = sg[i]
            s1[i,n2c:,n2c:] = tg[i] * (.25/c**2)
        log.timer(self, 'GIAO', *t0)
        return h1, s1

    def rkb(self, mol, scf0):
        '''RKB basis'''
        log.info(self, 'first order Fock matrix / RKB')
        t0 = (time.clock(), time.time())
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        if self.is_giao:
            t1 = mol.intor('cint1e_giao_sa10sp', 3)
        else:
            t1 = mol.intor('cint1e_cg_sa10sp', 3)
        h1 = numpy.zeros((3, n4c, n4c), complex)
        s1 = numpy.zeros((3, n4c, n4c), complex)
        for i in range(3):
            h1[i,:n2c,n2c:] += .5 * t1[i]
            h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
        log.timer(self, 'RKB h10', *t0)
        return h1, s1

    def rmb(self, mol, scf0):
        '''RMB basis'''
        log.info(self, 'first order Fock matrix / RMB')
        t0 = (time.clock(), time.time())
        n4c = mol.num_4C_function()
        n2c = n4c / 2
        c = mol.light_speed
        if self.is_giao:
            t1 = mol.intor('cint1e_giao_sa10sp', 3)
            v1 = mol.intor('cint1e_giao_sa10nucsp', 3)
        else:
            t1 = mol.intor('cint1e_cg_sa10sp', 3)
            v1 = mol.intor('cint1e_cg_sa10nucsp', 3)

        s1 = numpy.zeros((3, n4c, n4c), complex)
        if self.restart:
            h1 = scf.load_chkfile_key(self.scf.chkfile, 'vhf_RMB')
            log.info(self, 'restore vhf_RMB from chkfile')
        else:
            dm0 = scf0.calc_den_mat(scf0.mo_coeff, scf0.mo_occ)
            if self.is_giao:
                vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_coul, mol, dm0)
                h1 = vj - vk
                if self.scf.with_gaunt:
                    vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_gaunt, mol, dm0)
                    h1 += vj - vk
            else:
                vj,vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_coul, mol, dm0)
                h1 = vj - vk
                if self.scf.with_gaunt:
                    vj, vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_gaunt, mol, dm0)
                    h1 += vj - vk
            scf.dump_chkfile_key(self.scf.chkfile, 'vhf_RMB', h1)

        for i in range(3):
            t1cc = t1[i] + t1[i].conj().T
            h1[i,:n2c,n2c:] += t1cc * .5
            h1[i,n2c:,:n2c] += t1cc * .5
            h1[i,n2c:,n2c:] +=-t1cc * .5 + (v1[i]+v1[i].conj().T) * (.25/c**2)
            s1[i,n2c:,n2c:] = t1cc * (.25/c**2)
        log.timer(self, 'RMB h10', *t0)
        return h1, s1

    # cannot use NR-HF v_ind.  NR-HF anti-symmetrize v_ao. But v_ao of DHF is
    # time-reversal anti-symmetric
    def v_ind(self, mol, mo1):
        '''Induced potential'''
        scf0 = self.scf
        dm1 = self.calc_den_mat_1(mo1, scf0.mo_coeff, scf0.mo_occ)
        v_ao = self.scf.get_veff(mol, dm1)
        return self._mat_ao2mo(v_ao, scf0.mo_coeff, scf0.mo_occ)


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = None#'out_dhf'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.etb = {
        'He': { 'max_l' : 0
              , 's'     : (2, 2, 1.8)
              , 'p'     : (0, 1, 1.8)
              , 'd'     : (0, 1, 1.8)
              , 'f'     : (0,0,0)
              , 'g'     : (0,0,0)}, }
    mol.basis = {
        'He': [(0, 0, (1., 1.)),
               (0, 0, (3., 1.)),
               (1, 0, (1., 1.)), ]}
    mol.build()

    from pyscf.scf import dhf
    mf = dhf.UHF(mol)
    mf.scf()
    nmr = MSC(mf)
    nmr.MB = nmr.rmb
    nmr.is_cpscf = True
    msc = nmr.msc() # 63.0196725
