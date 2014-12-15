#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
NMR shielding of Dirac Hartree-Fock
'''

import time
import numpy
import pyscf.lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import pyscf.lib.pycint as pycint
import pyscf.scf as scf
import pyscf.scf._vhf as _vhf
import hf

class NMR(hf.NMR):
    __doc__ = 'magnetic shielding constants'
    def __init__(self, scf_method):
        hf.NMR.__init__(self, scf_method)
        self.giao = True
        self.cphf = True
        self.mb = 'RMB'
        self._keys = self._keys | set(['mb'])

    def dump_flags(self):
        hf.NMR.dump_flags(self)
        log.info(self, 'MB basis = %s', self.mb)

    def shielding(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.dump_flags()
        self.mol.check_sanity(self)

        mf = self._scf
        if mo1 is None:
            self.mo10 = self.solve_mo10()[1]
        else:
            self.mo10 = mo1

        res = self.para(self.mol, self.mo10, mf.mo_coeff, mf.mo_occ)
        fac2ppm = 1e6/param.LIGHTSPEED**2
        msc_para, para_pos, para_neg, para_occ = [x*fac2ppm for x in res]
        msc_dia = self.dia(self.mol, mf.mo_coeff, mf.mo_occ) * fac2ppm
        e11 = msc_para + msc_dia

        log.timer(self, 'NMR shielding', *cput0)
        if self.verbose > param.VERBOSE_QUIET:
            for i, atm_id in enumerate(self.shielding_nuc):
                self.write(e11[i], \
                           '\ntotal shielding of atom %d %s' \
                           % (atm_id, self.mol.symbol_of_atm(atm_id-1)))
                self.write(msc_dia[i], 'dia-magnetism')
                self.write(msc_para[i], 'para-magnetism')
                if self.verbose >= param.VERBOSE_INFO:
                    self.write(para_occ[i], 'occ part of para-magnetism')
                    self.write(para_pos[i], 'vir-pos part of para-magnetism')
                    self.write(para_neg[i], 'vir-neg part of para-magnetism')
        self.stdout.flush()
        return e11

    def dia(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if not self.giao:
            mol.set_common_origin(self.gauge_orig)

        t0 = (time.clock(), time.time())
        n4c = mo_coeff.shape[0]
        n2c = n4c / 2
        msc_dia = []
        dm0 = scf0.make_rdm1(mo_coeff, mo_occ)
        for n, nuc in enumerate(self.shielding_nuc):
            mol.set_rinv_by_atm_id(nuc)
            if self.mb.upper() == 'RMB' and self.giao:
                t11 = mol.intor('cint1e_giao_sa10sa01', 9)
                t11 += mol.intor('cint1e_spgsa01', 9)
            elif self.mb.upper() == 'RMB' and not self.giao:
                t11 = mol.intor('cint1e_cg_sa10sa01', 9)
            elif self.giao:
                t11 = mol.intor('cint1e_spgsa01', 9)
            else:
                t11 = 0
            h11 = numpy.zeros((9, n4c, n4c), complex)
            for i in range(9):
                h11[i,n2c:,:n2c] = t11[i] * .5
                h11[i,:n2c,n2c:] = t11[i].conj().T * .5
            a11 = [numpy.real(numpy.einsum('ij,ji', dm0, x)) for x in h11]
            # param.MI_POS XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
            #           => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
            msc_dia.append(a11)
        t0 = log.timer(self, 'h11', *t0)
        return numpy.array(msc_dia).reshape(-1, 3, 3)

    def para(self, mol, mo10, mo_coeff, mo_occ):
        t0 = (time.clock(), time.time())
        n4c = mo_coeff.shape[1]
        n2c = n4c / 2
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
            h01_mo = hf._mat_ao2mo(h01, mo_coeff, mo_occ)
            for b in range(3):
                for m in range(3):
                    # + c.c.
                    p = numpy.einsum('ij,ji->i', h01_mo[m],
                                     self.mo10[b].T.conj()).real * 2
                    msc_para[n,b,m] = p.sum()
                    para_neg[n,b,m] = p[:n2c].sum()
                    para_occ[n,b,m] = p[mo_occ>0].sum()
        para_pos = msc_para - para_neg - para_occ
        t0 = log.timer(self, 'h01', *t0)
        return msc_para, para_pos, para_neg, para_occ

    @pyscf.lib.omnimethod
    def make_rdm1_1(self, mo1, mo0, occ):
        ''' DM^1 = C_occ^1 C_occ^{0,dagger} + c.c.'''
        m = mo0[:,occ>0]
        dm1 = []
        for i in range(3):
            mo1_ao = numpy.dot(mo0, mo1[i])
            tmp = numpy.dot(mo1_ao, m.T.conj())
            dm1.append(tmp + tmp.T.conj())
        return numpy.array(dm1)

    def make_h10(self, mol, mo_coeff, mo_occ, scf0=None):
        if self.mb.upper() == 'RMB':
            h1 = self.make_h10rmb(mol, mo_coeff, mo_occ, scf0)
        else: # RKB
            h1 = self.make_h10rkb(mol, mo_coeff, mo_occ, scf0)
        if self.giao:
            h1 += self.make_h10giao(mol, mo_coeff, mo_occ, scf0)
        return h1

    def make_s10(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if not self.giao:
            mol.set_common_origin(self.gauge_orig)
        n4c = mo_coeff.shape[0]
        n2c = n4c / 2
        c = mol.light_speed
        s1 = numpy.zeros((3, n4c, n4c), complex)
        if self.mb.upper() == 'RMB':
            if self.giao:
                t1 = mol.intor('cint1e_giao_sa10sp', 3)
            else:
                t1 = mol.intor('cint1e_cg_sa10sp', 3)
            for i in range(3):
                t1cc = t1[i] + t1[i].conj().T
                s1[i,n2c:,n2c:] = t1cc * (.25/c**2)

        if self.giao:
            sg = mol.intor('cint1e_govlp', 3)
            tg = mol.intor('cint1e_spgsp', 3)
            s1[:,:n2c,:n2c] += sg
            s1[:,n2c:,n2c:] += tg * (.25/c**2)
        return hf._mat_ao2mo(s1, mo_coeff, mo_occ)

    def make_h10giao(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        t0 = (time.clock(), time.time())
        log.info(self, 'first order Fock matrix / GIAOs')
        n4c = mo_coeff.shape[0]
        n2c = n4c / 2
        c = mol.light_speed

        sg = mol.intor('cint1e_govlp', 3)
        tg = mol.intor('cint1e_spgsp', 3)
        vg = mol.intor('cint1e_gnuc', 3)
        wg = mol.intor('cint1e_spgnucsp', 3)

        dm0 = scf0.make_rdm1(mo_coeff, mo_occ)
        vj, vk = _call_giao_vhf1(mol, dm0)
        h1 = vj - vk
        if scf0.with_gaunt:
            vj, vk = scf.hf.get_vj_vk(pycint.rkb_giao_vhf_gaunt, mol, dm0)
            h1 += vj - vk
        pyscf.scf.chkfile.dump(self.chkfile, 'nmr/vhf_GIAO', h1)

        for i in range(3):
            h1[i,:n2c,:n2c] += vg[i]
            h1[i,n2c:,:n2c] += tg[i] * .5
            h1[i,:n2c,n2c:] += tg[i].conj().T * .5
            h1[i,n2c:,n2c:] += wg[i]*(.25/c**2) - tg[i]*.5
        log.timer(self, 'GIAO', *t0)
        return hf._mat_ao2mo(h1, mo_coeff, mo_occ)

    def make_h10rkb(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if not self.giao:
            mol.set_common_origin(self.gauge_orig)
        log.debug(self, 'first order Fock matrix / RKB')
        t0 = (time.clock(), time.time())
        n4c = mo_coeff.shape[0]
        n2c = n4c / 2
        if self.giao:
            t1 = mol.intor('cint1e_giao_sa10sp', 3)
        else:
            t1 = mol.intor('cint1e_cg_sa10sp', 3)
        h1 = numpy.zeros((3, n4c, n4c), complex)
        for i in range(3):
            h1[i,:n2c,n2c:] += .5 * t1[i]
            h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
        log.timer(self, 'RKB h10', *t0)
        return hf._mat_ao2mo(h1, mo_coeff, mo_occ)

#TODO the uncouupled force
    def make_h10rmb(self, mol, mo_coeff, mo_occ, scf0=None):
        if scf0 is None:
            scf0 = self._scf
        if not self.giao:
            mol.set_common_origin(self.gauge_orig)
        log.debug(self, 'first order Fock matrix / RMB')
        t0 = (time.clock(), time.time())
        n4c = mo_coeff.shape[0]
        n2c = n4c / 2
        c = mol.light_speed
        if self.giao:
            t1 = mol.intor('cint1e_giao_sa10sp', 3)
            v1 = mol.intor('cint1e_giao_sa10nucsp', 3)
        else:
            t1 = mol.intor('cint1e_cg_sa10sp', 3)
            v1 = mol.intor('cint1e_cg_sa10nucsp', 3)

        dm0 = scf0.make_rdm1(mo_coeff, mo_occ)
        if self.giao:
            #vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_coul, mol, dm0)
            vj, vk = _call_rmb_vhf1(mol, dm0, 'giao')
            h1 = vj - vk
            if scf0.with_gaunt:
                vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_gaunt, mol, dm0)
                h1 += vj - vk
        else:
            #vj,vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_coul, mol, dm0)
            vj, vk = _call_rmb_vhf1(mol, dm0, 'cg')
            h1 = vj - vk
            if scf0.with_gaunt:
                vj, vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_gaunt, mol, dm0)
                h1 += vj - vk
        pyscf.scf.chkfile.dump(self.chkfile, 'nmr/vhf_RMB', h1)

        for i in range(3):
            t1cc = t1[i] + t1[i].conj().T
            h1[i,:n2c,n2c:] += t1cc * .5
            h1[i,n2c:,:n2c] += t1cc * .5
            h1[i,n2c:,n2c:] +=-t1cc * .5 + (v1[i]+v1[i].conj().T) * (.25/c**2)
        log.timer(self, 'RMB h10', *t0)
        return hf._mat_ao2mo(h1, mo_coeff, mo_occ)

def _call_rmb_vhf1(mol, dm, key='giao'):
    c1 = .5/mol.light_speed
    n2c = dm.shape[0] / 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vx = _vhf.rdirect_mapdm('cint2e_'+key+'_sa10sp1spsp2', 'CVHFdot_rs2kl',
                            ('CVHFrs2kl_ji_s2kl', 'CVHFrs2kl_lk_s1ij',
                             'CVHFrs2kl_jk_s1il', 'CVHFrs2kl_li_s1kj'),
                            dmss, 3, mol._atm, mol._bas, mol._env) * c1**4
    for i in range(3):
        vx[0,i] = pyscf.lib.hermi_triu(vx[0,i], 2)
    vj[:,n2c:,n2c:] = vx[0] + vx[1]
    vk[:,n2c:,n2c:] = vx[2] + vx[3]

    vx = _vhf.rdirect_bindm('cint2e_'+key+'_sa10sp1', 'CVHFdot_rs2kl',
                            ('CVHFrs2kl_lk_s1ij', 'CVHFrs2kl_ji_s2kl',
                             'CVHFrs2kl_jk_s1il', 'CVHFrs2kl_li_s1kj'),
                            (dmll,dmss,dmsl,dmls), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    for i in range(3):
        vx[1,i] = pyscf.lib.hermi_triu(vx[1,i], 2)
    vj[:,n2c:,n2c:] += vx[0]
    vj[:,:n2c,:n2c] += vx[1]
    vk[:,n2c:,:n2c] += vx[2]
    vk[:,:n2c,n2c:] += vx[3]
    for i in range(3):
        vj[i] = vj[i] + vj[i].T.conj()
        vk[i] = vk[i] + vk[i].T.conj()
    return vj, vk

def _call_giao_vhf1(mol, dm):
    c1 = .5/mol.light_speed
    n2c = dm.shape[0] / 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vx = _vhf.rdirect_mapdm('cint2e_g1', 'CVHFdot_rs4',
                            ('CVHFrah4_lk_s2ij', 'CVHFrah4_jk_s1il'),
                            dmll, 3, mol._atm, mol._bas, mol._env)
    vj[:,:n2c,:n2c] = vx[0]
    vk[:,:n2c,:n2c] = vx[1]
    vx = _vhf.rdirect_mapdm('cint2e_spgsp1spsp2', 'CVHFdot_rs4',
                            ('CVHFrah4_lk_s2ij', 'CVHFrah4_jk_s1il'),
                            dmss, 3, mol._atm, mol._bas, mol._env) * c1**4
    vj[:,n2c:,n2c:] = vx[0]
    vk[:,n2c:,n2c:] = vx[1]
    vx = _vhf.rdirect_bindm('cint2e_g1spsp2', 'CVHFdot_rs4',
                            ('CVHFrah4_lk_s2ij', 'CVHFrah4_jk_s1il'),
                            (dmss,dmls), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,:n2c,:n2c] += vx[0]
    vk[:,:n2c,n2c:] += vx[1]
    vx = _vhf.rdirect_bindm('cint2e_spgsp1', 'CVHFdot_rs4',
                            ('CVHFrah4_lk_s2ij', 'CVHFrah4_jk_s1il'),
                            (dmll,dmsl), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,n2c:,n2c:] += vx[0]
    vk[:,n2c:,:n2c] += vx[1]
    for i in range(3):
        vj[i] = pyscf.lib.hermi_triu(vj[i], 1)
        vk[i] = vk[i] + vk[i].T.conj()
    return vj, vk


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None#'out_dhf'

    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {
        'He': [(0, 0, (1., 1.)),
               (0, 0, (3., 1.)),
               (1, 0, (1., 1.)), ]}
    mol.build()

    mf = scf.dhf.UHF(mol)
    mf.scf()
    nmr = NMR(mf)
    nmr.mb = 'RMB'
    nmr.cphf = True
    msc = nmr.shielding()
    print(msc) # 64.4318104
