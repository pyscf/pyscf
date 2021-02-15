#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
NMR shielding of Dirac Hartree-Fock
'''

import sys
import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.data import nist

def dia(mol, dm0, gauge_orig=None, shielding_nuc=None, mb='RMB'):
    '''Note the side effects of set_common_origin'''

    if shielding_nuc is None:
        shielding_nuc = range(mol.natm)
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)

    n4c = dm0.shape[0]
    n2c = n4c // 2
    msc_dia = []
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        if mb.upper() == 'RMB':
            if gauge_orig is None:
                t11 = mol.intor('int1e_giao_sa10sa01_spinor', 9)
                t11 += mol.intor('int1e_spgsa01_spinor', 9)
            else:
                t11 = mol.intor('int1e_cg_sa10sa01_spinor', 9)
        elif gauge_orig is None:
            t11 = mol.intor('int1e_spgsa01_spinor', 9)
        else:
            t11 = numpy.zeros(9)
        h11 = numpy.zeros((9, n4c, n4c), complex)
        for i in range(9):
            h11[i,n2c:,:n2c] = t11[i] * .5
            h11[i,:n2c,n2c:] = t11[i].conj().T * .5
        a11 = numpy.real(numpy.einsum('xij,ji->x', h11, dm0))
        #    XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ = 1..9
        #  => [[XX, XY, XZ], [YX, YY, YZ], [ZX, ZY, ZZ]]
        msc_dia.append(a11)
    return numpy.array(msc_dia).reshape(-1, 3, 3)

def para(mol, mo10, mo_coeff, mo_occ, shielding_nuc=None):
    if shielding_nuc is None:
        shielding_nuc = range(mol.natm)
    n4c = mo_coeff.shape[1]
    n2c = n4c // 2
    msc_para = numpy.zeros((len(shielding_nuc),3,3))
    para_neg = numpy.zeros((len(shielding_nuc),3,3))
    para_occ = numpy.zeros((len(shielding_nuc),3,3))
    h01 = numpy.zeros((3, n4c, n4c), complex)
    orbo = mo_coeff[:,mo_occ>0]
    for n, atm_id in enumerate(shielding_nuc):
        mol.set_rinv_origin(mol.atom_coord(atm_id))
        t01 = mol.intor('int1e_sa01sp_spinor', 3)
        for m in range(3):
            h01[m,:n2c,n2c:] = .5 * t01[m]
            h01[m,n2c:,:n2c] = .5 * t01[m].conj().T
        h01_mo = [reduce(numpy.dot, (mo_coeff.conj().T, x, orbo)) for x in h01]
        for b in range(3):
            for m in range(3):
                # + c.c.
                p = numpy.einsum('ij,ij->i', mo10[b].conj(), h01_mo[m]).real * 2
                msc_para[n,b,m] = p.sum()
                para_neg[n,b,m] = p[:n2c].sum()
                para_occ[n,b,m] = p[mo_occ>0].sum()
    para_pos = msc_para - para_neg - para_occ
    return msc_para, para_pos, para_neg, para_occ

def make_h10giao(mol, dm0, with_gaunt=False, verbose=logger.WARN):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    log.debug('first order Fock matrix / GIAOs')
    n4c = dm0.shape[0]
    n2c = n4c // 2
    c = lib.param.LIGHT_SPEED

    tg = mol.intor('int1e_spgsp_spinor', 3)
    vg = mol.intor('int1e_gnuc_spinor', 3)
    wg = mol.intor('int1e_spgnucsp_spinor', 3)

    vj, vk = _call_giao_vhf1(mol, dm0)
    h1 = vj - vk
    if with_gaunt:
        sys.stderr('NMR gaunt part not implemented\n')
#TODO:        import pyscf.lib.pycint as pycint
#TODO:        vj, vk = scf.hf.get_vj_vk(pycint.rkb_giao_vhf_gaunt, mol, dm0)
#TODO:        h1 += vj - vk

    for i in range(3):
        h1[i,:n2c,:n2c] += vg[i]
        h1[i,n2c:,:n2c] += tg[i] * .5
        h1[i,:n2c,n2c:] += tg[i].conj().T * .5
        h1[i,n2c:,n2c:] += wg[i]*(.25/c**2) - tg[i]*.5
    return h1

def make_h10rkb(mol, dm0, gauge_orig=None, with_gaunt=False,
                verbose=logger.WARN):
    '''Note the side effects of set_common_origin'''

    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    log.debug('first order Fock matrix / RKB')
    if gauge_orig is None:
        t1 = mol.intor('int1e_giao_sa10sp_spinor', 3)
    else:
        t1 = mol.intor('int1e_cg_sa10sp_spinor', 3)
    if with_gaunt:
        sys.stderr('NMR gaunt part not implemented\n')
    n2c = t1.shape[2]
    n4c = n2c * 2
    h1 = numpy.zeros((3, n4c, n4c), complex)
    for i in range(3):
        h1[i,:n2c,n2c:] += .5 * t1[i]
        h1[i,n2c:,:n2c] += .5 * t1[i].conj().T
    return h1

#TODO the uncouupled force
def make_h10rmb(mol, dm0, gauge_orig=None, with_gaunt=False,
                verbose=logger.WARN):
    '''Note the side effects of set_common_origin'''

    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    log.debug('first order Fock matrix / RMB')
    n4c = dm0.shape[0]
    n2c = n4c // 2
    c = lib.param.LIGHT_SPEED
    if gauge_orig is None:
        t1 = mol.intor('int1e_giao_sa10sp_spinor', 3)
        v1 = mol.intor('int1e_giao_sa10nucsp_spinor', 3)
    else:
        t1 = mol.intor('int1e_cg_sa10sp_spinor', 3)
        v1 = mol.intor('int1e_cg_sa10nucsp_spinor', 3)

    if gauge_orig is None:
        #vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_coul, mol, dm0)
        vj, vk = _call_rmb_vhf1(mol, dm0, 'giao')
        h1 = vj - vk
        if with_gaunt:
            sys.stderr('NMR gaunt part not implemented\n')
            #import pyscf.lib.pycint as pycint
            #vj, vk = scf.hf.get_vj_vk(pycint.rmb4giao_vhf_gaunt, mol, dm0)
            #h1 += vj - vk
    else:
        #vj,vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_coul, mol, dm0)
        vj, vk = _call_rmb_vhf1(mol, dm0, 'cg')
        h1 = vj - vk
        if with_gaunt:
            sys.stderr('NMR gaunt part not implemented\n')
            #import pyscf.lib.pycint as pycint
            #vj, vk = scf.hf.get_vj_vk(pycint.rmb4cg_vhf_gaunt, mol, dm0)
            #h1 += vj - vk

    for i in range(3):
        t1cc = t1[i] + t1[i].conj().T
        h1[i,:n2c,n2c:] += t1cc * .5
        h1[i,n2c:,:n2c] += t1cc * .5
        h1[i,n2c:,n2c:] +=-t1cc * .5 + (v1[i]+v1[i].conj().T) * (.25/c**2)
    return h1

def make_h10(mol, dm0, gauge_orig=None, mb='RMB', with_gaunt=False,
             verbose=logger.WARN):
    if mb.upper() == 'RMB':
        h1 = make_h10rmb(mol, dm0, gauge_orig, with_gaunt, verbose)
    else: # RKB
        h1 = make_h10rkb(mol, dm0, gauge_orig, with_gaunt, verbose)
    if gauge_orig is None:
        h1 += make_h10giao(mol, dm0, with_gaunt, verbose)
    return h1

def get_fock(nmrobj, dm0=None, gauge_orig=None):
    '''First order Fock matrix wrt external magnetic field.
    Note the side effects of set_common_origin.
    '''
    if dm0 is None: dm0 = nmrobj._scf.make_rdm1()
    if gauge_orig is None: gauge_orig = nmrobj.gauge_orig
    t0 = (time.clock(), time.time())
    log = logger.Logger(nmrobj.stdout, nmrobj.verbose)
    mol = nmrobj.mol
    if nmrobj.mb.upper() == 'RMB':
        h1 = make_h10rmb(mol, dm0, gauge_orig,
                         with_gaunt=nmrobj._scf.with_gaunt, verbose=log)
    else: # RKB
        h1 = make_h10rkb(mol, dm0, gauge_orig,
                         with_gaunt=nmrobj._scf.with_gaunt, verbose=log)
    t0 = log.timer('%s h1'%nmrobj.mb, *t0)
    if nmrobj.chkfile:
        lib.chkfile.dump(nmrobj.chkfile, 'nmr/h1', h1)

    if gauge_orig is None:
        h1 += make_h10giao(mol, dm0,
                           with_gaunt=nmrobj._scf.with_gaunt, verbose=log)
    t0 = log.timer('GIAO', *t0)
    if nmrobj.chkfile:
        lib.chkfile.dump(nmrobj.chkfile, 'nmr/h1giao', h1)
    return h1

def make_s10(mol, gauge_orig=None, mb='RMB'):
    '''First order overlap matrix wrt external magnetic field.
    Note the side effects of set_common_origin.
    '''
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED
    s1 = numpy.zeros((3, n4c, n4c), complex)
    if mb.upper() == 'RMB':
        if gauge_orig is None:
            t1 = mol.intor('int1e_giao_sa10sp_spinor', 3)
        else:
            t1 = mol.intor('int1e_cg_sa10sp_spinor', 3)
        for i in range(3):
            t1cc = t1[i] + t1[i].conj().T
            s1[i,n2c:,n2c:] = t1cc * (.25/c**2)

    if gauge_orig is None:
        sg = mol.intor('int1e_govlp_spinor', 3)
        tg = mol.intor('int1e_spgsp_spinor', 3)
        s1[:,:n2c,:n2c] += sg
        s1[:,n2c:,n2c:] += tg * (.25/c**2)
    return s1
get_ovlp = make_s10

def gen_vind(mf, mo_coeff, mo_occ):
    '''Induced potential'''
    mol = mf.mol
    occidx = mo_occ > 0
    orbo = mo_coeff[:,occidx]
    nao, nmo = mo_coeff.shape
    nocc = orbo.shape[1]
    def vind(mo1):
        #direct_scf_bak, mf.direct_scf = mf.direct_scf, False
        dm1 = numpy.asarray([reduce(numpy.dot, (mo_coeff, x, orbo.T.conj()))
                             for x in mo1.reshape(-1,nmo,nocc)])
        dm1 = dm1 + dm1.transpose(0,2,1).conj()
# hermi=1 because dm1 = C^1 C^{0dagger} + C^0 C^{1dagger}
        v1mo = numpy.asarray([reduce(numpy.dot, (mo_coeff.T.conj(), x, orbo))
                              for x in mf.get_veff(mol, dm1, hermi=1)])
        #mf.direct_scf = direct_scf_bak
        return v1mo.ravel()
    return vind

#TODO: merge to hessian.rhf.solve_mo1 function
@lib.with_doc(rhf_nmr.solve_mo1.__doc__)
def solve_mo1(nmrobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              h1=None, s1=None, with_cphf=None):
    if with_cphf is None:
        with_cphf = nmrobj.cphf

    # vind for DHF for first order equations
    if with_cphf and not callable(with_cphf):
        mf = nmrobj._scf
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
        with_cphf = gen_vind(mf, mo_coeff, mo_occ)
    return rhf_nmr.solve_mo1(nmrobj, mo_energy, mo_coeff, mo_occ,
                             h1, s1, with_cphf)


class NMR(rhf_nmr.NMR):
    __doc__ = 'magnetic shielding constants'
    def __init__(self, scf_method):
        rhf_nmr.NMR.__init__(self, scf_method)
        self.cphf = True
        self.mb = 'RMB'
        self._keys = self._keys.union(['mb'])

    def dump_flags(self, verbose=None):
        rhf_nmr.NMR.dump_flags(self, verbose)
        logger.info(self, 'MB basis = %s', self.mb)
        return self

    def shielding(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.dump_flags()
        if self.verbose >= logger.WARN:
            self.check_sanity()

        t0 = (time.clock(), time.time())
        unit_ppm = nist.ALPHA**2 * 1e6
        msc_dia = self.dia() * unit_ppm
        t0 = logger.timer(self, 'h11', *t0)
        msc_para, para_pos, para_neg, para_occ = \
                [x*unit_ppm for x in self.para(mo10=mo1)]
        e11 = msc_para + msc_dia

        logger.timer(self, 'NMR shielding', *cput0)
        if self.verbose > logger.QUIET:
            for i, atm_id in enumerate(self.shielding_nuc):
                rhf_nmr._write(self.stdout, e11[i],
                               '\ntotal shielding of atom %d %s'
                               % (atm_id, self.mol.atom_symbol(atm_id)))
                rhf_nmr._write(self.stdout, msc_dia[i], 'dia-magnetism')
                rhf_nmr._write(self.stdout, msc_para[i], 'para-magnetism')
                if self.verbose >= logger.INFO:
                    rhf_nmr._write(self.stdout, para_occ[i], 'occ part of para-magnetism')
                    rhf_nmr._write(self.stdout, para_pos[i], 'vir-pos part of para-magnetism')
                    rhf_nmr._write(self.stdout, para_neg[i], 'vir-neg part of para-magnetism')
        return e11

    def dia(self, mol=None, dm0=None, gauge_orig=None, shielding_nuc=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        if shielding_nuc is None: shielding_nuc = self.shielding_nuc
        if dm0 is None:
            dm0 = self._scf.make_rdm1(self._scf.mo_coeff, self._scf.mo_occ)
        return dia(mol, dm0, gauge_orig, shielding_nuc, self.mb)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None,
             shielding_nuc=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if shielding_nuc is None: shielding_nuc = self.shielding_nuc
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(mol, mo10, mo_coeff, mo_occ, shielding_nuc)

    make_h10 = get_fock = get_fock

    def make_s10(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return make_s10(mol, gauge_orig, mb=self.mb)
    get_ovlp = make_s10

    solve_mo1 = solve_mo1

from pyscf import scf
scf.dhf.UHF.NMR = lib.class_as_method(NMR)


def _call_rmb_vhf1(mol, dm, key='giao'):
    c1 = .5 / lib.param.LIGHT_SPEED
    n2c = dm.shape[0] // 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vx = _vhf.rdirect_mapdm('int2e_'+key+'_sa10sp1spsp2_spinor', 's2kl',
                            ('ji->s2kl', 'lk->s1ij', 'jk->s1il', 'li->s1kj'),
                            dmss, 3, mol._atm, mol._bas, mol._env) * c1**4
    for i in range(3):
        vx[0,i] = lib.hermi_triu(vx[0,i], 2)
    vj[:,n2c:,n2c:] = vx[0] + vx[1]
    vk[:,n2c:,n2c:] = vx[2] + vx[3]

    vx = _vhf.rdirect_bindm('int2e_'+key+'_sa10sp1_spinor', 's2kl',
                            ('lk->s1ij', 'ji->s2kl', 'jk->s1il', 'li->s1kj'),
                            (dmll,dmss,dmsl,dmls), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    for i in range(3):
        vx[1,i] = lib.hermi_triu(vx[1,i], 2)
    vj[:,n2c:,n2c:] += vx[0]
    vj[:,:n2c,:n2c] += vx[1]
    vk[:,n2c:,:n2c] += vx[2]
    vk[:,:n2c,n2c:] += vx[3]
    for i in range(3):
        vj[i] = vj[i] + vj[i].T.conj()
        vk[i] = vk[i] + vk[i].T.conj()
    return vj, vk

def _call_giao_vhf1(mol, dm):
    c1 = .5 / lib.param.LIGHT_SPEED
    n2c = dm.shape[0] // 2
    dmll = dm[:n2c,:n2c].copy()
    dmls = dm[:n2c,n2c:].copy()
    dmsl = dm[n2c:,:n2c].copy()
    dmss = dm[n2c:,n2c:].copy()
    vj = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vk = numpy.zeros((3,n2c*2,n2c*2), dtype=numpy.complex)
    vx = _vhf.rdirect_mapdm('int2e_g1_spinor', 'a4ij',
                            ('lk->s2ij', 'jk->s1il'), dmll, 3,
                            mol._atm, mol._bas, mol._env)
    vj[:,:n2c,:n2c] = vx[0]
    vk[:,:n2c,:n2c] = vx[1]
    vx = _vhf.rdirect_mapdm('int2e_spgsp1spsp2_spinor', 'a4ij',
                            ('lk->s2ij', 'jk->s1il'), dmss, 3,
                            mol._atm, mol._bas, mol._env) * c1**4
    vj[:,n2c:,n2c:] = vx[0]
    vk[:,n2c:,n2c:] = vx[1]
    vx = _vhf.rdirect_bindm('int2e_g1spsp2_spinor', 'a4ij',
                            ('lk->s2ij', 'jk->s1il'), (dmss,dmls), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,:n2c,:n2c] += vx[0]
    vk[:,:n2c,n2c:] += vx[1]
    vx = _vhf.rdirect_bindm('int2e_spgsp1_spinor', 'a4ij',
                            ('lk->s2ij', 'jk->s1il'), (dmll,dmsl), 3,
                            mol._atm, mol._bas, mol._env) * c1**2
    vj[:,n2c:,n2c:] += vx[0]
    vk[:,n2c:,:n2c] += vx[1]
    for i in range(3):
        vj[i] = lib.hermi_triu(vj[i], 1)
        vk[i] = vk[i] + vk[i].T.conj()
    return vj, vk


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#'out_dhf'

    mol.atom = [['He', (0.,0.,0.)], ]
    mol.basis = {
        'He': [(0, 0, (1., 1.)),
               (0, 0, (3., 1.)),
               (1, 0, (1., 1.)), ]}
    mol.build()

    mf = scf.dhf.UHF(mol)
    mf.scf()
    nmr = mf.NMR()
    nmr.mb = 'RMB'
    nmr.cphf = True
    msc = nmr.shielding()
    print(msc) # 64.4318104
