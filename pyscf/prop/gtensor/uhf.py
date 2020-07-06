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
Non-relativistic unrestricted Hartree-Fock g-tensor

Refs:
    JPCA 101, 3388 (1997); DOI:10.1021/jp963060t
    JCP 115, 11080 (2001); DOI:10.1063/1.1419058
    JCP 119, 10489 (2003); DOI:10.1063/1.1620497
Note g-tensor = 1/muB d^2 E/ dB dS
In some literature, muB is not explicitly presented in the perturbation formula.
'''

import time
from functools import reduce
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.zfs.uhf import koseki_charge
from pyscf.data import nist

def dia(gobj, dm0, gauge_orig=None):
    '''Note the side effects of set_common_origin'''

    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))
    mol = gobj.mol

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    alpha2 = nist.ALPHA ** 2
    #Many choices of qed_fac, see JPC, 101, 3388
    #qed_fac = (nist.G_ELECTRON - 1)
    #qed_fac = nist.G_ELECTRON / 2
    qed_fac = 1

# relativistic mass correction (RMC)
    rmc = -numpy.einsum('ij,ji', mol.intor('int1e_kin'), spindm)
    rmc *= qed_fac / effspin * alpha2
    logger.info(gobj, 'RMC = %s', rmc)

    assert(not mol.has_ecp())
# GC(1e)
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    h11 = 0
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        Z = mol.atom_charge(ia)
        if gobj.so_eff_charge or not gobj.dia_soc2e:
            Z = koseki_charge(Z)
# GC(1e) = 1/4c^2 Z/(2r_N^3) [vec{r}_N dot r sigma dot B - B dot vec{r}_N r dot sigma]
# a11part = (B dot) -1/2 frac{\vec{r}_N}{r_N^3} r (dot sigma)
        if gauge_orig is None:
            h11 += Z * mol.intor('int1e_giao_a11part', 9)
        else:
            h11 += Z * mol.intor('int1e_cg_a11part', 9)
    trh11 = h11[0] + h11[4] + h11[8]
    h11[0] -= trh11
    h11[4] -= trh11
    h11[8] -= trh11
    if gauge_orig is None:
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            Z = mol.atom_charge(ia)
            if gobj.so_eff_charge or not gobj.dia_soc2e:
                Z = koseki_charge(Z)
            h11 += Z * mol.intor('int1e_a01gp', 9)
    gc1e = numpy.einsum('xij,ji->x', h11, spindm).reshape(3,3)
    if gobj.mb:  # correction of order c^{-2} from MB basis
        gc1e += numpy.einsum('ij,ji', mol.intor('int1e_nuc'), spindm) * numpy.eye(3)

    gc1e *= (alpha2/4) / effspin / muB
    if gobj.verbose >= logger.INFO:
        _write(gobj, gobj.align(gc1e)[0], 'GC(1e)')

    if gobj.dia_soc2e:
        gc2e = gobj.make_dia_gc2e(dm0, gauge_orig, qed_fac)
        if gobj.verbose >= logger.INFO:
            _write(gobj, gobj.align(gc2e)[0], 'GC(2e)')
    else:
        gc2e = 0

    gdia = gc1e + gc2e + rmc * numpy.eye(3)
    return gdia

def make_dia_gc2e(gobj, dm0, gauge_orig, sso_qed_fac=1):
    '''Note the side effects of set_common_origin'''

    if (isinstance(gobj.dia_soc2e, str) and
        ('SOMF' in gobj.dia_soc2e.upper() or 'AMFI' in gobj.dia_soc2e.upper())):
        raise NotImplementedError(gobj.dia_soc2e)
    if isinstance(gobj.dia_soc2e, str):
        with_sso = 'SSO' in gobj.dia_soc2e.upper()
        with_soo = 'SOO' in gobj.dia_soc2e.upper()
    else:
        with_sso = with_soo = True

    mol = gobj.mol
    dma, dmb = dm0
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    alpha2 = nist.ALPHA ** 2
    #sso_qed_fac = (nist.G_ELECTRON - 1)

    # int2e_ip1v_r1 = (ij|\frac{\vec{r}_{12}}{r_{12}^3} \vec{r}_1|kl)
    if gauge_orig is None:
        intor = mol._add_suffix('int2e_ip1v_r1')
    else:
        mol.set_common_origin(gauge_orig)
        intor = mol._add_suffix('int2e_ip1v_rc1')
    vj, vk = _vhf.direct_mapdm(intor,
                               's2kl', ('lk->s1ij', 'jk->s1il'),
                               (dma, dmb), 9,
                               mol._atm, mol._bas, mol._env)
    ek = numpy.einsum('xil,li->x', vk[0], dma)
    ek-= numpy.einsum('xil,li->x', vk[1], dmb)
    ek = ek.reshape(3,3)
    gc2e = 0
    if with_sso:
        # spin-density should be contracted to electron 1 (associated to operator r1)
        ej = numpy.einsum('xij,ji->x', vj[0]+vj[1], dma-dmb).reshape(3,3)
        gc2e += sso_qed_fac * (ej - ek)
    if with_soo:
        # spin-density should be contracted to electron 2
        ej = numpy.einsum('xij,ji->x', vj[0]-vj[1], dma+dmb).reshape(3,3)
        gc2e += 2 * (ej - ek)
    gc2e -= numpy.eye(3) * gc2e.trace()
    gc2e *= (alpha2/8) / effspin / muB

    #   ([GIAO-i j] + [i GIAO-j]|\frac{\vec{r}_{12}}{r_{12}^3} x p1|kl)
    # + (ij|\frac{\vec{r}_{12}}{r_{12}^3} x p1|[GIAO-k l] + [k GIAO-l])
    if gauge_orig is None:
        nao = dma.shape[0]
        vj, vk = _vhf.direct_mapdm(mol._add_suffix('int2e_ipvg1_xp1'),
                                   's2kl', ('lk->s1ij', 'jk->s1il'),
                                   (dma, dmb), 9,
                                   mol._atm, mol._bas, mol._env)
        vk1 = _vhf.direct_mapdm(mol._add_suffix('int2e_ipvg2_xp1'),
                                   'aa4', 'jk->s1il',
                                   (dma, dmb), 9,
                                   mol._atm, mol._bas, mol._env)
        vj = vj.reshape(2,3,3,nao,nao)
        vk = vk.reshape(2,3,3,nao,nao)
        vk += vk1.reshape(2,3,3,nao,nao).transpose(0,2,1,3,4)
        ek = numpy.einsum('xyij,ji->xy', vk[0], dma)
        ek-= numpy.einsum('xyij,ji->xy', vk[1], dmb)
        dia_giao = 0
        if with_sso:
            ej = numpy.einsum('xyij,ji->xy', vj[0]+vj[1], dma-dmb)
            dia_giao += sso_qed_fac * (ej - ek)
        if with_soo:
            ej = numpy.einsum('xyij,ji->xy', vj[0]-vj[1], dma+dmb)
            dia_giao += 2 * (ej - ek)
        gc2e -= dia_giao * (alpha2/4) / effspin / muB

    if gobj.mb:  # correction of order c^{-2} from MB basis
        vj, vk = gobj._scf.get_jk(mol, dm0)
        vhf = vj[0] + vj[1] - vk
        gc_mb = numpy.einsum('ij,ji', vhf[0], dma)
        gc_mb-= numpy.einsum('ij,ji', vhf[1], dmb)
        gc2e += gc_mb * (alpha2/4) / effspin / muB * numpy.eye(3)

    return gc2e


# Note mo10 is the imaginary part of MO^1
def para(gobj, mo10, mo_coeff, mo_occ, qed_fac=1):
    mol = gobj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #qed_fac = (nist.G_ELECTRON - 1)
    #qed_fac = nist.G_ELECTRON / 2

    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    dm10a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]]
    dm10b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]]
    dm10a = numpy.asarray([x-x.T for x in dm10a])
    dm10b = numpy.asarray([x-x.T for x in dm10b])

    hso1e = make_h01_soc1e(gobj, mo_coeff, mo_occ, qed_fac)
    gpara1e =-numpy.einsum('xji,yij->xy', dm10a, hso1e)
    gpara1e+= numpy.einsum('xji,yij->xy', dm10b, hso1e)
    gpara1e *= 1. / effspin / muB
    if gobj.verbose >= logger.INFO:
        _write(gobj, gobj.align(gpara1e)[0], 'SOC(1e)/OZ')

    gpara2e = gobj.make_para_soc2e((dm0a,dm0b), (dm10a,dm10b), qed_fac)
    gpara = gpara1e + gpara2e
    return gpara

def make_para_soc2e(gobj, dm0, dm10, sso_qed_fac=1):
    if isinstance(gobj.para_soc2e, str):
        with_sso = 'SSO' in gobj.para_soc2e.upper()
        with_soo = 'SOO' in gobj.para_soc2e.upper()
        with_somf = 'SOMF' in gobj.para_soc2e.upper()
        with_amfi = 'AMFI' in gobj.para_soc2e.upper()
        assert(not (with_somf and (with_sso or with_soo)))
    elif gobj.para_soc2e:
        with_sso = with_soo = True
        with_somf = with_amfi = False
    else:
        return 0

    mol = gobj.mol
    alpha2 = nist.ALPHA ** 2
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    #sso_qed_fac = (nist.G_ELECTRON - 1)
    #sso_qed_fac = nist.G_ELECTRON / 2

    dm10a, dm10b = dm10
    if with_amfi:
        vj, vk = get_jk_amfi(mol, dm0)
    else:
        vj, vk = get_jk(mol, dm0)

    gpara2e = 0
    if with_sso or with_soo:
        ek  = numpy.einsum('yil,xli->xy', vk[0], dm10a)
        ek -= numpy.einsum('yil,xli->xy', vk[1], dm10b)
        if with_sso:
            ej = numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
# ~ <H^{01},MO^1> = - Tr(Im[H^{01}],Im[MO^1])
            gpara2e -= sso_qed_fac * (ej - ek)
        if with_soo:
            ej = numpy.einsum('yij,xji->xy', vj[0]-vj[1], dm10a+dm10b)
            gpara2e -= 2 * (ej - ek)
    else:  # SOMF, see JCP 122, 034107 (2005); DOI:10.1063/1.1829047 Eq (19)
        ej = numpy.einsum('yij,xji->xy', vj[0]+vj[1], dm10a-dm10b)
        ek = numpy.einsum('yil,xli->xy', vk[0]+vk[1], dm10a-dm10b)
        gpara2e -= ej - 1.5 * ek
    gpara2e *= (alpha2/4) / effspin / muB
    if gobj.verbose >= logger.INFO:
        _write(gobj, gobj.align(gpara2e)[0], 'SOC(2e)/OZ')
    return gpara2e


def para_for_debug(gobj, mo10, mo_coeff, mo_occ, qed_fac=1):
    mol = gobj.mol
    effspin = mol.spin * .5
    muB = .5  # Bohr magneton
    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm10a = [reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]]
    dm10b = [reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]]
    dm10a = numpy.asarray([x-x.T for x in dm10a])
    dm10b = numpy.asarray([x-x.T for x in dm10b])

    # <H^{01},MO^1> = - Tr(Im[H^{01}],Im[MO^1])
    hso1e = make_h01_soc1e(gobj, mo_coeff, mo_occ, qed_fac)
    gpara1e =-numpy.einsum('xji,yij->xy', dm10a, hso1e)
    gpara1e+= numpy.einsum('xji,yij->xy', dm10b, hso1e)
    gpara1e *= 1./effspin / muB
    if gobj.verbose >= logger.INFO:
        _write(gobj, gobj.align(gpara1e)[0], 'SOC(1e)/OZ')

    if gobj.para_soc2e:
        h1aa, h1bb = make_h01_soc2e(gobj, mo_coeff, mo_occ, qed_fac)
        gpara2e =-numpy.einsum('xji,yij->xy', dm10a, h1aa)
        gpara2e-= numpy.einsum('xji,yij->xy', dm10b, h1bb)
        gpara2e *= 1./effspin / muB
        if gobj.verbose >= logger.INFO:
            _write(gobj, gobj.align(gpara2e)[0], 'SOC(2e)/OZ')
    else:
        gpara2e = 0
    gpara = gpara1e + gpara2e
    return gpara


def make_h01_soc1e(gobj, mo_coeff, mo_occ, qed_fac=1):
    mol = gobj.mol
    assert(not mol.has_ecp())
    alpha2 = nist.ALPHA ** 2
    #qed_fac = (nist.G_ELECTRON - 1)

# hso1e is the imaginary part of [i sigma dot pV x p]
# JCP 122, 034107 (2005); DOI:10.1063/1.1829047 Eq (2) = 1/4c^2 hso1e
    if gobj.so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            Z = koseki_charge(mol.atom_charge(ia))
            mol.set_rinv_origin(mol.atom_coord(ia))
            hso1e += -Z * mol.intor_asymmetric('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor_asymmetric('int1e_pnucxp', 3)
    hso1e *= qed_fac * (alpha2/4)
    return hso1e

def get_jk(mol, dm0):
    # K_{pq} = (pi|iq) + (iq|pi)
    vj, vk, vk1 = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                                    'a4ij', ('lk->s2ij', 'jk->s1il', 'li->s1kj'),
                                    dm0, 3, mol._atm, mol._bas, mol._env)
    for i in range(3):
        lib.hermi_triu(vj[0,i], hermi=2, inplace=True)
        lib.hermi_triu(vj[1,i], hermi=2, inplace=True)
    vk += vk1
    return vj, vk

def get_jk_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    dma, dmb = dm0
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = numpy.zeros((2,3,nao,nao))
    vk = numpy.zeros((2,3,nao,nao))
    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_jk(atom, (dma[p0:p1,p0:p1],dmb[p0:p1,p0:p1]))
        vj[:,:,p0:p1,p0:p1] = vj1
        vk[:,:,p0:p1,p0:p1] = vk1
    return vj, vk

def get_j(mol, dm0):
    vj = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                           'a4ij', 'lk->s2ij',
                           dm0, 3, mol._atm, mol._bas, mol._env)
    for i in range(3):
        lib.hermi_triu(vj[0,i], hermi=2, inplace=True)
        lib.hermi_triu(vj[1,i], hermi=2, inplace=True)
    return vj

def get_j_amfi(mol, dm0):
    '''Atomic-mean-field approximation'''
    return get_jk_amfi(mol, dm0)[0]


# hso2e is the imaginary part of SSO
# SSO term of JCP 122, 034107 (2005); DOI:10.1063/1.1829047 Eq (3) = 1/4c^2 hso2e
def make_h01_soc2e(gobj, mo_coeff, mo_occ, sso_qed_fac=1):
    mol = gobj.mol
    alpha2 = nist.ALPHA ** 2
    #sso_qed_fac = (nist.G_ELECTRON - 1)

    dm0 = gobj._scf.make_rdm1(mo_coeff, mo_occ)
    vj, vk = get_jk(mol, dm0)

    vjaa = 0
    vjbb = 0
    vkaa = 0
    vkbb = 0
    if 'SSO' in gobj.para_soc2e.upper():
        vj1 = vj[0] + vj[1]
        vjaa += vj1 * sso_qed_fac
        vjbb -= vj1 * sso_qed_fac
        vkaa += vk[0] * sso_qed_fac
        vkbb -= vk[1] * sso_qed_fac
    if 'SOO' in gobj.para_soc2e.upper():
        vj1 = vj[0] - vj[1]
        vjaa += vj1 * 2
        vjbb += vj1 * 2
        vkaa += vk[0] * 2
        vkbb -= vk[1] * 2
    haa = (vjaa - vkaa) * (alpha2/4)
    hbb = (vjbb - vkbb) * (alpha2/4)
    return haa, hbb

def align(gtensor):
    '''Transform the orientation of g-tensor.
    The new orientations are the eigenvector of G matrix (G=g.gT)
    '''
    w, v = numpy.linalg.eigh(numpy.dot(gtensor, gtensor.T))
    idxmax = abs(v).argmax(axis=0)
    v[:,v[idxmax,[0,1,2]]<0] *= -1  # format phase
    sorted_axis = numpy.argsort(idxmax)
    v = v[:,sorted_axis]
    if numpy.linalg.det(v) < 0: # ensure new axes in RHS
        v[:,2] *= -1
    g2 = reduce(numpy.dot, (v.T, gtensor, v))
    return g2, v

def _write(gobj, gtensor, title):
    gobj.stdout.write('%s %s\n' % (title, gtensor.diagonal()))
    if gobj.verbose >= logger.DEBUG:
        rhf_nmr._write(gobj.stdout, gtensor, title+' tensor')
        w = numpy.linalg.svd(gtensor)[1]
        gobj.stdout.write('sqrt(ggT) %s\n' % w)


class GTensor(lib.StreamObject):
    '''dE = B dot gtensor dot s

    Attributes:
        dia_soc2e : str or bool
            2-electron spin-orbit coupling for diamagnetic term. Its value can
            be 'SSO', 'SOO', 'SSO+SOO', None/False or True (='SSO+SOO').
            Default is False.
        para_soc2e : str or bool
            2-electron spin-orbit coupling for paramagnetic term.  Its value
            can be 'SSO', 'SOO', 'SSO+SOO', None/False, True (='SSO+SOO')
            'SOMF', 'AMFI' (='AMFI+SSO+SOO'), 'SOMF+AMFI', 'AMFI+SSO',
            'AMFI+SOO', 'AMFI+SSO+SOO'.  Default is 'SSO+SOO'.
        koseki_charge : bool
            Whether to use Koseki effective SOC charge in 1-electron
            diamagnetic term and paramagnetic term.  Default is False.
    '''
    def __init__(self, mf):
        self.mol = mf.mol
        self.verbose = mf.mol.verbose
        self.stdout = mf.mol.stdout
        self.chkfile = mf.chkfile
        self._scf = mf

        # dia_soc2e is 2-electron spin-orbit coupling for diamagnetic term
        # dia_soc2e can be 'SSO', 'SOO', 'SSO+SOO', None/False, True (='SSO+SOO')
        self.dia_soc2e = False
        # para_soc2e is 2-electron spin-orbit coupling for paramagnetic term
        # para_soc2e can be 'SSO', 'SOO', 'SSO+SOO', None/False, True (='SSO+SOO')
        # 'SOMF', 'AMFI' (='AMFI+SSO+SOO'), 'SOMF+AMFI', 'AMFI+SSO',
        # 'AMFI+SOO', 'AMFI+SSO+SOO'
        self.para_soc2e = 'SSO+SOO'
        # Koseki effective SOC charge
        self.so_eff_charge = False

        # corresponding to RMB basis (DO NOT use. It's in testing)
        self.mb = False

# gauge_orig=None will call GIAO. A coordinate array leads to common gauge
        self.gauge_orig = None
        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s (In testing) ********',
                 self.__class__, self._scf.__class__)
        if self.gauge_orig is None:
            log.info('gauge = GIAO')
        else:
            log.info('Common gauge = %s', str(self.gauge_orig))
        log.info('with cphf = %s', self.cphf)
        if self.cphf:
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        log.info('dia_soc2e = %s', self.dia_soc2e)
        log.info('para_soc2e = %s', self.para_soc2e)
        log.info('so_eff_charge = %s (1e SO effective charge)',
                 self.so_eff_charge)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        gdia = self.dia(self._scf.make_rdm1(), self.gauge_orig)
        gpara = self.para(mo1, self._scf.mo_coeff, self._scf.mo_occ)
        gshift = gpara + gdia
        gtensor = gshift + numpy.eye(3) * nist.G_ELECTRON

        logger.timer(self, 'g-tensor', *cput0)
        if self.verbose >= logger.NOTE:
            logger.note(self, 'free electron g %s', nist.G_ELECTRON)
            gtot, v = self.align(gtensor)
            gdia = reduce(numpy.dot, (v.T, gdia, v))
            gpara = reduce(numpy.dot, (v.T, gpara, v))
            gshift = gtot - numpy.eye(3) * nist.G_ELECTRON
            if self.verbose >= logger.INFO:
                _write(self, gdia, 'g-tensor diamagnetic terms')
                _write(self, gpara, 'g-tensor paramagnetic terms')
            _write(self, gtot, 'g-tensor total')
            _write(self, gshift*1e3, 'g-shift (ppt)')
        return gtensor

    def dia(self, dm0=None, gauge_orig=None):
        if gauge_orig is None: gauge_orig = self.gauge_orig
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return dia(self, dm0, gauge_orig)

    def para(self, mo10=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self._scf.mo_coeff
        if mo_occ is None:   mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(self, mo10, mo_coeff, mo_occ)

    make_dia_gc2e = make_dia_gc2e
    make_para_soc2e = make_para_soc2e
    solve_mo1 = uhf_nmr.solve_mo1
    get_fock = uhf_nmr.get_fock

    def get_ovlp(self, mol=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        return rhf_nmr.get_ovlp(mol, gauge_orig)

    def align(self, gtensor):
        return align(gtensor)


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='C 0 0 0; O 0 0 1.25',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.newton(scf.UHF(mol)).run()
    gobj = GTensor(mf)
    gobj.para_soc2e = 'SSO+SOO'
    gobj.dia_soc2e = None
    gobj.so_eff_charge = False
    gobj.gauge_orig = (0,0,0)
    print(gobj.align(gobj.kernel())[0])

    mol = gto.M(atom='''
                H 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UHF(mol).run()
    gobj = GTensor(mf)
    print(gobj.align(gobj.kernel())[0])

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol).run()
    gobj = GTensor(mf)
    #print(gobj.kernel())
    gobj.para_soc2e = 'SSO+SOO'
    gobj.dia_soc2e = None
    gobj.so_eff_charge = False
    nao, nmo = mf.mo_coeff[0].shape
    nelec = mol.nelec
    numpy.random.seed(1)
    mo10 =[numpy.random.random((3,nmo,nelec[0])),
           numpy.random.random((3,nmo,nelec[1]))]
    print(lib.finger(para(gobj, mo10, mf.mo_coeff, mf.mo_occ)) - 4.3706065384682997e-05)
    print(lib.finger(para_for_debug(gobj, mo10, mf.mo_coeff, mf.mo_occ)) - 4.3706065384682997e-05)
    numpy.random.seed(1)
    dm0 = numpy.random.random((2,nao,nao))
    dm0 = dm0 + dm0.transpose(0,2,1)
    dm10 = numpy.random.random((2,3,nao,nao))
    dm10 = dm10 - dm10.transpose(0,1,3,2)
    print(lib.finger(make_para_soc2e(gobj, dm0, dm10)) - 0.011068947999868441)
    print(lib.finger(make_dia_gc2e(gobj, dm0, (.5,1,2))) - -0.0058333522256648652)
    print(lib.finger(make_dia_gc2e(gobj, dm0, None)) - 0.0015992772016390443)
