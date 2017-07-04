#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic unrestricted Hartree-Fock g-tensor

Refs:
    JPC, 101, 3388
    JCP, 115, 11080
    JCP, 119, 10489
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.prop.nmr import rhf as rhf_nmr
from pyscf.prop.nmr import uhf as uhf_nmr

def dia(gobj, mol, dm0, gauge_orig=None):
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5
    alpha2 = lib.param.ALPHA ** 2
    # FIXME: see JPC, 101, 3388, why?
    g_so = (lib.param.G_ELECTRON - 1) * 2

# relativistic mass correction (RMC)
    rmc = -numpy.einsum('ij,ji', mol.intor('int1e_kin'), spindm)
    rmc *= lib.param.G_ELECTRON/2 / effspin * alpha2
    logger.info(gobj, 'RMC = %s', rmc)

# GC(1e)
    if gauge_orig is not None:
        mol.set_common_origin(gauge_orig)
    h11 = 0
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        Z = mol.atom_charge(ia)
        #FIXME: when ECP is enabled
        if gobj.with_so_eff_charge:
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
            #FIXME: when ECP is enabled
            if gobj.with_so_eff_charge:
                Z = koseki_charge(Z)
            h11 += Z * mol.intor('int1e_a01gp', 9)
    gc1e = numpy.einsum('xij,ji->x', h11, spindm).reshape(3,3)

    if 0:  # correction of order c^{-2} from MB basis or DPT (JCP,115,7356), does it exist?
        gc1e += numpy.einsum('ij,ji', mol.intor('int1e_nuc'), spindm) * numpy.eye(3)

    gc1e *= g_so * (alpha2/4) / effspin
    _write(gobj, gc1e, 'GC(1e)')

    if gobj.with_sso or gobj.with_soo:
        gc2e = make_dia_gc2e(gobj, dm0, gauge_orig)
        _write(gobj, gc2e, 'GC(2e)')
    else:
        gc2e = 0

    gdia = gc1e + gc2e + rmc * numpy.eye(3)
    return gdia

def make_dia_gc2e(gobj, dm0, gauge_orig):
    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5
    alpha2 = lib.param.ALPHA ** 2
    # FIXME: see JPC, 101, 3388 Eq (11c), why?
    g_so = (lib.param.G_ELECTRON - 1) * 2
    nao = dma.shape[0]

    if gauge_orig is None:
        gc2e_ri = mol.intor('int2e_ip1v_r1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
    else:
        mol.set_common_origin(gauge_orig)
        gc2e_ri = mol.intor('int2e_ip1v_rc1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
    ej = numpy.zeros((3,3))
    ek = numpy.zeros((3,3))
    if gobj.with_sso:
        # spin-density should be contracted to electron 1 (associated to operator r_i)
        ej += g_so/2 * numpy.einsum('xyijkl,ij,kl->xy', gc2e_ri, spindm, dma+dmb)
        ek += g_so/2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dma, dma)
        ek -= g_so/2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dmb, dmb)
    if gobj.with_soo:
        # spin-density should be contracted to electron 1 (associated to operator r_i)
        ej += 2 * numpy.einsum('xyijkl,ij,kl->xy', gc2e_ri, dma+dmb, dma-dmb)
        ek += 2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dma, dma)
        ek -= 2 * numpy.einsum('xyijkl,jk,li->xy', gc2e_ri, dmb, dmb)
    gc2e = (ej - ek) / 2
    gc2e -= numpy.eye(3) * gc2e.trace()

    if gauge_orig is None:
        giao2e1 = mol.intor('int2e_ipvg1_xp1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
        giao2e2 = mol.intor('int2e_ipvg2_xp1', comp=9, aosym='s1').reshape(3,3,nao,nao,nao,nao)
        ej = numpy.zeros((3,3))
        ek = numpy.zeros((3,3))
        if gobj.with_sso:
            ej += -2 * g_so/2 * numpy.einsum('yxijkl,ij,kl->xy', giao2e1, spindm, dma+dmb)
            ek += -2 * g_so/2 * numpy.einsum('yxijkl,jk,li->xy', giao2e1, dma, dma)
            ek -= -2 * g_so/2 * numpy.einsum('yxijkl,jk,li->xy', giao2e1, dmb, dmb)
            ej += -2 * g_so/2 * numpy.einsum('yxijkl,ij,kl->xy', giao2e2, spindm, dma+dmb)
            ek += -2 * g_so/2 * numpy.einsum('yxijkl,jk,li->xy', giao2e2, dma, dma)
            ek -= -2 * g_so/2 * numpy.einsum('yxijkl,jk,li->xy', giao2e2, dmb, dmb)
        if gobj.with_soo:
            ej += -2 * 2 * numpy.einsum('yxijkl,ij,kl->xy', giao2e1, dma+dmb, spindm)
            ek += -2 * 2 * numpy.einsum('yxijkl,jk,li->xy', giao2e1, dma, dma)
            ek -= -2 * 2 * numpy.einsum('yxijkl,jk,li->xy', giao2e1, dmb, dmb)
            ej += -2 * 2 * numpy.einsum('yxijkl,ij,kl->xy', giao2e2, dma+dmb, spindm)
            ek += -2 * 2 * numpy.einsum('yxijkl,jk,li->xy', giao2e2, dma, dma)
            ek -= -2 * 2 * numpy.einsum('yxijkl,jk,li->xy', giao2e2, dmb, dmb)
        gc2e += (ej - ek) / 2

    if 0:  # correction of order c^{-2} from MB basis, does it exist?
        vj, vk = gobj._scf.get_jk(mol, dm0)
        vhf = vj[0] + vj[1] - vk
        gc2e += numpy.einsum('ij,ji', vhf[0], dma) * numpy.eye(3)
        gc2e -= numpy.einsum('ij,ji', vhf[1], dmb) * numpy.eye(3)

    gc2e *= (alpha2/2) / effspin
    return gc2e

def koseki_charge(z):
    '''Koseki effective charge in SO correction

    Ref:
    JPC 96, 10768
    JPC, 99, 12764
    JPCA, 102, 10430
    '''
    # JPC 96, 10768
    if z <= 2:
        return z
    elif z <= 10:
        return z * (.3 + z * .05)
    elif z <= 18:
        return z * (1.05 - z * .0125)
    else:
        return z


# Note mo10 is the imaginary part of MO^1
def para(mol, mo10, mo_coeff, mo_occ):
    alpha2 = lib.param.ALPHA ** 2
    effspin = mol.spin * .5
    nao = mo_coeff[0].shape[0]
    # FIXME: see JPC, 101, 3388 Eq (11c), why?
    g_so = (lib.param.G_ELECTRON - 1) * 2

    orboa = mo_coeff[0][:,mo_occ[0]>0]
    orbob = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(orboa, orboa.T)
    dm0b = numpy.dot(orbob, orbob.T)
    dm10a = numpy.asarray([reduce(numpy.dot, (mo_coeff[0], x, orboa.T)) for x in mo10[0]])
    dm10b = numpy.asarray([reduce(numpy.dot, (mo_coeff[1], x, orbob.T)) for x in mo10[1]])

# hso1e is the imaginary part of [i sigma dot pV x p]
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    if gobj.with_so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            #FIXME: when ECP is enabled
            Z = koseki_charge(mol.atom_charge(ia))
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor('int1e_pnucxp', 3)
    # <H^{01},MO^1> = - Tr(Im[H^{01}],Im[MO^1])
    gpara1e = -numpy.einsum('xji,yij->xy', dm10a, hso1e)
    gpara1e+=  numpy.einsum('xji,yij->xy', dm10b, hso1e)
    gpara1e *= 2 # *2 for + c.c.
    gpara1e *= g_so/2 * (alpha2/4) / effspin
    _write(gobj, gpara1e, 'SOC(1e)/OZ')

    if gobj.with_sso or gobj.with_soo:
        gpara2e = make_para_soc2e(gobj, (dm0a,dm0b), (dm10a,dm10b))
        _write(gobj, gpara2e, 'SOC(2e)/OZ')
    else:
        gpara2e = 0

    gpara = gpara1e + gpara2e
    return gpara

#TODO: option to use SOMF?  JCP 122, 034107
def make_para_soc2e(gobj, dm0, dm10):
    alpha2 = lib.param.ALPHA ** 2
    effspin = mol.spin * .5
    # FIXME: see JPC, 101, 3388 Eq (11c), why?
    g_so = (lib.param.G_ELECTRON - 1) * 2

    dm0a, dm0b = dm0
    dm10a, dm10b = dm10
    nao = dm0a.shape[0]

# hso2e is the imaginary part of SSO
# SSO term of JCP, 122, 034107 Eq (3) = 1/4c^2 hso2e
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3,nao,nao,nao,nao)
    ej = numpy.zeros((3,3))
    ek = numpy.zeros((3,3))
    if gobj.with_sso:
        ej += g_so/2 * numpy.einsum('yijkl,ji,xlk->xy', hso2e, dm0a-dm0b, dm10a+dm10b)
        ej += g_so/2 * numpy.einsum('yijkl,xji,lk->xy', hso2e, dm10a-dm10b, dm0a+dm0b)
        ek += g_so/2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0a, dm10a)
        ek -= g_so/2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0b, dm10b)
        ek += g_so/2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10a, dm0a)
        ek -= g_so/2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10b, dm0b)
    if gobj.with_soo:
        ej += 2 * numpy.einsum('yijkl,ji,xlk->xy', hso2e, dm0a+dm0b, dm10a-dm10b)
        ej += 2 * numpy.einsum('yijkl,xji,lk->xy', hso2e, dm10a+dm10b, dm0a-dm0b)
        ek += 2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0a, dm10a)
        ek -= 2 * numpy.einsum('yijkl,jk,xli->xy', hso2e, dm0b, dm10b)
        ek += 2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10a, dm0a)
        ek -= 2 * numpy.einsum('yijkl,xjk,li->xy', hso2e, dm10b, dm0b)
    gpara2e = -(ej - ek) * 2 # * 2 for + c.c.
    gpara2e *= (alpha2/4) / effspin
    return gpara2e

# FIXME: test the MB basis contribution
def make_h10(mol, dm0, gauge_orig=None, verbose=logger.WARN):
    log = logger.new_logger(mol, verbose=verbose)
    if gauge_orig is None:
        # A10_i dot p + p dot A10_i consistents with <p^2 g>
        # A10_j dot p + p dot A10_j consistents with <g p^2>
        # A10_j dot p + p dot A10_j => i/2 (rjxp - pxrj) = irjxp
        log.debug('First-order GIAO Fock matrix')
        h1 = -.5 * mol.intor('int1e_giao_irjxp', 3)
        h1 += uhf_nmr.make_h10giao(mol, dm0)
        if 0:  # from MB basis
            a10nucp = .5 * mol.intor('int1e_inuc_rxp', 3)
            h1 += a10nucp.transpose(0,2,1) - a10nucp
    else:
        mol.set_common_origin(gauge_orig)
        h1 = -.5 * mol.intor('int1e_cg_irxp', 3)
        if 0:  # from MB basis
            a10nucp = .5 * mol.intor('int1e_inuc_rcxp', 3)
            h1 += a10nucp.transpose(0,2,1) - a10nucp
        h1 = (h1, h1)
    return h1

def _write(gobj, gtensor, title, level=logger.INFO):
    if gobj.verbose >= level:
        w, v = numpy.linalg.eigh(numpy.dot(gtensor, gtensor.T))
        idxmax = abs(v).argmax(axis=0)
        v[:,v[idxmax,[0,1,2]]<0] *= -1  # format phase
        sorted_axis = numpy.argsort(idxmax)
        v = v[:,sorted_axis]
        if numpy.linalg.det(v) < 0: # ensure new axes in RHS
            v[:,2] *= -1
        g2 = reduce(numpy.dot, (v.T, gtensor, v))
        gobj.stdout.write('%s %s\n' % (title, g2.diagonal()))
        if gobj.verbose >= logger.DEBUG:
            rhf_nmr._write(gobj.stdout, gtensor, title+' tensor')


class GTensor(uhf_nmr.NMR):
    '''dE = B dot gtensor dot s'''
    def __init__(self, scf_method):
        self.with_sso = False  # Two-electron spin-same-orbit coupling
        self.with_soo = False  # Two-electron spin-other-orbit coupling
        self.with_so_eff_charge = True
        uhf_nmr.NMR.__init__(self, scf_method)

    def dump_flags(self):
        uhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'with_sso = %s (2e spin-same-orbit coupling)', self.with_sso)
        logger.info(self, 'with_soo = %s (2e spin-other-orbit coupling)', self.with_soo)
        logger.info(self, 'with_so_eff_charge = %s (1e SO effective charge)',
                    self.with_so_eff_charge)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        gdia = self.dia()
        gpara = self.para(mo10=mo1)
        gshift = gpara + gdia
        gtensor = gshift + numpy.eye(3) * lib.param.G_ELECTRON

        logger.timer(self, 'g-tensor', *cput0)
        if self.verbose > logger.QUIET:
            logger.note(self, 'free electron g %s', lib.param.G_ELECTRON)
            _write(gobj, gtensor, 'g-tensor', logger.NOTE)
            _write(gobj, gdia, 'g-tensor diamagnetic terms', logger.INFO)
            _write(gobj, gpara, 'g-tensor paramagnetic terms', logger.INFO)
            _write(gobj, gshift*1e3, 'g-shift (ppt)', logger.NOTE)
        self.stdout.flush()
        return gtensor

    def dia(self, mol=None, dm0=None, gauge_orig=None):
        if mol is None: mol = self.mol
        if gauge_orig is None: gauge_orig = self.gauge_orig
        if dm0 is None: dm0 = self._scf.make_rdm1()
        return dia(self, mol, dm0, gauge_orig)

    def para(self, mol=None, mo10=None, mo_coeff=None, mo_occ=None):
        if mol is None:           mol = self.mol
        if mo_coeff is None:      mo_coeff = self._scf.mo_coeff
        if mo_occ is None:        mo_occ = self._scf.mo_occ
        if mo10 is None:
            self.mo10, self.mo_e10 = self.solve_mo1()
            mo10 = self.mo10
        return para(mol, mo10, mo_coeff, mo_occ)


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    gobj = GTensor(mf)
    gobj.gauge_orig = (0,0,0)
    gobj.with_sso = True
    gobj.with_soo = True
    gobj.with_so_eff_charge = False
    print(gobj.kernel())
    exit()

    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    gobj = GTensor(mf)
    print(gobj.kernel())

    mol = gto.M(atom='''
                H 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    print(GTensor(mf).kernel())

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    print(GTensor(mf).kernel())

