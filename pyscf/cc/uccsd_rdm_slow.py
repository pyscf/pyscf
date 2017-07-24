#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jun Yang <junyang4711@gmail.com>
#

import time
import numpy
import h5py
import pyscf.lib as lib
from pyscf.lib import logger
from pyscf.cc import addons

einsum = numpy.einsum
#einsum = lib.einsum

def gamma1_intermediates(cc, t1, t2, l1, l2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

    dooa = numpy.zeros((nocca,nocca))
    doob = numpy.zeros((noccb,noccb))
    dooa += -numpy.einsum('ie,je->ij', t1a, l1a)
    dooa += -numpy.einsum('imef,jmef->ij', t2ab, l2ab)
    dooa += -numpy.einsum('imef,jmef->ij', t2aa, l2aa) * .5
    doob += -numpy.einsum('ie,je->ij', t1b, l1b)
    doob += -numpy.einsum('mief,mjef->ij', t2ab, l2ab)
    doob += -numpy.einsum('imef,jmef->ij', t2bb, l2bb) * .5

    dvva = numpy.zeros((nvira,nvira))
    dvvb = numpy.zeros((nvirb,nvirb))
    dvva += numpy.einsum('ma,mb->ab', l1a, t1a)
    dvva += numpy.einsum('mnae,mnbe->ab', l2ab, t2ab)
    dvva += numpy.einsum('mnae,mnbe->ab', l2aa, t2aa) * .5
    dvvb += numpy.einsum('ma,mb->ab', l1b, t1b)
    dvvb += numpy.einsum('mnea,mneb->ab', l2ab, t2ab)
    dvvb += numpy.einsum('mnae,mnbe->ab', l2bb, t2bb) * .5

    xt1a  = numpy.einsum('mnef,inef->mi', l2aa, t2aa)
    xt1a += numpy.einsum('mnef,inef->mi', l2ab, t2ab)
    xt2a  = numpy.einsum('mnaf,mnef->ae', t2aa, l2aa)
    xt2a += numpy.einsum('mnaf,mnef->ae', t2ab, l2ab)
    xtva  = numpy.einsum('ma,me->ae', t1a, l1a)

    dova  = numpy.zeros((nocca,nvira))
    dova += t1a
    dova += numpy.einsum('imae,me->ia', t2aa, l1a)
    dova += numpy.einsum('imae,me->ia', t2ab, l1b)
    dova -= numpy.einsum('ie,ae->ia', t1a, xtva)
    dova -= numpy.einsum('mi,ma->ia', xt1a, t1a) * .5
    dova -= numpy.einsum('ie,ae->ia', t1a, xt2a) * .5

    xt1b  = numpy.einsum('mnef,inef->mi', l2bb, t2bb)
    xt1b += numpy.einsum('nmef,nief->mi', l2ab, t2ab)
    xt2b  = numpy.einsum('mnaf,mnef->ae', t2bb, l2bb)
    xt2b += numpy.einsum('mnfa,mnfe->ae', t2ab, l2ab)
    xtvb  = numpy.einsum('ma,me->ae', t1b, l1b)

    dovb  = numpy.zeros((noccb,nvirb))
    dovb += t1b
    dovb += numpy.einsum('imae,me->ia', t2bb, l1b)
    dovb += numpy.einsum('miea,me->ia', t2ab, l1a)
    dovb -= numpy.einsum('ie,ae->ia', t1b, xtvb)
    dovb -= numpy.einsum('mi,ma->ia', xt1b, t1b) * .5
    dovb -= numpy.einsum('ie,ae->ia', t1b, xt2b) * .5

    dvoa = l1a.T.copy()
    dvob = l1b.T.copy()

    return (dooa, dova, dvoa, dvva), (doob, dovb, dvob, dvvb)

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(cc, t1, t2, l1, l2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape

#    tauaa = t2aa*.5 + numpy.einsum('ia,jb->ijab', t1a, t1a)
#    tauab = t2ab    + numpy.einsum('ia,jb->ijab', t1a, t1b)
#    taubb = t2bb*.5 + numpy.einsum('ia,jb->ijab', t1b, t1b)
#    gvvvv = numpy.einsum('ijab,ijcd->abcd', l2aa, tauaa) * .25
#    gvVvV = numpy.einsum('ijab,ijcd->abcd', l2ab, tauab) * .25
#    gVVVV = numpy.einsum('ijab,ijcd->abcd', l2bb, taubb) * .25
#
#    goooo = numpy.einsum('ijab,klab->klij', l2aa, tauaa) * .25
#    goOoO = numpy.einsum('ijab,klab->klij', l2ab, tauab) * .25
#    gOOOO = numpy.einsum('ijab,klab->klij', l2bb, taubb) * .25

    t1 = addons.spatial2spin(t1, cc.orbspin)
    t2 = addons.spatial2spin(t2, cc.orbspin)
    l1 = addons.spatial2spin(l1, cc.orbspin)
    l2 = addons.spatial2spin(l2, cc.orbspin)
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    miajb = einsum('ikac,kjcb->iajb', l2, t2)

    goovv = 0.25 * (l2 + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += 0.5 * einsum('cb,ijca->ijab', tmp, t2)
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += 0.5 * einsum('kiab,kj->ijab', tau, tmp)
    tmp = numpy.einsum('ldjd->lj', miajb)
    goovv -= einsum('lj,liba->ijab', tmp, t2) * .25
    goovv -= einsum('li,la,jb->ijab', tmp, t1, t1) * .5
    tmp = numpy.einsum('ldlb->db', miajb)
    goovv -= einsum('db,jida->ijab', tmp, t2) * .25
    goovv -= einsum('da,id,jb->ijab', tmp, t1, t1) * .5
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, tau)

    gvvvv = 0.125 * einsum('ijab,ijcd->abcd', l2, tau)
    goooo = 0.125 * einsum('klab,ijab->klij', tau, l2)

    gooov = -0.5 * einsum('jkba,ib->jkia', tau, l1)
    gooov += einsum('jkil,la->jkia', goooo, t1) * 2
    tmp = numpy.einsum('icjc->ij', miajb)
    gooov -= einsum('ij,ka->jkia', tmp, t1) * .5
    gooov += einsum('icja,kc->jkia', miajb, t1)
    gooov += einsum('jkab,ib->jkia', l2, t1) * .5

    govvo = einsum('ia,jb->jabi', l1, t1)
    govvo += numpy.einsum('iajb->jabi', miajb)
    govvo -= einsum('ikac,jc,kb->jabi', l2, t1, t1)

    gvovv = 0.5 * einsum('ja,jibc->aibc',l1,tau)
    gvovv -= einsum('adbc,id->aibc', gvvvv, t1) * 2
    tmp = numpy.einsum('kakb->ab', miajb)
    gvovv += einsum('ab,ic->aibc', tmp, t1) * .5
    gvovv -= einsum('kaib,kc->aibc', miajb, t1)
    gvovv -= 0.5 * einsum('ijbc,ja->aibc', l2, t1)

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dvvov = gvovv.transpose(0,2,1,3) - gvovv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    doovv = -numpy.einsum('jabi->jiab', govvo)
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, None, dooov)

def make_rdm1(cc, t1, t2, l1, l2, d1=None):
    if d1 is None:
        d1 = gamma1_intermediates(cc, t1, t2, l1, l2)
    d1a, d1b = d1
    dooa, dova, dvoa, dvva = d1a
    doob, dovb, dvob, dvvb = d1b

    nocca, nvira = dova.shape
    noccb, nvirb = dovb.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    dm1a = numpy.empty((nmoa,nmoa))
    dm1b = numpy.empty((nmoa,nmoa))
    dm1a[:nocca,:nocca] = dooa + dooa.T
    dm1a[:nocca,nocca:] = dova + dvoa.T
    dm1a[nocca:,:nocca] = dm1a[:nocca,nocca:].T
    dm1a[nocca:,nocca:] = dvva + dvva.T
    dm1b[:noccb,:noccb] = doob + doob.T
    dm1b[:noccb,noccb:] = dovb + dvob.T
    dm1b[noccb:,:noccb] = dm1b[:noccb,noccb:].T
    dm1b[noccb:,noccb:] = dvvb + dvvb.T

    for i in range(nocca):
        dm1a[i,i] += 1
    for i in range(noccb):
        dm1b[i,i] += 1
    return dm1a, dm1b

# spin-orbital rdm2 in Chemist's notation
def make_rdm2(cc, t1, t2, l1, l2, d1=None, d2=None):
    if d1 is None:
        d1 = gamma1_intermediates(cc, t1, t2, l1, l2)
    d1a, d1b = d1
    dooa, dova, dvoa, dvva = d1a
    doob, dovb, dvob, dvvb = d1b
    if d2 is None:
        d2 = gamma2_intermediates(cc, t1, t2, l1, l2)
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2

    nocca, nvira = dova.shape
    noccb, nvirb = dovb.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nocc = nocca + noccb
    nvir = nvira + nvirb
    nmo = nmoa + nmob

    dm2 = numpy.empty((nmo,nmo,nmo,nmo))

    dm2[:nocc,nocc:,:nocc,nocc:] = \
            (dovov                   +dovov.transpose(2,3,0,1))
    dm2[nocc:,:nocc,nocc:,:nocc] = \
            (dovov.transpose(1,0,3,2)+dovov.transpose(3,2,1,0))

    dm2[:nocc,:nocc,nocc:,nocc:] = \
            (doovv.transpose(0,1,3,2)+doovv.transpose(1,0,2,3))
    dm2[nocc:,nocc:,:nocc,:nocc] = \
            (doovv.transpose(3,2,0,1)+doovv.transpose(2,3,1,0))
    dm2[:nocc,nocc:,nocc:,:nocc] = \
            (dovvo                   +dovvo.transpose(3,2,1,0))
    dm2[nocc:,:nocc,:nocc,nocc:] = \
            (dovvo.transpose(2,3,0,1)+dovvo.transpose(1,0,3,2))

    dm2[nocc:,nocc:,nocc:,nocc:] = \
            (dvvvv                   +dvvvv.transpose(1,0,3,2)) * 2

    dm2[:nocc,:nocc,:nocc,:nocc] = \
            (doooo                   +doooo.transpose(1,0,3,2)) * 2

    dm2[nocc:,nocc:,:nocc,nocc:] = dvvov
    dm2[:nocc,nocc:,nocc:,nocc:] = dvvov.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dvvov.transpose(1,0,3,2)
    dm2[nocc:,:nocc,nocc:,nocc:] = dvvov.transpose(3,2,1,0)

    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2)
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0)

    dm1a = numpy.empty((nmoa,nmoa))
    dm1b = numpy.empty((nmoa,nmoa))
    dm1a[:nocca,:nocca] = dooa + dooa.T
    dm1a[:nocca,nocca:] = dova + dvoa.T
    dm1a[nocca:,:nocca] = dm1a[:nocca,nocca:].T
    dm1a[nocca:,nocca:] = dvva + dvva.T
    dm1b[:noccb,:noccb] = doob + doob.T
    dm1b[:noccb,noccb:] = dovb + dvob.T
    dm1b[noccb:,:noccb] = dm1b[:noccb,noccb:].T
    dm1b[noccb:,noccb:] = dvvb + dvvb.T

    oidxa = cc.orbspin[:nocc] == 0
    oidxb = cc.orbspin[:nocc] == 1
    vidxa = cc.orbspin[nocc:] == 0
    vidxb = cc.orbspin[nocc:] == 1
    doo = numpy.zeros((nocc,nocc))
    dov = numpy.zeros((nocc,nvir))
    dvv = numpy.zeros((nvir,nvir))
    doo[oidxa[:,None]&oidxa] = dooa.ravel()
    doo[oidxb[:,None]&oidxb] = doob.ravel()
    dov[oidxa[:,None]&vidxa] = dova.ravel()
    dov[oidxb[:,None]&vidxb] = dovb.ravel()
    dvv[vidxa[:,None]&vidxa] = dvva.ravel()
    dvv[vidxb[:,None]&vidxb] = dvvb.ravel()
    dvo = dov.T

    for i in range(nocc):
        dm2[i,i,:nocc,:nocc] += doo * 2
        dm2[:nocc,:nocc,i,i] += doo * 2
        dm2[i,i,nocc:,nocc:] += dvv * 2
        dm2[nocc:,nocc:,i,i] += dvv * 2
        dm2[:nocc,nocc:,i,i] += dov * 2
        dm2[i,i,:nocc,nocc:] += dov * 2
        dm2[nocc:,:nocc,i,i] += dvo * 2
        dm2[i,i,nocc:,:nocc] += dvo * 2
        dm2[:nocc,i,i,:nocc] -= doo
        dm2[i,:nocc,:nocc,i] -= doo
        dm2[nocc:,i,i,nocc:] -= dvv
        dm2[i,nocc:,nocc:,i] -= dvv
        dm2[:nocc,i,i,nocc:] -= dov
        dm2[i,:nocc,nocc:,i] -= dov
        dm2[nocc:,i,i,:nocc] -= dvo
        dm2[i,nocc:,:nocc,i] -= dvo

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= 1

    return dm2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import uccsd
    from pyscf.cc import addons

    nocc = 5
    nvir = 7
    mol = gto.M()
    mf = scf.UHF(mol)
    mf.mo_energy = [numpy.arange(nocc+nvir)]*2
    occ = numpy.zeros(nocc+nvir)
    occ[:nocc] = 1
    mf.mo_occ = [occ] * 2
    mycc = uccsd.UCCSD(mf)

    def antisym(t2):
        t2 = t2 - t2.transpose(0,1,3,2)
        t2 = t2 - t2.transpose(1,0,2,3)
        return t2
    numpy.random.seed(1)
    t1r = numpy.random.random((nocc,nvir))*.1
    t1 = (t1r,t1r)
    t2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
    t2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1)
    t2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1)
    t2 = (t2aa,t2r,t2bb)
    t1spin = addons.spatial2spin(t1)
    t2spin = addons.spatial2spin(t2)
    l1r = numpy.random.random((nocc,nvir))*.1
    l1 = (l1r,l1r)
    l2r = numpy.random.random((nocc,nocc,nvir,nvir))*.1
    l2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1)
    l2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1)
    l2 = (l2aa,l2r,l2bb)
    l1spin = addons.spatial2spin(l1)
    l2spin = addons.spatial2spin(l2)

#    import cclambda_rsp
#    mycc.t1 = t1spin
#    mycc.t2 = t2spin
#    mycc.L1 = l1spin
#    mycc.L2 = l2spin
#    doo, dvv = cclambda_rsp.Ddia(mycc)
#    d1a, d1b = gamma1_intermediates(mycc, t1, t2, l1, l2)
#    print(abs(doo[ ::2, ::2] - d1a[0]).max())
#    print(abs(doo[1::2,1::2] - d1b[0]).max())
#    print(abs(dvv[ ::2, ::2] - d1a[3]).max())
#    print(abs(dvv[1::2,1::2] - d1b[3]).max())
#
#    doovv,dvvvv,doooo,dooov,dovvo,dvovv = cclambda_rsp.Gamma(mycc)
#    d2 = gamma2_intermediates(mycc, t1, t2, l1, l2)
#    print(abs(doovv - d2[0]).max())
#    print(abs(dvvvv - d2[1]).max())
#    print(abs(doooo - d2[2]).max())
#    print(abs(dooov - d2[3]).max())
#    print(abs(dovvo - d2[4]).max())
#    print(abs(dvovv - d2[5]).max())

    dm1a, dm1b = make_rdm1(mycc, t1, t2, l1, l2)

    #TODO: test 1pdm, 2pdm against FCI

