#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jun Yang <junyang4711@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf.lib import logger

#einsum = numpy.einsum
einsum = lib.einsum

def gamma1_intermediates(mycc, t1, t2, l1, l2):
    doo  =-einsum('ie,je->ij', l1, t1)
    doo -= einsum('imef,jmef->ij', l2, t2) * .5

    dvv  = einsum('ma,mb->ab', t1, l1)
    dvv += einsum('mnea,mneb->ab', t2, l2) * .5

    xt1  = einsum('mnef,inef->mi', l2, t2) * .5
    xt2  = einsum('mnfa,mnfe->ae', t2, l2) * .5
    xt2 += einsum('ma,me->ae', t1, l1)
    dvo  = einsum('imae,me->ai', t2, l1)
    dvo -= einsum('mi,ma->ai', xt1, t1)
    dvo -= einsum('ie,ae->ai', t1, xt2)
    dvo += t1.T

    dov = l1

    return doo, dov, dvo, dvv

# gamma2 intermediates in Chemist's notation
def gamma2_intermediates(mycc, t1, t2, l1, l2):
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    miajb = einsum('ikac,kjcb->iajb', l2, t2)

    goovv = 0.25 * (l2.conj() + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
    tmp = numpy.einsum('ldjd->lj', miajb)
    goovv -= einsum('lj,liba->ijab', tmp, tau) * .25
    tmp = numpy.einsum('ldlb->db', miajb)
    goovv -= einsum('db,jida->ijab', tmp, tau) * .25
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, tau)
    goovv = goovv.conj()

    gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
    goooo = einsum('klab,ijab->klij', l2, tau) * 0.125

    gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
    gooov += einsum('iljk,la->jkia', goooo, t1)
    tmp = numpy.einsum('icjc->ij', miajb) * .25
    gooov -= einsum('ij,ka->jkia', tmp, t1)
    gooov += einsum('icja,kc->jkia', miajb, t1) * .5
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2, t1) * .25

    govvo  = einsum('ia,jb->ibaj', l1, t1)
    govvo += numpy.einsum('iajb->ibaj', miajb)
    govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)

    govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
    govvv += einsum('bcad,id->iabc', gvvvv, t1)
    tmp = numpy.einsum('kakb->ab', miajb) * .25
    govvv += einsum('ab,ic->iacb', tmp, t1)
    govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2, t1) * .25

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

def make_rdm1(mycc, t1, t2, l1, l2, d1=None):
    if d1 is None:
        d1 = gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir

    dm1 = numpy.empty((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T
    dm1 *= .5
    dm1[numpy.diag_indices(nocc)] += 1

    return dm1

# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, d1=None, d2=None):
    if d1 is None: d1 = gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    nocc, nvir = dov.shape
    nmo = nocc + nvir

    dm2 = _make_rdm2(mycc, t1, t2, l1, l2, d2)

    dm1 = numpy.empty((nmo,nmo))
    dm1[:nocc,:nocc] = doo + doo.conj().T
    dm1[:nocc,nocc:] = dov + dvo.conj().T
    dm1[nocc:,:nocc] = dm1[:nocc,nocc:].conj().T
    dm1[nocc:,nocc:] = dvv + dvv.conj().T
    dm1 *= .5

    if mycc.frozen is not 0:
        raise NotImplementedError

    for i in range(nocc):
        dm2[i,i,:,:] += dm1
        dm2[:,:,i,i] += dm1
        dm2[:,i,i,:] -= dm1
        dm2[i,:,:,i] -= dm1

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= 1

    return dm2

def _make_rdm2(mycc, t1, t2, l1, l2, d2=None):
    '''The 2-PDM associated to the normal ordered 2-particle interactions
    '''
    if d2 is None:
        d2 = gamma2_intermediates(mycc, t1, t2, l1, l2)
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dovov.shape[:2]
    nmo = nocc + nvir

    dm2 = numpy.empty((nmo,nmo,nmo,nmo))

    dovov = numpy.asarray(dovov)
    dm2[:nocc,nocc:,:nocc,nocc:] = dovov
    dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()
    dovov = None

    dovvo = numpy.asarray(dovvo)
    dm2[:nocc,:nocc,nocc:,nocc:] =-dovvo.transpose(0,3,2,1)
    dm2[nocc:,nocc:,:nocc,:nocc] =-dovvo.transpose(2,1,0,3)
    dm2[:nocc,nocc:,nocc:,:nocc] = dovvo
    dm2[nocc:,:nocc,:nocc,nocc:] = dovvo.transpose(1,0,3,2).conj()
    dovvo = None

    dm2[nocc:,nocc:,nocc:,nocc:] = dvvvv
    dm2[:nocc,:nocc,:nocc,:nocc] = doooo

    dovvv = numpy.asarray(dovvv)
    dm2[:nocc,nocc:,nocc:,nocc:] = dovvv
    dm2[nocc:,nocc:,:nocc,nocc:] = dovvv.transpose(2,3,0,1)
    dm2[nocc:,nocc:,nocc:,:nocc] = dovvv.transpose(3,2,1,0).conj()
    dm2[nocc:,:nocc,nocc:,nocc:] = dovvv.transpose(1,0,3,2).conj()
    dovvv = None

    dooov = numpy.asarray(dooov)
    dm2[:nocc,:nocc,:nocc,nocc:] = dooov
    dm2[:nocc,nocc:,:nocc,:nocc] = dooov.transpose(2,3,0,1)
    dm2[:nocc,:nocc,nocc:,:nocc] = dooov.transpose(1,0,3,2).conj()
    dm2[nocc:,:nocc,:nocc,:nocc] = dooov.transpose(3,2,1,0).conj()
    return dm2


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import gccsd
    from pyscf.cc import addons

    nocc = 6
    nvir = 10
    mol = gto.M()
    mol.nelectron = nocc
    mf = scf.GHF(mol)
    nmo = nocc + nvir
    npair = nmo*(nmo//2+1)//4
    numpy.random.seed(12)
    mf._eri = numpy.random.random(npair*(npair+1)//2)*.3
    hcore = numpy.random.random((nmo,nmo)) * .5
    hcore = hcore + hcore.T + numpy.diag(range(nmo))*2
    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: numpy.eye(nmo)
    mf.mo_coeff = numpy.eye(nmo)
    mf.mo_occ = numpy.zeros(nmo)
    mf.mo_occ[:nocc] = 1
    dm1 = mf.make_rdm1()
    mf.e_tot = mf.energy_elec()[0]
    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    nao = nmo // 2
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    eri  = ao2mo.kernel(mf._eri, mo_a)
    eri += ao2mo.kernel(mf._eri, mo_b)
    eri1 = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b))
    eri += eri1
    eri += eri1.T
    eri = ao2mo.restore(1, eri, nmo)
    h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,jilk', eri, dm2) * .5
    print(e1 - mycc.e_tot)


    def antisym(t2):
        t2 = t2 - t2.transpose(0,1,3,2)
        t2 = t2 - t2.transpose(1,0,2,3)
        return t2
    numpy.random.seed(1)
    t1 = numpy.random.random((2,nocc,nvir))*.1 - .1
    t2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
    t2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
    t2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
    t2 = (t2aa,t2ab,t2bb)
    t1 = addons.spatial2spin(t1)
    t2 = addons.spatial2spin(t2)
    l1 = numpy.random.random((2,nocc,nvir))*.1 - .1
    l2ab = numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1
    l2aa = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
    l2bb = antisym(numpy.random.random((nocc,nocc,nvir,nvir))*.1 - .1)
    l2 = (l2aa,l2ab,l2bb)
    l1 = addons.spatial2spin(l1)
    l2 = addons.spatial2spin(l2)

    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    print(abs(dm2-dm2.transpose(1,0,3,2).conj()).max())
    print(abs(dm2-dm2.transpose(2,3,0,1)       ).max())
    print(abs(dm2+dm2.transpose(2,1,0,3)       ).max())
    print(abs(dm2+dm2.transpose(0,3,2,1)       ).max())

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = gccsd.GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    dm1 = make_rdm1(mycc, t1, t2, l1, l2)
    dm2 = make_rdm2(mycc, t1, t2, l1, l2)
    nao = mol.nao_nr()
    mo_a = mf.mo_coeff[:nao]
    mo_b = mf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(mf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = mf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,jilk', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mycc.e_tot)

    #TODO: test 1pdm, 2pdm against FCI
