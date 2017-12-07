#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger

einsum = lib.einsum

def kernel(mycc, eris, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2

    imds = make_intermediates(mycc, t1, t2, eris)

    if mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file)
        adiis.space = mycc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1, t2)
    cput0 = log.timer('UCCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = update_amps(mycc, t1, t2, l1, l2, eris, imds)
        normt = numpy.linalg.norm(l1new-l1) + numpy.linalg.norm(l2new-l2)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        if mycc.diis:
            l1, l2 = mycc.diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('UCCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2


# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2

    v1 = fvv - einsum('ja,jb->ba', fov, t1)
    v1-= einsum('jbac,jc->ba', eris.ovvv, t1)
    v1+= einsum('jkca,jkbc->ba', eris.oovv, tau) * .5

    v2 = foo + einsum('ib,jb->ij', fov, t1)
    v2-= einsum('kijb,kb->ij', eris.ooov, t1)
    v2+= einsum('ikbc,jkbc->ij', eris.oovv, tau) * .5

    v3 = einsum('ijcd,klcd->ijkl', eris.oovv, tau)
    v4 = einsum('ljdb,klcd->jcbk', eris.oovv, t2)
    v4+= numpy.asarray(eris.ovvo)

    v5 = fvo + numpy.einsum('kc,jkbc->bj', fov, t2)
    tmp = fov - numpy.einsum('kldc,ld->kc', eris.oovv, t1)
    v5+= numpy.einsum('kc,kb,jc->bj', tmp, t1, t1)
    v5-= einsum('kljc,klbc->bj', eris.ooov, t2) * .5
    v5+= einsum('kbdc,jkcd->bj', eris.ovvv, t2) * .5

    w3 = v5 + numpy.einsum('jcbk,jb->ck', v4, t1)
    w3 += numpy.einsum('cb,jb->cj', v1, t1)
    w3 -= numpy.einsum('jk,jb->bk', v2, t1)

    woooo = numpy.asarray(eris.oooo) * .5
    woooo+= v3 * .25
    woooo+= einsum('jilc,kc->ijkl', eris.ooov, t1)

    wovvo = v4 - numpy.einsum('ljdb,lc,kd->jcbk', eris.oovv, t1, t1)
    wovvo-= einsum('ljkb,lc->jcbk', eris.ooov, t1)
    wovvo+= einsum('jcbd,kd->jcbk', eris.ovvv, t1)

    woovo = einsum('icdb,kjbd->kjci', eris.ovvv, tau) * .25
    woovo+= numpy.einsum('jkic->kjci', numpy.asarray(eris.ooov).conj()) * .5
    woovo+= einsum('icbk,jb->kjci', v4, t1)

    wovvv = einsum('jlka,jlbc->kbca', eris.ooov, tau) * .25
    wovvv-= numpy.einsum('jacb->jbca', numpy.asarray(eris.ovvv).conj()) * .5
    wovvv+= einsum('jcak,jb->kbca', v4, t1)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), 'f8')
    imds.woovo = imds.ftmp.create_dataset('woovo', (nocc,nocc,nvir,nocc), 'f8')
    imds.wovvv = imds.ftmp.create_dataset('wovvv', (nocc,nvir,nvir,nvir), 'f8')
    imds.woooo[:] = woooo
    imds.wovvo[:] = wovvo
    imds.woovo[:] = woovo
    imds.wovvv[:] = wovvv
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_amps(mycc, t1, t2, l1, l2, eris, imds):
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc * nvir
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    l1new = numpy.zeros_like(l1)
    l2new = numpy.zeros_like(l2)

    mba = einsum('klca,klcb->ba', l2, t2) * .5
    mij = einsum('kicd,kjcd->ij', l2, t2) * .5
    m3 = numpy.einsum('klab,ijkl->ijab', l2, numpy.asarray(imds.woooo))
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    tmp = numpy.einsum('ijcd,klcd->ijkl', l2, tau)
    oovv = numpy.asarray(eris.oovv)
    m3 += numpy.einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = numpy.einsum('ijcd,kd->ijck', l2, t1)
    m3 -= numpy.einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    m3 += numpy.einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

    l2new += oovv
    l2new += m3
    fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
    tmp = einsum('ia,jb->ijab', l1, fov1)
    tmp+= einsum('kica,jcbk->ijab', l2, numpy.asarray(imds.wovvo))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,ijkb->ijab', l1, eris.ooov)
    tmp+= einsum('ijca,cb->ijab', l2, imds.v1)
    tmp1vv = mba + einsum('ka,kb->ba', l1, t1)
    tmp+= einsum('ca,ijcb->ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ic,jcba->jiba', l1, eris.ovvv)
    tmp+= einsum('kiab,jk->ijab', l2, imds.v2)
    tmp1oo = mij + einsum('ic,kc->ik', l1, t1)
    tmp-= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)

    l1new += fov
    l1new += einsum('jb,ibaj->ia', l1, eris.ovvo)
    l1new += einsum('ib,ba->ia', l1, imds.v1)
    l1new -= einsum('ja,ij->ia', l1, imds.v2)
    l1new -= einsum('kjca,kjci->ia', l2, imds.woovo)
    l1new -= einsum('ikbc,kbca->ia', l2, imds.wovvv)
    l1new += einsum('ijab,jb->ia', m3, t1)
    l1new += einsum('jiba,bj->ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('bd,jd->jb', tmp1vv, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += numpy.einsum('jiba,jb->ia', oovv, tmp)
    l1new += numpy.einsum('icab,bc->ia', eris.ovvv, tmp1vv)
    l1new -= numpy.einsum('jika,kj->ia', eris.ooov, tmp1oo)
    tmp = fov - einsum('kjba,jb->ka', oovv, t1)
    l1new -= numpy.einsum('ik,ka->ia', mij, tmp)
    l1new -= numpy.einsum('ca,ic->ia', mba, tmp)

    tmp = einsum('kcad,jkbd->jacb', eris.ovvv, t2)
    l1new -= einsum('ijcb,jacb->ia', l2, tmp)
    tmp = einsum('likc,jlbc->jkib', eris.ooov, t2)
    l1new += einsum('kjab,jkib->ia', l2, tmp)

    mo_e = eris.fock.diagonal().real
    eia = lib.direct_sum('i-j->ij', mo_e[:nocc], mo_e[nocc:])
    l1new /= eia
    l1new += l1
    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    mf0 = mf
    mf = scf.addons.convert_to_ghf(mf)
    mycc = gccsd.GCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    conv, l1, l2 = kernel(mycc, eris, mycc.t1, mycc.t2, tol=1e-8)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(lib.finger(l1[0]) --0.0030030170069977758)
    print(lib.finger(l1[1]) --0.0030030170069977758)
    print(lib.finger(l2[0]) --0.041444910588788492 )
    print(lib.finger(l2[1]) - 0.1077575086912813   )
    print(lib.finger(l2[2]) --0.041444910588788492 )
    print(abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
    print(abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[0]).max())

    from pyscf.cc import ccsd, ccsd_lambda
    mycc0 = ccsd.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_lambda.update_amps(mycc0, t1, t2, t1, t2, eris0, imds)
    l1ref, l2ref = ccsd_lambda.update_amps(mycc0, t1, t2, l1, l2, eris0, imds)
    t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
    t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
    l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_amps(mycc, t1, t2, l1, l2, eris, imds)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(abs(l1[0]-l1ref).max())
    print(abs(l2[1]-l2ref).max())
