#!/usr/bin/env python

'''
Restricted CCSD implementation which supports both real and complex integrals.
The 4-index integrals are saved on disk entirely (without using any symmetry).
This code is slower than the pyscf.cc.ccsd implementation.

Note MO integrals are treated in chemist's notation

Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import rintermediates as imd

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new  =-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo, order='C')
    t1new +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)

    # T2 equation
    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris.ovvv).conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1,0,3,2)
    t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3)
    if cc.cc2:
        Woooo2 = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
        Woooo2 += lib.einsum('lcki,jc->klij', eris_ovoo, t1)
        Woooo2 += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
        Woooo2 += lib.einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
        t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = lib.einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += np.asarray(eris.vvvv).transpose(0,2,1,3)
        t2new += lib.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = lib.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= np.diag(np.diag(foo))
        Lvv -= np.diag(np.diag(fvv))
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
        t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    return e.real


def make_intermediates(mycc, t1, t2, eris):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    tau = _ccsd.make_tau(t2, t1, t1)
    ovov = imd._get_ovov(eris)
    ovoo = np.asarray(eris.ovoo)
    ovov1 = ovov * 2 - ovov.transpose(0,3,2,1)
    ovoo1 = ovoo * 2 - ovoo.transpose(2,1,0,3)

    v1  = fvv - lib.einsum('ja,jb->ba', fov, t1)
    v1 -= lib.einsum('jakc,jkbc->ba', ovov1, tau)
    v2  = foo + lib.einsum('ib,jb->ij', fov, t1)
    v2 += lib.einsum('ibkc,jkbc->ij', ovov1, tau)
    v2 += np.einsum('kbij,kb->ij', ovoo1, t1)

    v5  = np.einsum('kc,jkbc->bj', fov, t2) * 2
    v5 -= np.einsum('kc,jkcb->bj', fov, t2)
    v5 += fvo
    tmp = fov + np.einsum('kcld,ld->kc', ovov1, t1)
    v5 += lib.einsum('kc,kb,jc->bj', tmp, t1, t1)
    v5 -= lib.einsum('lckj,klbc->bj', ovoo1, t2)

    oooo = np.asarray(eris.oooo)
    woooo  = lib.einsum('icjl,kc->ikjl', ovoo, t1)
    woooo += lib.einsum('jcil,kc->iljk', ovoo, t1)
    woooo += oooo.copy()
    woooo += lib.einsum('icjd,klcd->ikjl', ovov, tau)

    theta = t2*2 - t2.transpose(0,1,3,2)
    v4OVvo  = lib.einsum('ldjb,klcd->jbck', ovov1, t2)
    v4OVvo -= lib.einsum('ldjb,kldc->jbck', ovov, t2)
    v4OVvo += np.asarray(eris.ovvo)

    v4oVVo  = lib.einsum('jdlb,kldc->jbck', ovov, t2)
    v4oVVo -= np.asarray(eris.oovv).transpose(0,3,2,1)

    v4ovvo = v4OVvo*2 + v4oVVo
    w3  = np.einsum('jbck,jb->ck', v4ovvo, t1)

    woovo  = lib.einsum('ibck,jb->ijck', v4ovvo, t1)
    woovo = woovo - woovo.transpose(0,3,2,1)
    woovo += lib.einsum('ibck,jb->ikcj', v4OVvo-v4oVVo, t1)
    woovo += ovoo1.conj().transpose(3,2,1,0)

    woovo += lib.einsum('lcik,jlbc->ikbj', ovoo1, theta)
    woovo -= lib.einsum('lcik,jlbc->ijbk', ovoo1, t2)
    woovo -= lib.einsum('iclk,ljbc->ijbk', ovoo1, t2)

    wvvvo  = lib.einsum('jack,jb->back', v4ovvo, t1)
    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)
    wvvvo += lib.einsum('jack,jb->cabk', v4OVvo-v4oVVo, t1)
    wvvvo -= lib.einsum('lajk,jlbc->cabk', ovoo1, tau)

    wOVvo  = v4OVvo
    woVVo  = v4oVVo
    wOVvo -= np.einsum('jbld,kd,lc->jbck', ovov, t1, t1)
    woVVo += np.einsum('jdlb,kd,lc->jbck', ovov, t1, t1)
    wOVvo -= lib.einsum('jblk,lc->jbck', ovoo, t1)
    woVVo += lib.einsum('lbjk,lc->jbck', ovoo, t1)
    v4ovvo = v4OVvo = v4oVVo = None

    ovvv = imd._get_ovvv(eris)
    wvvvo += lib.einsum('kacd,kjbd->bacj', ovvv, t2) * 1.5

    wOVvo += lib.einsum('jbcd,kd->jbck', ovvv, t1)
    woVVo -= lib.einsum('jdcb,kd->jbck', ovvv, t1)

    ovvv = ovvv*2 - ovvv.transpose(0,3,2,1)
    v1 += np.einsum('jcba,jc->ba', ovvv, t1)
    v5 += lib.einsum('kdbc,jkcd->bj', ovvv, t2)
    woovo += lib.einsum('idcb,jkdb->ijck', ovvv, tau)

    tmp = lib.einsum('kdca,jkbd->cabj', ovvv, theta)
    wvvvo -= tmp
    wvvvo += tmp.transpose(2,1,0,3) * .5
    wvvvo -= ovvv.conj().transpose(3,2,1,0)
    ovvv = tmp = None

    w3 += v5
    w3 += np.einsum('cb,jb->cj', v1, t1)
    w3 -= np.einsum('jk,jb->bk', v2, t1)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), 'f8')
    imds.woVVo = imds.ftmp.create_dataset('woVVo', (nocc,nvir,nvir,nocc), 'f8')
    imds.woovo = imds.ftmp.create_dataset('woovo', (nocc,nocc,nvir,nocc), 'f8')
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvir,nvir,nvir,nocc), 'f8')

    imds.woooo[:] = woooo
    imds.wovvo[:] = wOVvo*2 + woVVo
    imds.woVVo[:] = woVVo
    imds.woovo[:] = woovo
    imds.wvvvo[:] = wvvvo
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time1 = time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    l1new = np.zeros_like(l1)

    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]

    tau = _ccsd.make_tau(t2, t1, t1)

    theta = t2*2 - t2.transpose(0,1,3,2)
    mvv = lib.einsum('klca,klcb->ba', l2, theta)
    moo = lib.einsum('kicd,kjcd->ij', l2, theta)
    mvv1 = lib.einsum('jc,jb->bc', l1, t1) + mvv
    moo1 = lib.einsum('ic,kc->ik', l1, t1) + moo

    m3 = mycc._add_vvvv(None, l2, eris, with_ovvv=False, t2sym='jiba')
    m3 += np.einsum('klab,ikjl->ijab', l2, imds.woooo)
    m3 *= .5

    ovov = np.asarray(eris.ovov)
    l2tau = np.einsum('ijcd,klcd->ijkl', l2, tau)
    m3 += np.einsum('kalb,ijkl->ijab', ovov, l2tau) * .5
    l2tau = None

    l2new = ovov.transpose(0,2,1,3) * .5
    l2new += lib.einsum('ijac,cb->ijab', l2, imds.v1)
    l2new -= lib.einsum('ikab,jk->ijab', l2, imds.v2)
    l2new -= lib.einsum('ca,icjb->ijab', mvv1, ovov)
    l2new -= lib.einsum('ik,kajb->ijab', moo1, ovov)

    ovov = ovov * 2 - ovov.transpose(0,3,2,1)
    fov1 = fov + np.einsum('jbkc,kc->jb', ovov, t1)
    l1new -= np.einsum('ik,ka->ia', moo, fov1)
    l1new -= np.einsum('ca,ic->ia', mvv, fov1)
    l2new += np.einsum('ia,jb->ijab', l1, fov1)

    tmp  = t1 + np.einsum('kc,kjcb->jb', l1, theta)
    tmp -= lib.einsum('bd,jd->jb', mvv1, t1)
    tmp -= lib.einsum('lj,lb->jb', moo, t1)
    l1new += np.einsum('jbia,jb->ia', ovov, tmp)
    ovov = tmp = None

    ovvv = imd._get_ovvv(eris)
    l1new += np.einsum('iacb,bc->ia', ovvv, mvv1) * 2
    l1new -= np.einsum('ibca,bc->ia', ovvv, mvv1)
    l2new += lib.einsum('ic,jbca->jiba', l1, ovvv)
    l2t1 = np.einsum('ijcd,kd->ijck', l2, t1)
    m3 -= np.einsum('kbca,ijck->ijab', ovvv, l2t1)
    l2t1 = ovvv = None

    l2new += m3
    l1new += np.einsum('ijab,jb->ia', m3, t1) * 2
    l1new += np.einsum('jiba,jb->ia', m3, t1) * 2
    l1new -= np.einsum('ijba,jb->ia', m3, t1)
    l1new -= np.einsum('jiab,jb->ia', m3, t1)

    ovoo = np.asarray(eris.ovoo)
    l1new -= np.einsum('iajk,kj->ia', ovoo, moo1) * 2
    l1new += np.einsum('jaik,kj->ia', ovoo, moo1)
    l2new -= lib.einsum('ka,jbik->ijab', l1, ovoo)
    ovoo = None

    l2theta = l2*2 - l2.transpose(0,1,3,2)
    l2new += lib.einsum('ikac,jbck->ijab', l2theta, imds.wovvo) * .5
    tmp = lib.einsum('ikca,jbck->ijab', l2, imds.woVVo)
    l2new += tmp * .5
    l2new += tmp.transpose(1,0,2,3)
    l2theta = None

    l1new += fov
    l1new += lib.einsum('ib,ba->ia', l1, imds.v1)
    l1new -= lib.einsum('ja,ij->ia', l1, imds.v2)

    l1new += np.einsum('jb,iabj->ia', l1, eris.ovvo) * 2
    l1new -= np.einsum('jb,ijba->ia', l1, eris.oovv)

    l1new -= lib.einsum('ijbc,bacj->ia', l2, imds.wvvvo)
    l1new -= lib.einsum('kjca,ijck->ia', l2, imds.woovo)

    l1new += np.einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= np.einsum('ijba,bj->ia', l2, imds.w3)

    eia = lib.direct_sum('i-j->ij', foo.diagonal(), fvv.diagonal())
    l1new /= eia
    l1new += l1

    l2new = l2new + l2new.transpose(1,0,3,2)
    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


class RCCSD(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])

    def init_amps(self, eris):
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e = eris.fock.diagonal().real
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:].conj() / eia
        eris_ovov = np.asarray(eris.ovov)
        t2 = eris_ovov.transpose(0,2,1,3).conj() / eijab
        self.emp2  = 2*np.einsum('ijab,iajb', t2, eris_ovov)
        self.emp2 -=   np.einsum('ijab,ibja', t2, eris_ovov)
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            pt = ccsd.mp2.MP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError
            #return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)

    energy = energy
    update_amps = update_amps

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=False, t2sym=None):
        assert(not self.direct)
        return ccsd.CCSD._add_vvvv(self, t1, t2, eris, out, with_ovvv, t2sym)


    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose,
                                   fintermediates=_make_lambda_intermediates,
                                   fupdate=update_lambda)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
#?        # Note
#?        assert(t1.dtype == np.double)
#?        assert(t2.dtype == np.double)
        return ccsd.CCSD.ccsd_t(self, t1, t2, eris)

def _contract_vvvv_t2(mol, vvvv, t2, out=None, max_memory=2000, verbose=None):
    '''Ht2 = np.einsum('ijcd,acbd->ijab', t2, vvvv)

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    if vvvv is None:   # AO-direct CCSD
        assert(t2.dtype == np.double)
        return ccsd._contract_vvvv_t2(mol, vvvv, t2, out, max_memory, verbose)

    time0 = time.clock(), time.time()
    log = logger.new_logger(mol, verbose)

    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    Ht2 = np.ndarray(x2.shape, buffer=out)

    unit = nvirb**2*nvira*2 + nocc2*nvirb
    blksize = min(nvira, max(ccsd.BLKMIN, int(max_memory*1e6/8/unit)))

    for p0,p1 in lib.prange(0, nvira, blksize):
        Ht2[:,p0:p1] = lib.einsum('xcd,acbd->xab', x2, vvvv[p0:p1])
        time0 = log.timer_debug1('vvvv [%d:%d]' % (p0,p1), *time0)
    return Ht2.reshape(t2.shape)

class _ChemistsERIs(ccsd._ChemistsERIs):
    def _contract_vvvv_t2(self, t2, direct=False, out=None, max_memory=2000,
                          verbose=None):
        if direct:
            vvvv = None
        else:
            vvvv = self.vvvv
        return _contract_vvvv_t2(self.mol, vvvv, t2, out, max_memory, verbose)

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'f8', chunks=(nocc,1,nvir,nvir))
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'f8')
    max_memory = max(2000, mycc.max_memory-lib.current_memory()[0])

    ftmp = lib.H5TmpFile()
    ao2mo.full(mol, mo_coeff, ftmp, max_memory=max_memory, verbose=log)
    eri = ftmp['eri_mo']

    nocc_pair = nocc*(nocc+1)//2
    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    oo = eri[:nocc_pair]
    eris.oooo[:] = ao2mo.restore(1, oo[:,:nocc_pair], nocc)
    oovv = lib.take_2d(oo, tril2sq[:nocc,:nocc].ravel(), tril2sq[nocc:,nocc:].ravel())
    eris.oovv[:] = oovv.reshape(nocc,nocc,nvir,nvir)
    oo = oovv = None

    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    blksize = min(nvir, max(2, int(max_memory*1e6/8/nmo**3/2)))
    for p0, p1 in lib.prange(0, nvir, blksize):
        q0, q1 = p0+nocc, p1+nocc
        off0 = q0*(q0+1)//2
        off1 = q1*(q1+1)//2
        buf = lib.unpack_tril(eri[off0:off1])

        tmp = buf[ tril2sq[q0:q1,:nocc] - off0 ]
        eris.ovoo[:,p0:p1] = tmp[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = tmp[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = tmp[:,:,:nocc,nocc:].transpose(1,0,2,3)
        eris.ovvv[:,p0:p1] = tmp[:,:,nocc:,nocc:].transpose(1,0,2,3)

        tmp = buf[ tril2sq[q0:q1,nocc:q1] - off0 ]
        eris.vvvv[p0:p1,:p1] = tmp[:,:,nocc:,nocc:]
        if p0 > 0:
            eris.vvvv[:p0,p0:p1] = tmp[:,:p0,nocc:,nocc:].transpose(1,0,2,3)
        buf = tmp = None
    log.timer('CCSD integral transformation', *cput0)
    return eris


if __name__ == '__main__':
    #from pyscf import scf
    #from pyscf import gto

    #mol = gto.Mole()
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]
    #mol.basis = 'cc-pvdz'
    ##mol.basis = '3-21G'
    #mol.verbose = 0
    #mol.spin = 0
    #mol.build()
    #mf = scf.RHF(mol).run(conv_tol=1e-14)

    #mycc = RCCSD(mf)
    #mycc.max_memory = 0
    #eris = mycc.ao2mo()
    #emp2, t1, t2 = mycc.init_amps(eris)
    #print(lib.finger(t2) - 0.044540097905897198)
    #np.random.seed(1)
    #t1 = np.random.random(t1.shape)*.1
    #t2 = np.random.random(t2.shape)*.1
    #t2 = t2 + t2.transpose(1,0,3,2)
    #t1, t2 = update_amps(mycc, t1, t2, eris)
    #print(lib.finger(t1) - 0.25118555558133576)
    #print(lib.finger(t2) - 0.02352137419932243)

    #ecc, t1, t2 = mycc.kernel()
    #print(ecc - -0.21334326214236796)

    #e, v = mycc.ipccsd(nroots=3)
    #print(e[0] - 0.43356041409195489)
    #print(e[1] - 0.51876598058509493)
    #print(e[2] - 0.6782879569941862 )

    #e, v = mycc.eeccsd(nroots=4)
    #print(e[0] - 0.2757159395886167)
    #print(e[1] - 0.2757159395886167)
    #print(e[2] - 0.2757159395886167)
    #print(e[3] - 0.3005716731825082)

    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import ccsd
    from pyscf.cc import ccsd_lambda

    mol = gto.M()
    mf = scf.RHF(mol)

    mcc = ccsd.CCSD(mf)

    np.random.seed(12)
    nocc = 5
    nmo = 12
    nvir = nmo - nocc
    eri0 = np.random.random((nmo,nmo,nmo,nmo))
    eri0 = ao2mo.restore(1, ao2mo.restore(8, eri0, nmo), nmo)
    fock0 = np.random.random((nmo,nmo))
    fock0 = fock0 + fock0.T + np.diag(range(nmo))*2
    t1 = np.random.random((nocc,nvir))
    t2 = np.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    l1 = np.random.random((nocc,nvir))
    l2 = np.random.random((nocc,nocc,nvir,nvir))
    l2 = l2 + l2.transpose(1,0,3,2)

    eris = ccsd._ChemistsERIs()
    eris.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
    eris.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
    idx = np.tril_indices(nvir)
    eris.ovvv = eri0[:nocc,nocc:,nocc:,nocc:][:,:,idx[0],idx[1]].copy()
    eris.vvvv = ao2mo.restore(4,eri0[nocc:,nocc:,nocc:,nocc:],nvir)
    eris.fock = fock0

    imds = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_lambda(mcc, t1, t2, l1, l2, eris, imds)
    print(lib.finger(l1new) - -6699.5335665027187)
    print(lib.finger(l2new) - -514.7001243502192 )
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())

    mcc.max_memory = 0
    imds = make_intermediates(mcc, t1, t2, eris)
    l1new, l2new = update_lambda(mcc, t1, t2, l1, l2, eris, imds)
    print(lib.finger(l1new) - -6699.5335665027187)
    print(lib.finger(l2new) - -514.7001243502192 )
    print(abs(l2new-l2new.transpose(1,0,3,2)).sum())

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-16
    rhf.scf()

    mcc = ccsd.CCSD(rhf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()

    nmo = rhf.mo_energy.size
    fock0 = np.diag(rhf.mo_energy)
    nocc = mol.nelectron // 2
    nvir = nmo - nocc

    eris = mcc.ao2mo()
    conv, l1, l2 = ccsd_lambda.kernel(mcc, eris, t1, t2, tol=1e-8)
    print(np.linalg.norm(l1)-0.0132626841292)
    print(np.linalg.norm(l2)-0.212575609057)

    from pyscf.cc import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(np.dot, (rhf.mo_coeff.T, rhf.get_hcore(), rhf.mo_coeff))
    eri = ao2mo.full(rhf._eri, rhf.mo_coeff)
    eri = ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = np.einsum('pq,pq', h1, dm1)
    e2 = np.einsum('pqrs,pqrs', eri, dm2) * .5
    print(e1+e2+mol.energy_nuc() - rhf.e_tot - ecc)
