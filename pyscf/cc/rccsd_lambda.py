#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Restricted CCSD lambda equation solver which supports both real and complex
integrals.  This code is slower than the pyscf.cc.ccsd_lambda implementation.

Note MO integrals are treated in chemist's notation
'''


import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_lambda

einsum = lib.einsum

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)


def make_intermediates(mycc, t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    tau = _ccsd.make_tau(t2, t1, t1)
    ovov = np.asarray(eris.ovov)
    ovoo = np.asarray(eris.ovoo)
    ovov1 = ovov * 2 - ovov.transpose(0,3,2,1)
    ovoo1 = ovoo * 2 - ovoo.transpose(2,1,0,3)

    v1  = fvv - lib.einsum('ja,jb->ba', fov, t1)
    v1 -= lib.einsum('jakc,jkbc->ba', ovov1, tau)
    v2  = foo + lib.einsum('ib,jb->ij', fov, t1)
    v2 += lib.einsum('ibkc,jkbc->ij', ovov1, tau)
    v2 += np.einsum('kbij,kb->ij', ovoo1, t1)
    v4 = fov + np.einsum('jbkc,kc->jb', ovov1, t1)

    v5  = np.einsum('kc,jkbc->bj', fov, t2) * 2
    v5 -= np.einsum('kc,jkcb->bj', fov, t2)
    v5 += fvo
    v5 += lib.einsum('kc,kb,jc->bj', v4, t1, t1)
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
    w3 = np.einsum('jbck,jb->ck', v4ovvo, t1)

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

    ovvv = np.asarray(eris.get_ovvv())
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
    dtype = np.result_type(t2, eris.vvvv).char
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), dtype)
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), dtype)
    imds.woVVo = imds.ftmp.create_dataset('woVVo', (nocc,nvir,nvir,nocc), dtype)
    imds.woovo = imds.ftmp.create_dataset('woovo', (nocc,nocc,nvir,nocc), dtype)
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvir,nvir,nvir,nocc), dtype)

    imds.woooo[:] = woooo
    imds.wovvo[:] = wOVvo*2 + woVVo
    imds.woVVo[:] = woVVo
    imds.woovo[:] = woovo
    imds.wvvvo[:] = wvvvo
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.v4 = v4
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = time.clock(), time.time()
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

    # m3 = einsum('ijab,acbd->ijcd', l2, vvvv)
    #    = einsum('ijab,cadb->ijcd', l2.conj(), vvvv).conj()
    m3 = mycc._add_vvvv(None, l2.conj(), eris, with_ovvv=False).conj()
    m3 += lib.einsum('klab,ikjl->ijab', l2, imds.woooo)
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
    l1new -= np.einsum('ik,ka->ia', moo, imds.v4)
    l1new -= np.einsum('ca,ic->ia', mvv, imds.v4)
    l2new += np.einsum('ia,jb->ijab', l1, imds.v4)

    tmp  = t1 + np.einsum('kc,kjcb->jb', l1, theta)
    tmp -= lib.einsum('bd,jd->jb', mvv1, t1)
    tmp -= lib.einsum('lj,lb->jb', moo, t1)
    l1new += np.einsum('jbia,jb->ia', ovov, tmp)
    ovov = tmp = None

    ovvv = np.asarray(eris.get_ovvv())
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

    eia = lib.direct_sum('i-j->ij', foo.diagonal(), fvv.diagonal() + mycc.level_shift)
    l1new /= eia
    l1new += l1

    l2new = l2new + l2new.transpose(1,0,3,2)
    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import rccsd

    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-16
    mf.scf()

    mcc = rccsd.RCCSD(mf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()

    eris = mcc.ao2mo()
    l1, l2 = mcc.solve_lambda(t1, t2, eris=eris)
    print(np.linalg.norm(l1)-0.0132626841292)
    print(np.linalg.norm(l2)-0.212575609057)

    from pyscf.cc import ccsd_rdm
    dm1 = ccsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = ccsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    h1 = reduce(np.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    nmo = h1.shape[0]
    eri = ao2mo.full(mf._eri, mf.mo_coeff)
    eri = ao2mo.restore(1, eri, nmo).reshape((nmo,)*4)
    e1 = np.einsum('pq,pq', h1, dm1)
    e2 = np.einsum('pqrs,pqrs', eri, dm2) * .5
    print(e1+e2+mol.energy_nuc() - mf.e_tot - ecc)

