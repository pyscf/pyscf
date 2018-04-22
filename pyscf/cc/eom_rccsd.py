#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf import __config__


def kernel(eom, nroots=1, koopmans=False, guess=None, left=False,
           eris=None, imds=None, **kwargs):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(eom.stdout, eom.verbose)
    if eom.verbose >= logger.WARN:
        eom.check_sanity()
    eom.dump_flags()

    if imds is None:
        imds = eom.make_imds(eris)

    matvec, diag = eom.gen_matvec(imds, left=left, **kwargs)

    size = eom.vector_size()
    nroots = min(nroots, size)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = eom.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    eig = lib.davidson_nosym1
    if user_guess or koopmans:
        assert len(guess) == nroots
        def eig_close_to_init_guess(w, v, nr, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            s = np.dot(s.conj().T, s).diagonal()
            idx = np.argsort(-s)[:nroots]
            idx = idx[np.argsort(w[idx])]  # sort eigenvalue w
            return w[idx].real, v[:,idx].real, idx
        conv, es, vs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                           max_space=eom.max_space, nroots=nroots, verbose=log)
    else:
        conv, es, vs = eig(matvec, guess, precond,
                           tol=eom.conv_tol, max_cycle=eom.max_cycle,
                           max_space=eom.max_space, nroots=nroots, verbose=log)

    if eom.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1, r2 = eom.vector_to_amplitudes(vn)
            if isinstance(r1, np.ndarray):
                qp_weight = np.linalg.norm(r1)**2
            else: # for EOM-UCCSD
                r1 = np.hstack([x.ravel() for x in r1])
                qp_weight = np.linalg.norm(r1)**2
            logger.info(eom, 'EOM-CCSD root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qp_weight, convn)
        log.timer('EOM-CCSD', *cput0)
    if nroots == 1:
        return conv[0], es[0].real, vs[0]
    else:
        return conv, es.real, vs


class EOM(lib.StreamObject):
    def __init__(self, cc):
        self.mol = cc.mol
        self._cc = cc
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.max_memory = cc.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', cc.max_cycle)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', cc.conv_tol)
        self.partition = getattr(__config__, 'eom_rccsd_EOM_partition', None)

##################################################
# don't modify the following attributes, they are not input options
        self.e = None
        self.v = None
        self.nocc = cc.nocc
        self.nmo = cc.nmo
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        logger.info(self, '')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'partition = %s', self.partition)
        #logger.info(self, 'nocc = %d', self.nocc)
        #logger.info(self, 'nmo = %d', self.nmo)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self


########################################
# EOM-IP-CCSD
########################################

def ipccsd(eom, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
    '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        partition : bool or str
            Use a matrix-partitioning for the doubles-doubles block.
            Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
            or 'full' (full diagonal elements).
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    if partition is not None:
        eom.partition = partition.lower()
        assert eom.partition in ['mp','full']
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, left, eris=eris, imds=imds)
    return eom.e, eom.v

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kbij,k->ijb', imds.Wovoo, r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('bd,ijd->ijb', fvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', foo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('bd,ijd->ijb', imds.Lvv, r2)
        Hr2 += -lib.einsum('ki,kjb->ijb', imds.Loo, r2)
        Hr2 += -lib.einsum('lj,ilb->ijb', imds.Loo, r2)
        Hr2 +=  lib.einsum('klij,klb->ijb', imds.Woooo, r2)
        Hr2 += 2*lib.einsum('lbdj,ild->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('kbdj,kid->ijb', imds.Wovvo, r2)
        Hr2 +=  -lib.einsum('lbjd,ild->ijb', imds.Wovov, r2) #typo in Ref
        Hr2 +=  -lib.einsum('kbid,kjd->ijb', imds.Wovov, r2)
        tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2)
        tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2)
        Hr2 += -np.einsum('c,ijcb->ijb', tmp, imds.t2)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def lipccsd_matvec(eom, vector, imds=None, diag=None):
    '''For left eigenvector'''
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,i->k', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += -np.einsum('kbij,ijb->k', imds.Wovoo, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kd,l->kld', imds.Fov, r1)
    Hr2 += 2.*np.einsum('ld,k->kld', imds.Fov, r1)
    Hr2 += -np.einsum('klid,i->kld', 2.*imds.Wooov-imds.Wooov.transpose(1,0,2,3), r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('bd,klb->kld', fvv, r2)
        Hr2 += -lib.einsum('ki,ild->kld', foo, r2)
        Hr2 += -lib.einsum('lj,kjd->kld', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ip(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('bd,klb->kld', imds.Lvv, r2)
        Hr2 += -lib.einsum('ki,ild->kld', imds.Loo, r2)
        Hr2 += -lib.einsum('lj,kjd->kld', imds.Loo, r2)
        Hr2 += lib.einsum('lbdj,kjb->kld', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
        Hr2 += -lib.einsum('kbdj,ljb->kld', imds.Wovvo, r2)
        Hr2 += lib.einsum('klij,ijd->kld', imds.Woooo, r2)
        Hr2 += -lib.einsum('kbid,ilb->kld', imds.Wovov, r2)
        tmp = np.einsum('ijcb,ijb->c', imds.t2, r2)
        Hr2 += -np.einsum('lkdc,c->kld', 2.*imds.Woovv-imds.Woovv.transpose(1,0,2,3), tmp)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), t1.dtype)
    for i in range(nocc):
        for j in range(nocc):
            for b in range(nvir):
                if eom.partition == 'mp':
                    Hr2[i,j,b] += fvv[b,b]
                    Hr2[i,j,b] += -foo[i,i]
                    Hr2[i,j,b] += -foo[j,j]
                else:
                    Hr2[i,j,b] += imds.Lvv[b,b]
                    Hr2[i,j,b] += -imds.Loo[i,i]
                    Hr2[i,j,b] += -imds.Loo[j,j]
                    Hr2[i,j,b] +=  imds.Woooo[i,j,i,j]
                    Hr2[i,j,b] +=2*imds.Wovvo[j,b,b,j]
                    Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                    Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                    Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                    Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:], t2[i,j,:,b])
                    Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:], t2[i,j,:,b])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector

def ipccsd_star(eom, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, eris=None):
    assert(eom.partition == None)
    if eris is None:
        eris = eom._cc.ao2mo()
    t1, t2 = eom._cc.t1, eom._cc.t2
    fock = eris.fock
    nocc, nvir = t1.shape

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    oovv = _cp(eris.ovov).transpose(0,2,1,3)
    ovvv = _cp(eris.get_ovvv()).transpose(0,2,1,3)
    ovov = _cp(eris.oovv).transpose(0,2,1,3)
    ovvo = _cp(eris.ovvo).transpose(0,2,1,3)
    ooov = _cp(eris.ovoo).transpose(2,0,3,1)
    vooo = ooov.conj().transpose(3,2,1,0)
    vvvo = ovvv.transpose(3,2,1,0).conj()
    oooo = _cp(eris.oooo).transpose(0,2,1,3)

    eijkab = np.zeros((nocc,nocc,nocc,nvir,nvir))
    for i,j,k in lib.cartesian_prod([range(nocc),range(nocc),range(nocc)]):
        for a,b in lib.cartesian_prod([range(nvir),range(nvir)]):
            eijkab[i,j,k,a,b] = foo[i,i] + foo[j,j] + foo[k,k] - fvv[a,a] - fvv[b,b]

    ipccsd_evecs  = np.array(ipccsd_evecs)
    lipccsd_evecs = np.array(lipccsd_evecs)
    e = []
    for _eval, _evec, _levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        l1, l2 = eom.vector_to_amplitudes(_levec)
        r1, r2 = eom.vector_to_amplitudes(_evec)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(), r2.ravel())
        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

        _eijkab = eijkab + _eval
        _eijkab = 1./_eijkab

        lijkab = 0.5*np.einsum('ijab,k->ijkab', oovv, l1)
        lijkab += lib.einsum('ieab,jke->ijkab', ovvv, l2)
        lijkab += -lib.einsum('kjmb,ima->ijkab', ooov, l2)
        lijkab += -lib.einsum('ijmb,mka->ijkab', ooov, l2)
        lijkab = lijkab + lijkab.transpose(1,0,2,4,3)

        tmp = np.einsum('mbke,m->bke', ovov, r1)
        rijkab = -lib.einsum('bke,ijae->ijkab', tmp, t2)
        tmp = np.einsum('mbej,m->bej', ovvo, r1)
        rijkab += -lib.einsum('bej,ikae->ijkab', tmp, t2)
        tmp = np.einsum('mnjk,n->mjk', oooo, r1)
        rijkab += lib.einsum('mjk,imab->ijkab', tmp, t2)
        rijkab += lib.einsum('baei,kje->ijkab', vvvo, r2)
        rijkab += -lib.einsum('bmjk,mia->ijkab', vooo, r2)
        rijkab += -lib.einsum('bmji,kma->ijkab', vooo, r2)
        rijkab = rijkab + rijkab.transpose(1,0,2,4,3)

        lijkab = 4.*lijkab \
               - 2.*lijkab.transpose(1,0,2,3,4) \
               - 2.*lijkab.transpose(2,1,0,3,4) \
               - 2.*lijkab.transpose(0,2,1,3,4) \
               + 1.*lijkab.transpose(1,2,0,3,4) \
               + 1.*lijkab.transpose(2,0,1,3,4)

        deltaE = 0.5*np.einsum('ijkab,ijkab,ijkab', lijkab, rijkab, _eijkab)
        deltaE = deltaE.real
        logger.info(eom, "Exc. energy, delta energy = %16.12f, %16.12f",
                    _eval+deltaE, deltaE)
        e.append(_eval+deltaE)
    return e

class EOMIP(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    kernel = ipccsd
    ipccsd = ipccsd
    matvec = ipccsd_matvec
    l_matvec = lipccsd_matvec
    get_diag = ipccsd_diag
    ipccsd_star = ipccsd_star

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ip(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*nocc*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip(self.partition)
        return imds

    @property
    def eip(self):
        return self.e


########################################
# EOM-EA-CCSD
########################################

def eaccsd(eom, nroots=1, left=False, koopmans=False, guess=None,
           partition=None, eris=None, imds=None):
    '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

    Kwargs:
        See also ipccd()
    '''
    return ipccsd(eom, nroots, left, koopmans, guess, partition, eris, imds)

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nvir].copy()
    r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    vector = np.hstack((r1, r2.ravel()))
    return vector

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (30)
    # 1p-1p block
    Hr1 =  np.einsum('ac,c->a', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('ld,lad->a', 2.*imds.Fov, r2)
    Hr1 += np.einsum('ld,lda->a',   -imds.Fov, r2)
    Hr1 += np.einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 +=  lib.einsum('ac,jcb->jab', fvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', fvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ea(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 +=  lib.einsum('ac,jcb->jab', imds.Lvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', imds.Lvv, r2)
        Hr2 += -lib.einsum('lj,lab->jab', imds.Loo, r2)
        Hr2 += lib.einsum('lbdj,lad->jab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
        Hr2 += -lib.einsum('lajc,lcb->jab', imds.Wovov, r2)
        Hr2 += -lib.einsum('lbcj,lca->jab', imds.Wovvo, r2)
        for a in range(nvir):
            Hr2[:,a,:] += lib.einsum('bcd,jcd->jb', imds.Wvvvv[a], r2)
        tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
        Hr2 += -np.einsum('k,kjab->jab', tmp, imds.t2)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def leaccsd_matvec(eom, vector, imds=None, diag=None):
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    # Eq. (30)
    # 1p-1p block
    Hr1 = np.einsum('ac,a->c', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('abcj,jab->c', imds.Wvvvo, r2)
    # Eq. (31)
    # 2p1h-1p block
    Hr2 = 2.*np.einsum('c,ld->lcd', r1, imds.Fov)
    Hr2 +=  -np.einsum('d,lc->lcd', r1, imds.Fov)
    Hr2 += np.einsum('a,alcd->lcd', r1, 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2))
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += lib.einsum('lad,ac->lcd', r2, fvv)
        Hr2 += lib.einsum('lcb,bd->lcd', r2, fvv)
        Hr2 += -lib.einsum('jcd,lj->lcd', r2, foo)
    elif eom.partition == 'full':
        diag_matrix2 = vector_to_amplitudes_ea(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += lib.einsum('lad,ac->lcd', r2, imds.Lvv)
        Hr2 += lib.einsum('lcb,bd->lcd', r2, imds.Lvv)
        Hr2 += -lib.einsum('jcd,lj->lcd', r2, imds.Loo)
        Hr2 += lib.einsum('jcb,lbdj->lcd', r2, 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2))
        Hr2 += -lib.einsum('lajc,jab->lcb', imds.Wovov, r2)
        Hr2 += -lib.einsum('lbcj,jab->lca', imds.Wovvo, r2)
        for a in range(nvir):
            Hr2 += lib.einsum('lb,bcd->lcd', r2[:,a,:], imds.Wvvvv[a])
        tmp = np.einsum('ijcb,ibc->j', imds.t2, r2)
        Hr2 += -np.einsum('kjfe,j->kef', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2),tmp)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = np.diag(imds.Lvv)
    Hr2 = np.zeros((nocc,nvir,nvir), t1.dtype)
    for a in range(nvir):
        if eom.partition != 'mp':
            _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(nvir):
            for j in range(nocc):
                if eom.partition == 'mp':
                    Hr2[j,a,b] += fvv[a,a]
                    Hr2[j,a,b] += fvv[b,b]
                    Hr2[j,a,b] += -foo[j,j]
                else:
                    Hr2[j,a,b] += imds.Lvv[a,a]
                    Hr2[j,a,b] += imds.Lvv[b,b]
                    Hr2[j,a,b] += -imds.Loo[j,j]
                    Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                    Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                    Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                    Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                    Hr2[j,a,b] += _Wvvvva[b,a,b]
                    Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b])
                    Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b])

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector

def eaccsd_star(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, eris=None):
    assert(eom.partition == None)
    if eris is None:
        eris = eom._cc.ao2mo()
    t1, t2 = eom._cc.t1, eom._cc.t2
    fock = eris.fock
    nocc, nvir = t1.shape

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    oovv = _cp(eris.ovov).transpose(0,2,1,3)
    ovvv = _cp(eris.get_ovvv()).transpose(0,2,1,3)
    vvov = ovvv.transpose(2,3,0,1).conj()
    ooov = _cp(eris.ovoo).transpose(2,0,3,1)
    vooo = ooov.conj().transpose(3,2,1,0)
    ovov = _cp(eris.oovv).transpose(0,2,1,3)
    vvvv = imd._get_vvvv(eris).transpose(0,2,1,3)
    ovvo = _cp(eris.ovvo).transpose(0,2,1,3)

    eijabc = np.zeros((nocc,nocc,nvir,nvir,nvir))
    for i,j in lib.cartesian_prod([range(nocc),range(nocc)]):
        for a,b,c in lib.cartesian_prod([range(nvir),range(nvir),range(nvir)]):
            eijabc[i,j,a,b,c] = foo[i,i] + foo[j,j] - fvv[a,a] - fvv[b,b] - fvv[c,c]

    eaccsd_evecs  = np.array(eaccsd_evecs)
    leaccsd_evecs = np.array(leaccsd_evecs)
    e = []
    for _eval, _evec, _levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        l1, l2 = eom.vector_to_amplitudes(_levec)
        r1, r2 = eom.vector_to_amplitudes(_evec)
        ldotr = np.dot(l1, r1) + np.dot(l2.ravel(),r2.ravel())
        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
        r2 = r2.transpose(0,2,1)

        _eijabc = eijabc + _eval
        _eijabc = 1./_eijabc

        lijabc = -0.5*np.einsum('ijab,c->ijabc', oovv, l1)
        lijabc += lib.einsum('jima,mbc->ijabc', ooov, l2)
        lijabc -= lib.einsum('ieab,jec->ijabc', ovvv, l2)
        lijabc -= lib.einsum('jebc,iae->ijabc', ovvv, l2)
        lijabc = lijabc + lijabc.transpose(1,0,3,2,4)

        tmp = np.einsum('bcef,f->bce', vvvv, r1)
        rijabc = -lib.einsum('bce,ijae->ijabc', tmp, t2)
        tmp = np.einsum('mcje,e->mcj', ovov, r1)
        rijabc += lib.einsum('mcj,imab->ijabc', tmp, t2)
        tmp = np.einsum('mbej,e->mbj', ovvo, r1)
        rijabc += lib.einsum('mbj,imac->ijabc', tmp, t2)
        rijabc += lib.einsum('amij,mbc->ijabc', vooo, r2)
        rijabc += -lib.einsum('bcje,iae->ijabc', vvov, r2)
        rijabc += -lib.einsum('abie,jec->ijabc', vvov, r2)
        rijabc = rijabc + rijabc.transpose(1,0,3,2,4)

        lijabc =  4.*lijabc \
                - 2.*lijabc.transpose(0,1,3,2,4) \
                - 2.*lijabc.transpose(0,1,4,3,2) \
                - 2.*lijabc.transpose(0,1,2,4,3) \
                + 1.*lijabc.transpose(0,1,3,4,2) \
                + 1.*lijabc.transpose(0,1,4,2,3)
        deltaE = 0.5*np.einsum('ijabc,ijabc,ijabc', lijabc,rijabc,_eijabc)
        deltaE = deltaE.real
        logger.info(eom, "Exc. energy, delta energy = %16.12f, %16.12f",
                    _eval+deltaE, deltaE)
        e.append(_eval+deltaE)
    return e


class EOMEA(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(size, dtype)
                g[n] = 1.0
                guess.append(g)
        else:
            idx = diag.argsort()[:nroots]
            for i in idx:
                g = np.zeros(size, dtype)
                g[i] = 1.0
                guess.append(g)
        return guess

    kernel = eaccsd
    eaccsd = eaccsd
    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    get_diag = eaccsd_diag
    eaccsd_star = eaccsd_star

    def gen_matvec(self, imds=None, left=False, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        if left:
            matvec = lambda xs: [self.l_matvec(x, imds, diag) for x in xs]
        else:
            matvec = lambda xs: [self.matvec(x, imds, diag) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_ea(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nvir + nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea(self.partition)
        return imds

    @property
    def eea(self):
        return self.e


########################################
# EOM-EE-CCSD
########################################

#TODO: double spin-flip EOM-EE

def eeccsd(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    '''Calculate N-electron neutral excitations via EOM-EE-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (1p1h) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    if eris is None: eris = eom._cc.ao2mo()
    if imds is None: imds = eom.make_imds(eris)

    spinvec_size = eom.vector_size()
    nroots = min(nroots, spinvec_size)

    diag_eeS, diag_eeT, diag_sf = eom.get_diag(imds)
    guess_eeS = []
    guess_eeT = []
    guess_sf = []
    if guess:
        for g in guess:
            if g is None: # beta->alpha spin-flip excitation
                pass
            elif g.size == diag_eeS.size:
                guess_eeS.append(g)
            elif g.size == diag_eeT.size:
                guess_eeT.append(g)
            else:
                guess_sf.append(g)
        nroots_eeS = len(guess_eeS)
        nroots_eeT = len(guess_eeT)
        nroots_sf = len(guess_sf)
        if len(guess) != nroots:
            logger.warn(eom, 'Number of states in initial guess %d does not '
                        'equal to nroots %d.', len(guess), nroots)
    else:
        deeS = np.sort(diag_eeS)[:nroots]
        deeT = np.sort(diag_eeT)[:nroots]
        dsf = np.sort(diag_sf)[:nroots]
        dmax = np.sort(np.hstack([deeS,deeT,dsf,dsf]))[nroots-1]
        nroots_eeS = np.count_nonzero(deeS <= dmax)
        nroots_eeT = np.count_nonzero(deeT <= dmax)
        nroots_sf = np.count_nonzero(dsf <= dmax)
        guess_eeS = guess_eeT = guess_sf = None

    def eomee_sub(cls, nroots, guess, diag):
        ee_sub = cls(eom._cc)
        ee_sub.__dict__.update(eom.__dict__)
        e, v = ee_sub.kernel(nroots, koopmans, guess, eris, imds, diag=diag)
        if nroots == 1:
            e, v = [e], [v]
            ee_sub.converged = [ee_sub.converged]
        return list(ee_sub.converged), list(e), list(v)

    e0 = e1 = e2 = []
    v0 = v1 = v2 = []
    conv0 = conv1 = conv2 = []
    if nroots_eeS > 0:
        conv0, e0, v0 = eomee_sub(EOMEESinglet, nroots_eeS, guess_eeS, diag_eeS)
    if nroots_eeT > 0:
        conv2, e2, v2 = eomee_sub(EOMEETriplet, nroots_eeT, guess_eeT, diag_eeT)
    if nroots_sf > 0:
        conv1, e1, v1 = eomee_sub(EOMEESpinFlip, nroots_sf, guess_sf, diag_sf)
        # The associated solutions of beta->alpha excitations
        e1 = e1 + e1
        conv1 = conv1 + conv1
        v1 = v1 + [None] * len(v1)
# beta->alpha spin-flip excitations, the coefficients are (-r1, (-r2[0], r2[1]))
# as below.  The EOMEESpinFlip class only handles alpha->beta excitations.
# Setting beta->alpha to None to bypass the vectors in initial guess
        #for i in range(nroots_sf):
        #    r1, r2 = vector_to_amplitudes_eomsf(v1[i], eom.nmo, eom.nocc)
        #    v1.append(amplitudes_to_vector_eomsf(-r1, (-r2[0], r2[1])))

    e = np.hstack([e0,e2,e1])
    idx = e.argsort()
    e = e[idx]
    conv = conv0 + conv2 + conv1
    conv = [conv[x] for x in idx]
    v = v0 + v2 + v1
    v = [v[x] for x in idx]

    if nroots == 1:
        conv = conv[0]
        e = e[0]
        v = v[0]
    eom.converged = conv
    eom.e = e
    eom.v = v
    return eom.e, eom.v


def eomee_ccsd_singlet(eom, nroots=1, koopmans=False, guess=None,
                       eris=None, imds=None, diag=None):
    '''EOM-EE-CCSD singlet
    '''
    eom.converged, eom.e, eom.v \
            = kernel(eom, nroots, koopmans, guess, eris=eris, imds=imds, diag=diag)
    return eom.e, eom.v

def eomee_ccsd_triplet(eom, nroots=1, koopmans=False, guess=None,
                       eris=None, imds=None, diag=None):
    '''EOM-EE-CCSD triplet
    '''
    return eomee_ccsd_singlet(eom, nroots, koopmans, guess, eris, imds, diag)

def eomsf_ccsd(eom, nroots=1, koopmans=False, guess=None,
               eris=None, imds=None, diag=None):
    '''Spin flip EOM-EE-CCSD
    '''
    return eomee_ccsd_singlet(eom, nroots, koopmans, guess, eris, imds, diag)

vector_to_amplitudes_ee = vector_to_amplitudes_singlet = ccsd.vector_to_amplitudes
amplitudes_to_vector_ee = amplitudes_to_vector_singlet = ccsd.amplitudes_to_vector

def amplitudes_to_vector_eomsf(t1, t2, out=None):
    nocc, nvir = t1.shape
    t2baaa, t2aaba = t2
    otril = np.tril_indices(nocc, k=-1)
    vtril = np.tril_indices(nvir, k=-1)
    baaa = np.take(t2baaa.reshape(nocc*nocc,nvir*nvir),
                   vtril[0]*nvir+vtril[1], axis=1)
    vector = np.hstack((t1.ravel(), baaa.ravel(), t2aaba[otril].ravel()))
    return vector

def vector_to_amplitudes_eomsf(vector, nmo, nocc):
    nvir = nmo - nocc
    t1 = vector[:nocc*nvir].reshape(nocc,nvir).copy()
    pvec = vector[t1.size:]

    nbaaa = nocc*nocc*nvir*(nvir-1)//2
    naaba = nocc*(nocc-1)//2*nvir*nvir
    t2baaa = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
    t2aaba = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
    otril = np.tril_indices(nocc, k=-1)
    vtril = np.tril_indices(nvir, k=-1)

    v = pvec[:nbaaa].reshape(nocc*nocc,nvir*(nvir-1)//2)
    t2baaa[:,vtril[0]*nvir+vtril[1]] = v
    t2baaa[:,vtril[1]*nvir+vtril[0]] = -v

    v = pvec[nbaaa:nbaaa+naaba].reshape(-1,nvir*nvir)
    t2aaba[otril[0]*nocc+otril[1]] = v
    t2aaba[otril[1]*nocc+otril[0]] = -v

    t2baaa = t2baaa.reshape(nocc,nocc,nvir,nvir)
    t2aaba = t2aaba.reshape(nocc,nocc,nvir,nvir)
    return t1, (t2baaa, t2aaba)

def amplitudes_to_vector_triplet(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    size = size1 + nov*(nov+1)//2
    vector = np.ndarray(size, t1.dtype, buffer=out)
    ccsd.amplitudes_to_vector_s4(t1, t2[0], out=vector)
    t2ab = t2[1].transpose(0,2,1,3).reshape(nov,nov)
    lib.pack_tril(t2ab, out=vector[size1:])
    return vector

def vector_to_amplitudes_triplet(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    size = size1 + nov*(nov+1)//2
    t1, t2aa = ccsd.vector_to_amplitudes_s4(vector[:size1], nmo, nocc)
    t2ab = lib.unpack_tril(vector[size1:size], filltriu=2)
    t2ab = t2ab.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3).copy()
    return t1, (t2aa, t2ab)

def eeccsd_matvec_singlet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = vector_to_amplitudes_singlet(vector, nmo, nocc)
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    rho = r2*2 - r2.transpose(0,1,3,2)
    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, rho)

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2 += lib.einsum('ijef,aebf->ijab', tau2, eris_vvvv) * .5
    tau2 = _make_tau(r2, r1, t1, fac=2)
    Hr2 = eom._cc._add_vvvv(None, tau2, eris, with_ovvv=False, t2sym='jiba')
    Hr2 *= .5

    Hr2 += lib.einsum('mnij,mnab->ijab', imds.woOoO, r2) * .5
    Hr2 += lib.einsum('be,ijae->ijab', imds.Fvv   , r2)
    Hr2 -= lib.einsum('mj,imab->ijab', imds.Foo   , r2)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, rho[:,p0:p1])
        tmp = lib.einsum('meaf,ijef->maij', ovvv, tau2)
        Hr2 -= lib.einsum('ma,mbij->ijab', t1[p0:p1], tmp)
        tmp  = lib.einsum('meaf,me->af', ovvv, r1[p0:p1]) * 2
        tmp -= lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2 += lib.einsum('af,ijfb->ijab', tmp, t2)
        ovvv = tmp = None
    Hr2 -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)

    Hr1-= lib.einsum('mnie,mnae->ia', imds.woOoV, rho)
    tmp = lib.einsum('nmie,me->ni', imds.woOoV, r1) * 2
    tmp-= lib.einsum('mnie,me->ni', imds.woOoV, r1)
    Hr2 -= lib.einsum('ni,njab->ijab', tmp, t2)
    tmp = None
    for p0, p1 in lib.prange(0, nvir, nocc):
        Hr2 += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    oVVo = np.asarray(imds.woVVo)
    tmp = lib.einsum('mbej,imea->jiab', oVVo, r2)
    Hr2 += tmp
    Hr2 += tmp.transpose(0,1,3,2) * .5
    oVvO = np.asarray(imds.woVvO) + oVVo * .5
    oVVo = tmp = None
    Hr1 += np.einsum('maei,me->ia', oVvO, r1) * 2
    Hr2 += lib.einsum('mbej,imae->ijab', oVvO, rho)
    oVvO = None

    eris_ovov = np.asarray(eris.ovov)
    tau2 = _make_tau(r2, r1, t1, fac=2)
    tmp = lib.einsum('menf,ijef->mnij', eris_ovov, tau2)
    tau2 = None
    tau = _make_tau(t2, t1, t1)
    Hr2 += lib.einsum('mnij,mnab->ijab', tmp, tau) * .5
    tau = tmp = None

    tmp = lib.einsum('nemf,imef->ni', eris_ovov, rho)
    Hr1 -= lib.einsum('na,ni->ia', t1, tmp)
    Hr2 -= lib.einsum('mj,miab->ijba', tmp, t2)
    tmp = None

    tmp  = np.einsum('mfne,mf->en', eris_ovov, r1) * 2
    tmp -= np.einsum('menf,mf->en', eris_ovov, r1)
    tmp  = np.einsum('en,nb->eb', tmp, t1)
    tmp += lib.einsum('menf,mnbf->eb', eris_ovov, rho)
    Hr2 -= lib.einsum('eb,ijea->jiab', tmp, t2)
    tmp = eris_ovov = rho = None

    Hr2 = Hr2 + Hr2.transpose(1,0,3,2)
    vector = amplitudes_to_vector_ee(Hr1, Hr2)
    return vector

def eeccsd_matvec_triplet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = vector_to_amplitudes_triplet(vector, nmo, nocc)
    r2aa, r2ab = r2
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    theta = r2aa + r2ab

    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, r2aa)
    Hr1 += np.einsum('ME,iMaE->ia',imds.Fov, r2ab)

    Hr2aa = lib.einsum('mnij,mnab->ijab', imds.woOoO, r2aa) * .25
    Hr2ab = lib.einsum('mNiJ,mNaB->iJaB', imds.woOoO, r2ab) * .5
    Hr2aa+= lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aa)
    Hr2aa-= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aa)
    Hr2ab+= lib.einsum('BE,iJaE->iJaB', imds.Fvv, r2ab)
    Hr2ab-= lib.einsum('MJ,iMaB->iJaB', imds.Foo, r2ab)

    tau2ab = np.einsum('ia,jb->ijab', r1, t1)
    tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
    tau2ab+= r2ab
    tau2aa = np.einsum('ia,jb->ijab', r1, t1)
    tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
    tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
    tau2aa+= r2aa

    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, theta[:,p0:p1])
        tmpaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aa)
        tmpab = lib.einsum('meAF,iJeF->mAiJ', ovvv, tau2ab)
        tmp1 += lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2aa+= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmpaa)
        Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1[p0:p1], tmpab)
        ovvv = tmpaa = tmpab = None
    tmpa = lib.einsum('mnie,me->ni', imds.woOoV, r1)
    tmp  = lib.einsum('ni,njab->ijab', tmpa, t2)
    tmp -= lib.einsum('af,ijfb->ijab', tmp1, t2)
    tmp -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)
    for p0,p1 in lib.prange(0, nvir, nocc):
        tmp += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    oVVo = np.asarray(imds.woVVo)
    Hr1 += np.einsum('maei,me->ia',imds.woVVo, r1)
    Hr2aa+= lib.einsum('mbej,imae->ijba', oVVo, r2ab)
    Hr2ab+= lib.einsum('MBEJ,iMEa->iJaB', oVVo, r2aa)
    Hr2ab+= lib.einsum('MbeJ,iMeA->iJbA', oVVo, r2ab)
    oVVo += np.asarray(imds.woVvO)
    tmp += lib.einsum('mbej,imae->ijab', oVVo, theta)
    oVVo = None
    Hr1-= lib.einsum('mnie,mnae->ia', imds.woOoV, theta)
    Hr2aa+= tmp
    Hr2ab+= tmp
    tmp = None

    eris_ovov = np.asarray(eris.ovov)
    tau = _make_tau(t2, t1, t1)
    tmpaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aa)
    tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovov, tau2ab)
    Hr2aa += lib.einsum('mnij,mnab->ijab', tmpaa, tau) * 0.25
    Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tau) * .5
    tmpaa = tmpab = tau = None

    tmpa = -lib.einsum('menf,imfe->ni', eris_ovov, theta)
    Hr1 += lib.einsum('na,ni->ia', t1, tmpa)
    tmp  = lib.einsum('mj,imab->ijab', tmpa, t2)
    tmp1 = np.einsum('menf,mf->en', eris_ovov, r1)
    tmpa = np.einsum('en,nb->eb', tmp1, t1)
    tmpa-= lib.einsum('menf,mnbf->eb', eris_ovov, theta)
    tmp += lib.einsum('eb,ijae->ijab', tmpa, t2)
    Hr2aa+= tmp
    Hr2ab-= tmp
    tmp = theta = eris_ovov = None

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .25
    #:Hr2ab += lib.einsum('ijef,aebf->ijab', tau2ab, eris_vvvv) * .5
    tau2aa *= .25
    tau2ab *= .5
    Hr2aa += eom._cc._add_vvvv(None, tau2aa, eris, with_ovvv=False, t2sym='jiba')
    Hr2ab += eom._cc._add_vvvv(None, tau2ab, eris, with_ovvv=False, t2sym='-jiba')
    tau2aa = tau2ab = None

    Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
    Hr2ab = Hr2ab - Hr2ab.transpose(1,0,3,2)
    vector = amplitudes_to_vector_triplet(Hr1, (Hr2aa,Hr2ab))
    return vector

def eeccsd_matvec_sf(eom, vector, imds=None):
    '''Spin flip EOM-CCSD'''
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    t1, t2, eris = imds.t1, imds.t2, imds.eris
    r1, r2 = vector_to_amplitudes_eomsf(vector, nmo, nocc)
    r2baaa, r2aaba = r2
    nocc, nvir = t1.shape

    Hr1  = np.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= np.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia', imds.Fov, r2baaa)
    Hr1 += np.einsum('me,imae->ia', imds.Fov, r2aaba)
    Hr2baaa = .5*lib.einsum('mnij,mnab->ijab', imds.woOoO, r2baaa)
    Hr2aaba = .5*lib.einsum('mnij,mnab->ijab', imds.woOoO, r2aaba)
    Hr2baaa -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2baaa)
    Hr2aaba -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aaba)
    Hr2baaa -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2baaa)
    Hr2aaba -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2aaba)
    Hr2baaa += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2baaa)
    Hr2aaba += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aaba)
    Hr2baaa += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2baaa)
    Hr2aaba += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2aaba)

    tau2baaa = np.einsum('ia,jb->ijab', r1, t1)
    tau2baaa += r2baaa * .5
    tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
    tau2aaba = np.einsum('ia,jb->ijab', r1, t1)
    tau2aaba += r2aaba * .5
    tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)

    tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, r2baaa[:,p0:p1])
        Hr1 += lib.einsum('mfae,imef->ia', ovvv, r2aaba[:,p0:p1])
        tmp1aaba = lib.einsum('meaf,ijef->maij', ovvv, tau2baaa)
        tmp1baaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aaba)
        tmp1abaa = lib.einsum('meaf,ijfe->maij', ovvv, tau2aaba)
        tmp2aaba = lib.einsum('meaf,ijfe->maij', ovvv, tau2baaa)
        Hr2baaa -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp2aaba)
        Hr2aaba -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp1abaa)
        Hr2baaa -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1aaba)
        Hr2aaba -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1baaa)
        tmp = lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
        tmp = lib.einsum('af,jibf->ijab', tmp, t2)
        Hr2baaa -= tmp
        Hr2aaba -= tmp
        tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = ovvv = None
    tmp = lib.einsum('mbij,ma->ijab', imds.woVoO, r1)
    Hr2baaa -= tmp
    Hr2aaba -= tmp
    tmp = None

    Hr1 -= lib.einsum('mnie,mnae->ia', imds.woOoV, r2aaba)
    Hr1 -= lib.einsum('mnie,mnae->ia', imds.woOoV, r2baaa)
    tmp = lib.einsum('mnie,me->ni', imds.woOoV, r1)
    tmp = lib.einsum('ni,njab->ijab', tmp, t2)
    Hr2baaa += tmp
    Hr2aaba += tmp
    for p0,p1 in lib.prange(0, nvir, nocc):
        tmp = lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])
        Hr2baaa += tmp
        Hr2aaba += tmp
    tmp = None

    oVVo = np.asarray(imds.woVVo)
    Hr1 += np.einsum('maei,me->ia', oVVo, r1)
    Hr2baaa += lib.einsum('mbej,miea->jiba', oVVo, r2baaa)
    Hr2aaba += lib.einsum('mbej,miea->jiba', oVVo, r2aaba)
    oVvO = np.asarray(imds.woVvO)
    Hr2baaa += lib.einsum('mbej,imae->ijab', oVvO, r2aaba)
    Hr2aaba += lib.einsum('mbej,imae->ijab', oVvO, r2baaa)
    oVvO += oVVo
    Hr2baaa += lib.einsum('mbej,imae->ijab', oVvO, r2baaa)
    Hr2aaba += lib.einsum('mbej,imae->ijab', oVvO, r2aaba)
    oVvO = oVVo = None

    eris_ovov = np.asarray(eris.ovov)
    tau = _make_tau(t2, t1, t1)
    tmp1baaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aaba)
    tmp1aaba = lib.einsum('menf,ijef->mnij', eris_ovov, tau2baaa)
    Hr2baaa += .5*lib.einsum('mnij,mnab->ijab', tmp1aaba, tau)
    Hr2aaba += .5*lib.einsum('mnij,mnab->ijab', tmp1baaa, tau)
    tau2 = tmp1baaa = tmp1aaba = None

    rhoaaba = r2aaba + r2baaa
    tmp = lib.einsum('nfme,imfe->ni', eris_ovov, rhoaaba)
    Hr1 -= np.einsum('na,ni->ia', t1, tmp)
    Hr2baaa -= lib.einsum('mj,imba->jiab', tmp, t2)
    Hr2aaba -= lib.einsum('mj,imba->jiab', tmp, t2)

    tmp = np.einsum('menf,mf->en', eris_ovov, r1)
    tmp = np.einsum('en,nb->eb', tmp, t1)
    tmp-= lib.einsum('menf,mnbf->eb', eris_ovov, rhoaaba)
    Hr2baaa += lib.einsum('ea,ijbe->jiab', tmp, t2)
    Hr2aaba += lib.einsum('ea,ijbe->jiab', tmp, t2)
    eris_ovov = rhoaaba = tmp = None

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2baaa += .5*lib.einsum('ijef,aebf->ijab', tau2baaa, eris_vvvv)
    #:Hr2aaba += .5*lib.einsum('ijef,aebf->ijab', tau2aaba, eris_vvvv)
    tau2aaba *= .5
    Hr2aaba += eom._cc._add_vvvv(None, tau2aaba, eris, with_ovvv=False, t2sym='-jiab')
    tau2baaa *= .5
    Hr2baaa += eom._cc._add_vvvv(None, tau2baaa, eris, with_ovvv=False, t2sym=False)
    tau2aaba = tau2baaa = None

    Hr2baaa = Hr2baaa - Hr2baaa.transpose(0,1,3,2)
    Hr2aaba = Hr2aaba - Hr2aaba.transpose(1,0,2,3)
    vector = amplitudes_to_vector_eomsf(Hr1, (Hr2baaa,Hr2aaba))
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    eris = imds.eris
    t1, t2 = imds.t1, imds.t2
    tau = _make_tau(t2, t1, t1)
    nocc, nvir = t1.shape

    Fo = imds.Foo.diagonal()
    Fv = imds.Fvv.diagonal()
    Wovab = np.einsum('iaai->ia', imds.woVVo)
    Wovaa = Wovab + np.einsum('iaai->ia', imds.woVvO)

    eia = lib.direct_sum('-i+a->ia', Fo, Fv)
    Hr1aa = eia + Wovaa
    Hr1ab = eia + Wovab

    eris_ovov = np.asarray(eris.ovov)
    Wvvab = np.einsum('mnab,manb->ab', tau, eris_ovov)
    Wvvaa = .5*Wvvab - .5*np.einsum('mnba,manb->ab', tau, eris_ovov)
    ijb = np.einsum('iejb,ijeb->ijb', eris_ovov, t2)
    Hr2ab = lib.direct_sum('iJB+a->iJaB',-ijb, Fv)
    jab = np.einsum('kajb,kjab->jab', eris_ovov, t2)
    Hr2ab+= lib.direct_sum('-i-jab->ijab', Fo, jab)

    jib = np.einsum('iejb,ijbe->jib', eris_ovov, t2)
    jib = jib + jib.transpose(1,0,2)
    jib-= ijb + ijb.transpose(1,0,2)
    jba = np.einsum('kajb,jkab->jba', eris_ovov, t2)
    jba = jba + jba.transpose(0,2,1)
    jba-= jab + jab.transpose(0,2,1)
    Hr2aa = lib.direct_sum('jib+a->jiba', jib, Fv)
    Hr2aa+= lib.direct_sum('-i+jba->ijba', Fo, jba)
    eris_ovov = None

    Hr2baaa = lib.direct_sum('ijb+a->ijba',-ijb, Fv)
    Hr2baaa += Wovaa.reshape(1,nocc,1,nvir)
    Hr2baaa += Wovab.reshape(nocc,1,1,nvir)
    Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
    Hr2baaa+= lib.direct_sum('-i+jab->ijab', Fo, jba)
    Hr2baaa-= Fo.reshape(1,-1,1,1)
    Hr2aaba = lib.direct_sum('-i-jab->ijab', Fo, jab)
    Hr2aaba += Wovaa.reshape(1,nocc,1,nvir)
    Hr2aaba += Wovab.reshape(1,nocc,nvir,1)
    Hr2aaba = Hr2aaba + Hr2aaba.transpose(1,0,2,3)
    Hr2aaba+= lib.direct_sum('ijb+a->ijab', jib, Fv)
    Hr2aaba+= Fv.reshape(1,1,1,-1)
    Hr2ab += Wovaa.reshape(1,nocc,1,nvir)
    Hr2ab += Wovab.reshape(nocc,1,1,nvir)
    Hr2ab = Hr2ab + Hr2ab.transpose(1,0,3,2)
    Hr2aa += Wovaa.reshape(1,nocc,1,nvir) * 2
    Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
    Hr2aa *= .5

    Wooab = np.einsum('ijij->ij', imds.woOoO)
    Wooaa = Wooab - np.einsum('ijji->ij', imds.woOoO)
    Hr2aa += Wooaa.reshape(nocc,nocc,1,1)
    Hr2ab += Wooab.reshape(nocc,nocc,1,1)
    Hr2baaa += Wooab.reshape(nocc,nocc,1,1)
    Hr2aaba += Wooaa.reshape(nocc,nocc,1,1)

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
    #:tmp = np.einsum('mb,mbaa->ab', t1, eris_ovvv)
    #:Wvvaa += np.einsum('mb,maab->ab', t1, eris_ovvv)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp = np.zeros((nvir,nvir), dtype=t1.dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        tmp += np.einsum('mb,mbaa->ab', t1[p0:p1], ovvv)
        Wvvaa += np.einsum('mb,maab->ab', t1[p0:p1], ovvv)
        ovvv = None
    Wvvaa -= tmp
    Wvvab -= tmp
    Wvvab -= tmp.T
    Wvvaa = Wvvaa + Wvvaa.T
    if eris.vvvv is None: # AO-direct CCSD, vvvv is not generated.
        pass
    elif len(eris.vvvv.shape) == 4:  # DO not use .ndim here for h5py library
                                     # backward compatbility
        eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
        tmp = np.einsum('aabb->ab', eris_vvvv)
        Wvvaa += tmp
        Wvvaa -= np.einsum('abba->ab', eris_vvvv)
        Wvvab += tmp
    else:
        for i in range(nvir):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            Wvvab[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = None

    Hr2aa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2ab += Wvvab.reshape(1,1,nvir,nvir)
    Hr2baaa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2aaba += Wvvab.reshape(1,1,nvir,nvir)

    vec_eeS = amplitudes_to_vector_singlet(Hr1aa, Hr2ab)
    vec_eeT = amplitudes_to_vector_triplet(Hr1aa, (Hr2aa,Hr2ab))
    vec_sf = amplitudes_to_vector_eomsf(Hr1ab, (Hr2baaa,Hr2aaba))
    return vec_eeS, vec_eeT, vec_sf


class EOMEE(EOM):
    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        if koopmans:
            nocc = self.nocc
            nvir = self.nmo - nocc
            idx = diag[:nocc*nvir].argsort()
        else:
            idx = diag.argsort()

        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    kernel = eeccsd
    eeccsd = eeccsd
    get_diag = eeccsd_diag

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc*nvir + nocc*nocc*nvir*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds

    @property
    def eee(self):
        return self.e


class EOMEESinglet(EOMEE):
    kernel = eomee_ccsd_singlet
    eomee_ccsd_singlet = eomee_ccsd_singlet
    matvec = eeccsd_matvec_singlet

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[0]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_singlet(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_singlet(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        return nov + nov*(nov+1)//2


class EOMEETriplet(EOMEE):
    kernel = eomee_ccsd_triplet
    eomee_ccsd_triplet = eomee_ccsd_triplet
    matvec = eeccsd_matvec_triplet

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[1]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_triplet(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_triplet(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nov = nocc * nvir
        return nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2 + nov*(nov+1)//2


class EOMEESpinFlip(EOMEE):
    kernel = eomsf_ccsd
    eomsf_ccsd = eomsf_ccsd
    matvec = eeccsd_matvec_sf

    def gen_matvec(self, imds=None, diag=None, **kwargs):
        if imds is None: imds = self.make_imds()
        if diag is None: diag = self.get_diag(imds)[2]
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes_eomsf(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1, r2):
        return amplitudes_to_vector_eomsf(r1, r2)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nbaaa = nocc*nocc*nvir*(nvir-1)//2
        naaba = nocc*(nocc-1)//2*nvir*nvir
        return nocc*nvir + nbaaa + naaba


class _IMDS:
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1, t2, eris)
        self.Lvv = imd.Lvv(t1, t2, eris)
        self.Fov = imd.cc_Fov(t1, t2, eris)

        logger.timer_debug1(self, 'EOM-CCSD shared one-electron '
                            'intermediates', *cput0)
        return self

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris)
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        self._made_shared_2e = True
        log.timer_debug1('EOM-CCSD shared two-electron intermediates', *cput0)
        return self

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ip_partition != 'mp':
            self._make_shared_2e()

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)
        log.timer_debug1('EOM-CCSD IP intermediates', *cput0)
        return self

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ea_partition != 'mp':
            self._make_shared_2e()

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)
        log.timer_debug1('EOM-CCSD EA intermediates', *cput0)
        return self


    def make_ee(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        if np.iscomplexobj(t2):
            raise NotImplementedError('Complex integrals are not supported in EOM-EE-CCSD')

        nocc, nvir = t1.shape
        nvir_pair = nvir*(nvir+1)//2

        self.saved = lib.H5TmpFile()
        self.wvOvV = self.saved.create_dataset('vOvV', (nvir,nocc,nvir,nvir), t1.dtype.char)

        foo = eris.fock[:nocc,:nocc]
        fov = eris.fock[:nocc,nocc:]
        fvv = eris.fock[nocc:,nocc:]

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:self.Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv) * 2
        #:self.Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        #:self.woVvO = lib.einsum('jf,mebf->mbej', t1, eris_ovvv)
        #:self.woVVo = lib.einsum('jf,mfbe->mbej',-t1, eris_ovvv)
        #:tau = _make_tau(t2, t1, t1)
        #:self.woVoO  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tau)
        #:self.woVoO += 0.5 * lib.einsum('mfbe,ijfe->mbij', eris_ovvv, tau)
        self.Fvv = np.zeros((nvir,nvir), dtype=t1.dtype)
        self.woVoO = np.empty((nocc,nvir,nocc,nocc), dtype=t1.dtype)
        woVvO = np.empty((nocc,nvir,nvir,nocc), dtype=t1.dtype)
        woVVo = np.empty((nocc,nvir,nvir,nocc), dtype=t1.dtype)
        tau = _make_tau(t2, t1, t1)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            self.Fvv += np.einsum('mf,mfae->ae', t1[p0:p1], ovvv) * 2
            self.Fvv -= np.einsum('mf,meaf->ae', t1[p0:p1], ovvv)
            self.woVoO[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', ovvv, tau)
            self.woVoO[p0:p1]+= 0.5 * lib.einsum('mfbe,ijfe->mbij', ovvv, tau)
            woVvO[p0:p1] = lib.einsum('jf,mebf->mbej', t1, ovvv)
            woVVo[p0:p1] = lib.einsum('jf,mfbe->mbej',-t1, ovvv)
            ovvv = None

        eris_ovov = np.asarray(eris.ovov)
        tmp = lib.einsum('njbf,mfne->mbej', t2, eris_ovov)
        woVvO -= tmp * .5
        woVVo += tmp

        eris_ovoo = np.asarray(eris.ovoo)
        ovoo = eris_ovoo + np.einsum('menf,jf->menj', eris_ovov, t1)
        woVvO -= lib.einsum('nb,menj->mbej', t1, ovoo)
        woVVo += lib.einsum('nb,nemj->mbej', t1, ovoo)
        ovoo = None

        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        eris_ovov = tmp = None
        self.Fov = np.einsum('nf,menf->me', t1, ovov)
        tilab = _make_tau(t2, t1, t1, fac=.5)
        self.Foo  = lib.einsum('inef,menf->mi', tilab, ovov)
        self.Fvv -= lib.einsum('mnaf,menf->ae', tilab, ovov)
        theta = t2*2 - t2.transpose(1,0,2,3)
        woVvO += lib.einsum('njfb,menf->mbej', theta, ovov) * .5
        ovov = tilab = None

        self.Foo += foo + 0.5*np.einsum('me,ie->mi', self.Fov+fov, t1)
        self.Fvv += fvv - 0.5*np.einsum('me,ma->ae', self.Fov+fov, t1)

        # 0 or 1 virtuals
        self.woOoO = lib.einsum('je,nemi->mnij', t1, eris_ovoo)
        self.woOoO = self.woOoO + self.woOoO.transpose(1,0,3,2)
        self.woOoO += np.asarray(eris.oooo).transpose(0,2,1,3)
        tmp = lib.einsum('meni,jneb->mbji', eris_ovoo, t2)
        self.woVoO -= tmp.transpose(0,1,3,2) * .5
        self.woVoO -= tmp
        tmp = None
        ovoo = eris_ovoo*2 - eris_ovoo.transpose(2,1,0,3)
        self.woVoO += lib.einsum('nemi,njeb->mbij', ovoo, theta) * .5
        self.woOoV = eris_ovoo.transpose(2,0,3,1).copy()
        self.Foo += np.einsum('ne,nemi->mi', t1, ovoo)
        ovoo = None

        eris_ovov = np.asarray(eris.ovov)
        tau = _make_tau(t2, t1, t1)
        self.woOoO += lib.einsum('ijef,menf->mnij', tau, eris_ovov)
        self.woOoV += lib.einsum('if,mfne->mnie', t1, eris_ovov)
        tau = None

        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        tmp1abba = lib.einsum('njbf,nemf->mbej', t2, eris_ovov)
        eris_ovov = None
        tmp1ab = lib.einsum('nifb,menf->mbei', theta, ovov) * -.5
        tmp1ab+= tmp1abba * .5
        tmpab = lib.einsum('ie,mbej->mbij', t1, tmp1ab)
        tmpab+= lib.einsum('ie,mbej->mbji', t1, tmp1abba)
        self.woVoO -= tmpab
        self.woVoO += np.asarray(eris.ovoo).transpose(3,1,2,0)
        tmpab = None

        # 3 or 4 virtuals
        eris_ovvo = np.asarray(eris.ovvo)
        eris_oovv = np.asarray(eris.oovv)
        woVvO += eris_ovvo.transpose(0,2,1,3)
        woVVo -= eris_oovv.transpose(0,2,3,1)
        self.saved['woVvO'] = woVvO
        self.saved['woVVo'] = woVVo
        self.woVvO = self.saved['woVvO']
        self.woVVo = self.saved['woVVo']

        self.woVoO += lib.einsum('mebj,ie->mbij', eris_ovvo, t1)
        self.woVoO += lib.einsum('mjbe,ie->mbji', eris_oovv, t1)
        self.woVoO += lib.einsum('me,ijeb->mbij', self.Fov, t2)
        self.woVoO -= lib.einsum('nb,mnij->mbij', t1, self.woOoO)

        #:theta = t2*2 - t2.transpose(0,1,3,2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
        #:tmpab = lib.einsum('mebf,miaf->eiab', eris_ovvv, t2)
        #:tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
        #:tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta) * .5
        #:self.wvOvV += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvOvV -= tmpab
        tmp1ab -= eris_ovvo.transpose(0,2,1,3)
        tmp1abba -= eris_oovv.transpose(0,2,3,1)
        eris_ovvo = eris_oovv = None
        tau = _make_tau(t2, t1, t1)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, lib.param.MAX_MEMORY - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*6))))
        for i0,i1 in lib.prange(0, nocc, blksize):
            wvOvV = lib.einsum('meni,mnab->eiab', eris_ovoo[:,:,:,i0:i1], tau)

            wvOvV -= lib.einsum('me,miab->eiab', self.Fov, t2[:,i0:i1])
            tmpab = lib.einsum('ma,mbei->eiab', t1, tmp1ab[:,:,:,i0:i1])
            tmpab+= lib.einsum('ma,mbei->eiba', t1, tmp1abba[:,:,:,i0:i1])
            wvOvV += tmpab

            for p0,p1 in lib.prange(0, nocc, blksize):
                ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
                if p0 == i0:
                    wvOvV += ovvv.transpose(2,0,3,1).conj()
                tmpab = lib.einsum('mebf,miaf->eiab', ovvv, t2[p0:p1,i0:i1])
                tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
                ovvv = ovvv*2 - ovvv.transpose(0,3,2,1)
                tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta[p0:p1,i0:i1]) * .5
                wvOvV -= tmpab
                self.wvOvV[:,i0:i1] = wvOvV
                ovvv = tmpab = None

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)
        return self

def _make_tau(t2, t1, r1, fac=1, out=None):
    tau = np.einsum('ia,jb->ijab', t1, r1)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= fac * .5
    tau += t2
    return tau

def _cp(a):
    return np.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc import rccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = rccsd.RCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    myeom = EOMIP(mycc)
    print("IP energies... (right eigenvector)")
    e,v = ipccsd(myeom, nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = ipccsd(myeom, nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = myeom.ipccsd_star(e, v, lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    myeom = EOMEA(mycc)
    print("EA energies... (right eigenvector)")
    e,v = eaccsd(myeom, nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = eaccsd(myeom, nroots=3, left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = myeom.eaccsd_star(e,v,lv)
    print(e[0] - 0.16656250953550664)
    print(e[1] - 0.23944144521387614)
    print(e[2] - 0.41399436888830721)

    myeom = EOMEESpinFlip(mycc)
    np.random.seed(1)
    v = np.random.random(myeom.vector_size())
    r1, r2 = vector_to_amplitudes_eomsf(v, myeom.nmo, myeom.nocc)
    print(lib.finger(r1)    - 0.017703197938757409)
    print(lib.finger(r2[0]) --21.605764517401415)
    print(lib.finger(r2[1]) - 6.5857056438834842)
    print(abs(amplitudes_to_vector_eomsf(r1, r2) - v).max())

    myeom = EOMEE(mycc)
    e,v = myeom.eeccsd(nroots=1)
    print(e - 0.2757159395886167)

    e,v = myeom.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, koopmans=True)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, guess=v[:4])
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)


    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    myeom = EOMIP(mycc)
    print("IP energies... (right eigenvector)")
    e,v = ipccsd(myeom, nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = ipccsd(myeom, nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = myeom.ipccsd_star(e, v, lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    myeom = EOMEA(mycc)
    print("EA energies... (right eigenvector)")
    e,v = eaccsd(myeom, nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = eaccsd(myeom, nroots=3, left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = myeom.eaccsd_star(e,v,lv)
    print(e[0] - 0.16656250953550664)
    print(e[1] - 0.23944144521387614)
    print(e[2] - 0.41399436888830721)

    myeom = EOMEE(mycc)
    e,v = myeom.eeccsd(nroots=1)
    print(e - 0.2757159395886167)

    e,v = myeom.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, koopmans=True)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = myeom.eeccsd(nroots=4, guess=v[:4])
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
