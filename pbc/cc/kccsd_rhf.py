import time
import tempfile
import numpy
import numpy as np
import h5py

import pyscf.pbc.tools.pbc as tools
from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
from pyscf.cc.ccsd import _cp
from pyscf.pbc.cc import kintermediates_rhf as imdk
from pyscf.pbc.lib.linalg_helper import eigs

#einsum = np.einsum
einsum = pbclib.einsum

# This is restricted (R)CCSD
# following Hirata, ..., Barlett, J. Chem. Phys. 120, 2581 (2004)

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nkpts, nocc, nvir = t1.shape
    eold = 0.0 + 1j*0.0
    eccsd = 0.0 + 1j*0.0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:,:nocc,nocc:]
    foo = fock[:,:nocc,:nocc]
    fvv = fock[:,nocc:,nocc:]

    #mo_e = eris.fock.diagonal()
    #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Loo = imdk.Loo(cc,t1,t2,eris)
    Lvv = imdk.Lvv(cc,t1,t2,eris)
    Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    Wvoov = imdk.cc_Wvoov(cc,t1,t2,eris)
    Wvovo = imdk.cc_Wvovo(cc,t1,t2,eris)

    # Move energy terms to the other side
    Foo -= foo
    Fvv -= fvv
    Loo -= foo
    Lvv -= fvv

    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    # T1 equation
    # TODO: Check this conj(). Hirata and Bartlett has
    # f_{vo}(a,i), which should be equal to f_{ov}^*(i,a)
    t1new = fov.conj().copy()
    for ka in range(nkpts):
        ki = ka
        # kc == ki; kk == ka
        t1new[ka] += -2.*einsum('kc,ka,ic->ia',fov[ki],t1[ka],t1[ki])
        t1new[ka] += einsum('ac,ic->ia',Fvv[ka],t1[ki])
        t1new[ka] += -einsum('ki,ka->ia',Foo[ki],t1[ka])
        for kk in range(nkpts):
            kc = kk
            t1new[ka] +=  einsum('kc,kica->ia',Fov[kc],2*t2[kk,ki,kc])
            t1new[ka] +=  einsum('kc,ikca->ia',Fov[kc], -t2[ki,kk,kc])

            # NOTE: can move this summation out of kk ...
            if kk == ka:
                t1new[ka] +=  einsum('kc,ic,ka->ia',Fov[kc],t1[ki],t1[ka])
            t1new[ka] +=  einsum('akic,kc->ia',2*eris.voov[ka,kk,ki],t1[kc])
            t1new[ka] +=  einsum('akci,kc->ia', -eris.vovo[ka,kk,kc],t1[kc])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kk]
                t1new[ka] +=  einsum('akcd,ikcd->ia',2*eris.vovv[ka,kk,kc],t2[ki,kk,kc])
                t1new[ka] +=  einsum('akdc,ikcd->ia', -eris.vovv[ka,kk,kd],t2[ki,kk,kc])
                # NOTE: can move these 2 summations out of kc

                if ki == kc and kk == kd:
                    t1new[ka] +=  einsum('akcd,ic,kd->ia',2*eris.vovv[ka,kk,ki],t1[ki],t1[kk])
                    t1new[ka] +=  einsum('akdc,ic,kd->ia', -eris.vovv[ka,kk,kk],t1[ki],t1[kk])

                # kk - ki + kl = kc
                #  => kl = ki - kk + kc
                kl = kconserv[ki,kk,kc]
                t1new[ka] += -einsum('klic,klac->ia',2*eris.ooov[kk,kl,ki],t2[kk,kl,ka])
                t1new[ka] += -einsum('klci,klac->ia', -eris.oovo[kk,kl,kc],t2[kk,kl,ka])

                # NOTE: can move these 2 summations out of kk
                if kk == ka and kl == kc:
                    t1new[ka] += -einsum('klic,ka,lc->ia',2*eris.ooov[ka,kc,ki],t1[ka],t1[kc])
                    t1new[ka] += -einsum('klci,ka,lc->ia', -eris.oovo[ka,kc,kc],t1[ka],t1[kc])

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = np.array(eris.oovv, copy=True).conj()
    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
            kb = kconserv[ki,ka,kj]

            for kl in range(nkpts):
                # kk - ki + kl = kj
                # => kk = kj - kl + ki
                kk = kconserv[kj,kl,ki]
                t2new[ki,kj,ka] += einsum('klij,klab->ijab',Woooo[kk,kl,ki],t2[kk,kl,ka])
                if kl == kb and kk == ka:
                    t2new[ki,kj,ka] += einsum('klij,ka,lb->ijab',Woooo[ka,kb,ki],t1[ka],t1[kb])

            for kc in range(nkpts):
                kd = kconserv[ka,kc,kb]
                t2new[ki,kj,ka] += einsum('abcd,ijcd->ijab',Wvvvv[ka,kb,kc],t2[ki,kj,kc])
                if ki == kc and kj == kd:
                    t2new[ki,kj,ka] += einsum('abcd,ic,jd->ijab',Wvvvv[ka,kb,ki],t1[ki],t1[kj])

            t2new[ki,kj,ka] += einsum('ac,ijcb->ijab',Lvv[ka],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('bc,jica->ijab',Lvv[kb],t2[kj,ki,kb])

            t2new[ki,kj,ka] += einsum('ki,kjab->ijab',-Loo[ki],t2[ki,kj,ka])
            #P(ij)P(ab)
            t2new[ki,kj,ka] += einsum('kj,kiba->ijab',-Loo[kj],t2[kj,ki,kb])

            tmp2 = eris.vvov[ka,kb,ki] - einsum('kbic,ka->abic',eris.ovov[ka,kb,ki],t1[ka])
            tmp  = einsum('abic,jc->ijab',tmp2,t1[kj])
            t2new[ki,kj,ka] += tmp
            #P(ij)P(ab)
            tmp2 = eris.vvov[kb,ka,kj] - einsum('kajc,kb->bajc',eris.ovov[kb,ka,kj],t1[kb])
            tmp  = einsum('bajc,ic->ijab',tmp2,t1[ki])
            t2new[ki,kj,ka] += tmp

            # ka - ki + kk = kj
            # => kk = ki - ka + kj
            kk = kconserv[ki,ka,kj]
            tmp2 = eris.vooo[ka,kk,ki] + einsum('akic,jc->akij',eris.voov[ka,kk,ki],t1[kj])
            tmp  = einsum('akij,kb->ijab',tmp2,t1[kb])
            t2new[ki,kj,ka] -= tmp
            #P(ij)P(ab)
            kk = kconserv[kj,kb,ki]
            tmp2 = eris.vooo[kb,kk,kj] + einsum('bkjc,ic->bkji',eris.voov[kb,kk,kj],t1[ki])
            tmp  = einsum('bkji,ka->ijab',tmp2,t1[ka])
            t2new[ki,kj,ka] -= tmp

            for kk in range(nkpts):
                kc = kconserv[ka,ki,kk]
                tmp = 2*einsum('akic,kjcb->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kc]) - \
                        einsum('akci,kjcb->ijab',Wvovo[ka,kk,kc],t2[kk,kj,kc])
                t2new[ki,kj,ka] += tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp = 2*einsum('bkjc,kica->ijab',Wvoov[kb,kk,kj],t2[kk,ki,kc]) - \
                        einsum('bkcj,kica->ijab',Wvovo[kb,kk,kc],t2[kk,ki,kc])
                t2new[ki,kj,ka] += tmp

                kc = kconserv[ka,ki,kk]
                tmp = einsum('akic,kjbc->ijab',Wvoov[ka,kk,ki],t2[kk,kj,kb])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kb,kj,kk]
                tmp = einsum('bkjc,kiac->ijab',Wvoov[kb,kk,kj],t2[kk,ki,ka])
                t2new[ki,kj,ka] -= tmp

                kc = kconserv[kk,ka,kj]
                tmp = einsum('bkci,kjac->ijab',Wvovo[kb,kk,kc],t2[kk,kj,ka])
                t2new[ki,kj,ka] -= tmp
                #P(ij)P(ab)
                kc = kconserv[kk,kb,ki]
                tmp = einsum('akcj,kibc->ijab',Wvovo[ka,kk,kc],t2[kk,ki,kb])
                t2new[ki,kj,ka] -= tmp

    eia = numpy.zeros(shape=t1new.shape, dtype=t1new.dtype)
    for ki in range(nkpts):
        for i in range(nocc):
            for a in range(nvir):
                eia[ki,i,a] = foo[ki,i,i] - fvv[ki,a,a]
        t1new[ki] /= eia[ki]

    eijab = numpy.zeros(shape=t2new.shape, dtype=t2new.dtype)
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                for i in range(nocc):
                    for a in range(nvir):
                        for j in range(nocc):
                            for b in range(nvir):
                                eijab[ki,kj,ka,i,j,a,b] = ( foo[ki,i,i] + foo[kj,j,j]
                                                          - fvv[ka,a,a] - fvv[kb,b,b] )
                t2new[ki,kj,ka] /= eijab[ki,kj,ka]

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    kconserv = tools.get_kconserv(cc._scf.cell,cc._kpts)
    fock = eris.fock
    e = 0.0 + 1j*0.0
    for ki in range(nkpts):
        e += 2*einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka] = einsum('ia,jb->ijab',t1[ki],t1[kj])
    tau = t2 + t1t1
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                e += einsum('ijab,ijab', 2*tau[ki,kj,ka], eris.oovv[ki,kj,ka])
                e += einsum('ijab,ijba',  -tau[ki,kj,ka], eris.oovv[ki,kj,kb])
    e /= nkpts
    return e.real


class RCCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, abs_kpts, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        self._kpts = abs_kpts
        nkpts = len(self._kpts)
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** EOM CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = len(self._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        woovv = numpy.empty((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
        self.emp2 = 0
        foo = eris.fock[:,:nocc,:nocc].copy()
        fvv = eris.fock[:,nocc:,nocc:].copy()
        eris_oovv = eris.oovv.copy()
        eia = numpy.zeros((nocc,nvir))
        eijab = numpy.zeros((nocc,nocc,nvir,nvir))

        kconserv = tools.get_kconserv(self._scf.cell,self._kpts)
        for ki in range(nkpts):
          for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[ki,ka,kj]
                for i in range(nocc):
                    for a in range(nvir):
                        #eia[i,a] = foo[ki,i,i] - fvv[ka,a,a]
                        for j in range(nocc):
                            for b in range(nvir):
                                eijab[i,j,a,b] = ( foo[ki,i,i] + foo[kj,j,j]
                                                 - fvv[ka,a,a] - fvv[kb,b,b] ).real
                                woovv[ki,kj,ka,i,j,a,b] = (2*eris_oovv[ki,kj,ka,i,j,a,b] - eris_oovv[ki,kj,kb,i,j,b,a])
                                t2   [ki,kj,ka,i,j,a,b] = eris_oovv[ki,kj,ka,i,j,a,b] / eijab[i,j,a,b]


        t2 = numpy.conj(t2)
        self.emp2 = numpy.einsum('pqrijab,pqrijab',t2,woovv).real
        self.emp2 /= nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def nocc(self):
        # Spin orbitals
        # TODO: Possibly change this to make it work with k-points with frozen
        #       As of right now it works, but just not sure how the frozen list will work
        #       with it
        self._nocc = pyscf.cc.ccsd.CCSD.nocc(self)
        self._nocc = (self._nocc // len(self._kpts))
        return self._nocc

    def nmo(self):
        # TODO: Change this for frozen at k-points, seems like it should work
        if isinstance(self.frozen, (int, numpy.integer)):
            self._nmo = len(self.mo_energy[0]) - self.frozen
        else:
            if len(self.frozen) > 0:
                self._nmo = len(self.mo_energy[0]) - len(self.frozen[0])
            else:
                self._nmo = len(self.mo_energy[0])
        return self._nmo

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self.eris = eris
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        if self._conv:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.ecc)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.e_tot, self.ecc)
        return self.ecc, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.shape, dtype=numpy.bool)
        nkpts = len(cc._kpts)
        nmo = cc.nmo()
        #TODO check that this and kccsd work for frozen...
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:,:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[:,numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.mo_coeff[kp] = cc.mo_coeff[kp][:,moidx[kp]]
            mo_coeff = self.mo_coeff
            self.fock = numpy.zeros((nkpts,nmo,nmo),dtype=cc.mo_coeff.dtype)
            for kp in range(nkpts):
                self.fock[kp] = numpy.diag(cc.mo_energy[kp][moidx[kp]]).astype(mo_coeff.dtype)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            kconserv = tools.get_kconserv(cc._scf.cell,cc._kpts)

            eri = numpy.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo), dtype=numpy.complex128)
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                    (mo_coeff[kp,:,:],mo_coeff[kq,:,:],mo_coeff[kr,:,:],mo_coeff[ks,:,:]),
                                    (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                        eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                        eri[kp,kq,kr] = eri_kpt.copy()

            # Checking some things...
            maxdiff = 0.0
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        for p in range(nmo):
                            for q in range(nmo):
                                for r in range(nmo):
                                    for s in range(nmo):
                                        pqrs = eri[kp,kq,kr,p,q,r,s]
                                        rspq = eri[kr,ks,kp,r,s,p,q]
                                        diff = numpy.linalg.norm(pqrs - rspq).real
                                        if diff > 1e-5:
                                            print "** Warning: ERI diff at ",
                                            print "kp,kq,kr,ks,p,q,r,s =", kp, kq, kr, ks, p, q, r, s
                                        maxdiff = max(maxdiff,diff)
            print "Max difference in (pq|rs) - (rs|pq) = %.15g" % maxdiff
            #print "ERI ="
            #print eri

            # Chemist -> physics notation
            eri = eri.transpose(0,2,1,3,5,4,6)

            self.dtype = eri.dtype
            self.oooo = eri[:,:,:,:nocc,:nocc,:nocc,:nocc].copy() / nkpts
            self.ooov = eri[:,:,:,:nocc,:nocc,:nocc,nocc:].copy() / nkpts
            self.ovoo = eri[:,:,:,:nocc,nocc:,:nocc,:nocc].copy() / nkpts
            self.oovv = eri[:,:,:,:nocc,:nocc,nocc:,nocc:].copy() / nkpts
            self.ovov = eri[:,:,:,:nocc,nocc:,:nocc,nocc:].copy() / nkpts
            self.ovvv = eri[:,:,:,:nocc,nocc:,nocc:,nocc:].copy() / nkpts
            self.vvvv = eri[:,:,:,nocc:,nocc:,nocc:,nocc:].copy() / nkpts
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

            # TODO: Avoid this.
            # Store all for now, while DEBUGGING
            self.voov = eri[:,:,:,nocc:,:nocc,:nocc,nocc:].copy() / nkpts
            self.vovo = eri[:,:,:,nocc:,:nocc,nocc:,:nocc].copy() / nkpts
            self.vovv = eri[:,:,:,nocc:,:nocc,nocc:,nocc:].copy() / nkpts
            self.oovo = eri[:,:,:,:nocc,:nocc,nocc:,:nocc].copy() / nkpts
            self.vvov = eri[:,:,:,nocc:,nocc:,:nocc,nocc:].copy() / nkpts
            self.vooo = eri[:,:,:,nocc:,:nocc,:nocc,:nocc].copy() / nkpts


        log.timer('CCSD integral transformation', *cput0)
