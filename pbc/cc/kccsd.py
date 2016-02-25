#!/usr/bin/env python

import numpy
import pyscf.pbc.scf
import pyscf.pbc.tools.pbc as tools

import pyscf.cc
import pyscf.cc.ccsd

import time
from functools import reduce

import pyscf.lib
from pyscf.lib import logger

from pyscf.pbc.cc import kintermediates as imdk
from pyscf.pbc import lib as pbclib
einsum = pbclib.einsum

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
        nkpts = len(cc._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), numpy.complex128)
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nkpts, nocc, nvir = t1.shape
    eold = 0
    eccsd = 0

    if cc.diis:
        adiis = pyscf.lib.diis.DIIS(cc, cc.diis_file)
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

def energy(cc, t1, t2, eris):
    nkpts, nocc, nvir = t1.shape
    fock = eris.fock
    eris_oovv = eris.oovv.copy()
    e = 0.0 + 1j * 0.0
    for ki in range(nkpts):
        e += einsum('ia,ia', fock[ki,:nocc,nocc:], t1[ki,:,:])
    t1t1 = numpy.zeros(shape=t2.shape,dtype=t2.dtype)
    for ki in range(nkpts):
        ka = ki
        for kj in range(nkpts):
            #kb = kj
            t1t1[ki,kj,ka,:,:,:,:] = einsum('ia,jb->ijab',t1[ki,:,:],t1[kj,:,:])
    tau = t2 + 2*t1t1
    e += 0.25 * numpy.dot(tau.flatten(), eris_oovv.flatten())
    #e += (0.25*einsum('ijab,ijab',t2,eris_oovv)
    #      + 0.5*einsum('ia,jb,ijab',t1,t1,eris_oovv))
    print "Energy Real / Imaginary = %.15g %.15g" % (e.real, e.imag)
    return e.real

def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nkpts, nocc, nvir = t1.shape
    #nov = nocc*nvir
    fock = eris.fock
    #t1new = numpy.zeros_like(t1)
    #t2new = numpy.zeros_like(t2)

    fov = fock[:,:nocc,nocc:].copy()
    foo = fock[:,:nocc,:nocc].copy()
    fvv = fock[:,nocc:,nocc:].copy()

    #mo_e = eris.fock.diagonal()
    #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    #eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    tau = imdk.make_tau(cc,t2,t1,t1)

    ### From eom-cc hackathon code ###
    Fvv = imdk.cc_Fvv(cc,t1,t2,eris)
    Foo = imdk.cc_Foo(cc,t1,t2,eris)
    Fov = imdk.cc_Fov(cc,t1,t2,eris)
    Woooo = imdk.cc_Woooo(cc,t1,t2,eris)
    Wvvvv = imdk.cc_Wvvvv(cc,t1,t2,eris)
    Wovvo = imdk.cc_Wovvo(cc,t1,t2,eris)

    # Move energy terms to the other side
    Fvv -= fvv
    Foo -= foo

    # Get the momentum conservation array
    # Note: chemist's notation for momentum conserving t2(ki,kj,ka,kb), even though
    # integrals are in physics notation
    kconserv = tools.get_kconserv(cc._scf.cell, cc._kpts)

    eris_ovvo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=t2.dtype)
    eris_oovo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nocc,nocc,nvir,nocc), dtype=t2.dtype)
    eris_vvvo = numpy.zeros(shape=(nkpts,nkpts,nkpts,nvir,nvir,nvir,nocc), dtype=t2.dtype)
    for km in range(nkpts):
        for kb in range(nkpts):
            for ke in range(nkpts):
                kj = kconserv[km,ke,kb]
                # <mb||je> -> -<mb||ej>
                eris_ovvo[km,kb,ke] = -eris.ovov[km,kb,kj].transpose(0,1,3,2)
                # <mn||je> -> -<mn||ej>
                # let kb = kn as a dummy variable
                eris_oovo[km,kb,ke] = -eris.ooov[km,kb,kj].transpose(0,1,3,2)
                # <ma||be> -> - <be||am>*
                # let kj = ka as a dummy variable
                kj = kconserv[km,ke,kb]
                eris_vvvo[ke,kj,kb] = -eris.ovvv[km,kb,ke].transpose(2,3,1,0).conj()

    # T1 equation
    t1new = numpy.zeros(shape=t1.shape, dtype=t1.dtype)
    for ka in range(nkpts):
        ki = ka
        # TODO: Does this fov need a conj()? Usually zero w/ canonical HF.
        t1new[ka] += fov[ka,:,:]
        t1new[ka] +=  einsum('ie,ae->ia',t1[ka],Fvv[ka])
        t1new[ka] += -einsum('ma,mi->ia',t1[ka],Foo[ka])
        for km in range(nkpts):
            t1new[ka] +=  einsum('imae,me->ia',t2[ka,km,ka],Fov[km])
            t1new[ka] += -einsum('nf,naif->ia',t1[km],eris.ovov[km,ka,ki])
            for kn in range(nkpts):
                ke = kconserv[km,ki,kn]
                t1new[ka] += -0.5*einsum('imef,maef->ia',t2[ki,km,ke],eris.ovvv[km,ka,ke])
                t1new[ka] += -0.5*einsum('mnae,nmei->ia',t2[km,kn,ka],eris_oovo[kn,km,ke])

    # T2 equation
    # For conj(), see Hirata and Bartlett, Eq. (36)
    t2new = eris.oovv.copy().conj()

    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
            kb = kconserv[ki,ka,kj]

            Ftmp = Fvv[kb] - 0.5*einsum('mb,me->be',t1[kb],Fov[kb])
            tmp = einsum('ijae,be->ijab',t2[ki,kj,ka],Ftmp)
            t2new[ki,kj,ka] += tmp

            Ftmp = Fvv[ka] - 0.5*einsum('ma,me->ae',t1[ka],Fov[ka])
            tmp = einsum('ijbe,ae->ijab',t2[ki,kj,kb],Ftmp)
            t2new[ki,kj,ka] -= tmp

            #t2new[ki,kj,kb] -= tmp.transpose(0,1,3,2)

            Ftmp = Foo[kj] + 0.5*einsum('je,me->mj',t1[kj],Fov[kj])
            tmp = einsum('imab,mj->ijab',t2[ki,kj,ka],Ftmp)
            t2new[ki,kj,ka] -= tmp

            Ftmp = Foo[ki] + 0.5*einsum('ie,me->mi',t1[ki],Fov[ki])
            tmp = einsum('jmab,mi->ijab',t2[kj,ki,ka],Ftmp)
            t2new[ki,kj,ka] += tmp

            #t2new[kj,ki,ka] += tmp.transpose(1,0,2,3)

            for km in range(nkpts):
                # Wminj
                #     - km - kn + ka + kb = 0
                # =>  kn = ka - km + kb
                kn = kconserv[ka,km,kb]
                t2new[ki,kj,ka] += 0.5*einsum('mnab,mnij->ijab',tau[km,kn,ka],Woooo[km,kn,ki])
                ke = km
                t2new[ki,kj,ka] += 0.5*einsum('ijef,abef->ijab',tau[ki,kj,ke],Wvvvv[ka,kb,ke])

                # Wmbej
                #     - km - kb + ke + kj = 0
                #  => ke = km - kj + kb
                ke = kconserv[km,kj,kb]
                tmp = einsum('imae,mbej->ijab',t2[ki,km,ka],Wovvo[km,kb,ke])
                #     - km - kb + ke + kj = 0
                # =>  ke = km - kj + kb
                #
                # t[i,e] => ki = ke
                # t[m,a] => km = ka
                if km == ka and ke == ki:
                    tmp -= einsum('ie,ma,mbej->ijab',t1[ki],t1[km],eris_ovvo[km,kb,ke])
                t2new[ki,kj,ka] += tmp
                t2new[ki,kj,kb] -= tmp.transpose(0,1,3,2)
                t2new[kj,ki,ka] -= tmp.transpose(1,0,2,3)
                t2new[kj,ki,kb] += tmp.transpose(1,0,3,2)

            ke = ki
            tmp = einsum('ie,abej->ijab',t1[ki],eris_vvvo[ka,kb,ke])
            t2new[ki,kj,ka] += tmp
            # P(ij) term
            ke = kj
            tmp = einsum('je,abei->ijab',t1[kj],eris_vvvo[ka,kb,ke])
            t2new[ki,kj,ka] -= tmp

            km = ka
            tmp = einsum('ma,mbij->ijab',t1[ka],eris.ovoo[km,kb,ki])
            t2new[ki,kj,ka] -= tmp
            # P(ab) term
            km = kb
            tmp = einsum('mb,maij->ijab',t1[kb],eris.ovoo[km,ka,ki])
            t2new[ki,kj,ka] += tmp

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


class CCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, abs_kpts, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        self._kpts = abs_kpts
        nkpts = len(self._kpts)

        nmo = mf.cell.nao_nr()
        nso = 2*nmo
        # calculating spin orbitals...
        if mo_energy is None:
            mo_energy = numpy.zeros(shape=(nkpts,nso))
            mo_energy[:,0::2] = mo_energy[:,1::2] = mf.mo_energy
        if mo_coeff is None:
            # TODO: Careful for real/complex here, in the future
            so_coeffT = numpy.zeros((nkpts,nso,nso), dtype=numpy.complex128)
            mo_coeffT = numpy.zeros((nkpts,nmo,nmo), dtype=numpy.complex128)
            mo_coeff = numpy.zeros((nkpts,nso,nso), dtype=numpy.complex128)
            for k in range(nkpts):
                mo_coeffT[k] = numpy.conj(mf.mo_coeff[k]).T
            for k in range(nkpts):
                for i in range(nso):
                    if i%2 == 0:
                        so_coeffT[k][i][:nso/2] = mo_coeffT[k][i//2]
                    else:
                        so_coeffT[k][i][nso/2:] = mo_coeffT[k][i//2]
            # Each col is an eigenvector, first n/2 rows are alpha, then n/2 beta
            for k in range(nkpts):
                mo_coeff[k] = numpy.conj(so_coeffT[k]).T
        if mo_occ is None:
            mo_occ = numpy.zeros((nkpts,nso))
            for k in range(nkpts):
                mo_occ[k,0:mf.cell.nelectron] = 1

        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)


    def nocc(self):
        # Spin orbitals
        # TODO: Possibly change this to make it work with k-points with frozen
        #       As of right now it works, but just not sure how the frozen list will work
        #       with it
        self._nocc = 2*pyscf.cc.ccsd.CCSD.nocc(self)
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

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        nkpts = len(self._kpts)
        t1 = numpy.zeros((nkpts,nocc,nvir), dtype=numpy.complex128)
        t2 = numpy.zeros((nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=numpy.complex128)
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
                        eia[i,a] = foo[ki,i,i] - fvv[ka,a,a]
                        for j in range(nocc):
                            for b in range(nvir):
                                eijab[i,j,a,b] = ( foo[ki,i,i] + foo[kj,j,j]
                                                 - fvv[ka,a,a] - fvv[kb,b,b] )
                                t2[ki,kj,ka,i,j,a,b] = eris_oovv[ki,kj,ka,i,j,a,b]/eijab[i,j,a,b]

        t2 = numpy.conj(t2)
        self.emp2 = 0.25*numpy.einsum('pqrijab,pqrijab',t2,eris_oovv).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        print "MP2 energy =", self.emp2
        return self.emp2, t1, t2

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
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
    """_ERIS handler for PBCs."""
    def __init__(self, cc, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(shape=cc.mo_energy.shape, dtype=numpy.bool)
        nkpts = len(cc._kpts)
        nmo = cc.nmo()
        # TODO change this for k-points ... seems like it should work
        if isinstance(cc.frozen, (int, numpy.integer)):
            for k in range(nkpts):
                moidx[k,:cc.frozen] = False
        elif len(cc.frozen) > 0:
            for k in range(nkpts):
                moidx[k,numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            # TODO make this work for frozen maybe... seems like it should work
            self.mo_coeff = numpy.zeros((nkpts,nmo,nmo),dtype=numpy.complex128)
            for k in range(nkpts):
                self.mo_coeff[k] = cc.mo_coeff[k][:,moidx[k]]
            mo_coeff = self.mo_coeff
            self.fock = numpy.zeros((len(cc._kpts),nmo,nmo))
            for k in range(nkpts):
                self.fock[k] = numpy.diag(cc.mo_energy[k][moidx[k]])
        else:  # If mo_coeff is not canonical orbital
            # TODO does this work for k-points? changed to conjugate.
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (numpy.conj(mo_coeff.T), fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        # Convert to spin-orbitals and anti-symmetrize
        so_coeff = numpy.zeros((nkpts,nmo/2,nmo),dtype=numpy.complex128)
        so_coeff[:,:,::2] = so_coeff[:,:,1::2] = mo_coeff[:,:nmo/2,::2]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):

            kconserv = tools.get_kconserv(cc._scf.cell,cc._kpts)

            eri = numpy.zeros((nkpts,nkpts,nkpts,nmo,nmo,nmo,nmo), dtype=numpy.complex128)
            for kp in range(nkpts):
                for kq in range(nkpts):
                    for kr in range(nkpts):
                        ks = kconserv[kp,kq,kr]
                        eri_kpt = pyscf.pbc.ao2mo.general(cc._scf.cell,
                                    (so_coeff[kp,:,:],so_coeff[kq,:,:],so_coeff[kr,:,:],so_coeff[ks,:,:]),
                                    (cc._kpts[kp],cc._kpts[kq],cc._kpts[kr],cc._kpts[ks]))
                        eri_kpt = eri_kpt.reshape(nmo,nmo,nmo,nmo)
                        eri[kp,kq,kr] = eri_kpt.copy()

            eri[:,:,:,::2,1::2] = eri[:,:,:,1::2,::2] = eri[:,:,:,:,:,::2,1::2] = eri[:,:,:,:,:,1::2,::2] = 0.

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

            # Antisymmetrizing (pq|rs)-(ps|rq), where the latter integral is equal to
            # (rq|ps); done since we aren't tracking the kpoint of orbital 's'
            eri1 = eri - eri.transpose(2,1,0,5,4,3,6)
            # Chemist -> physics notation
            eri1 = eri1.transpose(0,2,1,3,5,4,6)

            self.dtype = eri1.dtype
            self.oooo = eri1[:,:,:,:nocc,:nocc,:nocc,:nocc].copy() / nkpts
            self.ooov = eri1[:,:,:,:nocc,:nocc,:nocc,nocc:].copy() / nkpts
            self.ovoo = eri1[:,:,:,:nocc,nocc:,:nocc,:nocc].copy() / nkpts
            self.oovv = eri1[:,:,:,:nocc,:nocc,nocc:,nocc:].copy() / nkpts
            self.ovov = eri1[:,:,:,:nocc,nocc:,:nocc,nocc:].copy() / nkpts
            self.ovvv = eri1[:,:,:,:nocc,nocc:,nocc:,nocc:].copy() / nkpts
            self.vvvv = eri1[:,:,:,nocc:,nocc:,nocc:,nocc:].copy() / nkpts
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)


        log.timer('CCSD integral transformation', *cput0)


def check_antisymm_12( cc, kpts, integrals ):
    kconserv = tools.get_kconserv(cc._scf.cell,cc._kpts)
    nkpts = len(kpts)
    diff = 0.0
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp,kr,kq]
                for p in range(integrals.shape[3]):
                    for q in range(integrals.shape[4]):
                        for r in range(integrals.shape[5]):
                            for s in range(integrals.shape[6]):
                                pqrs = integrals[kp,kq,kr,p,q,r,s]
                                qprs = integrals[kq,kp,kr,q,p,r,s]
                                cdiff = numpy.linalg.norm(pqrs+qprs).real
                                print "AS diff = %.15g" % cdiff, pqrs, qprs, kp, kq, kr, ks, p, q, r, s
                                diff = max(diff,cdiff)
    print "antisymmetrization : max diff = %.15g" % diff

def check_antisymm_34( cc, kpts, integrals ):
    kconserv = tools.get_kconserv(cc._scf.cell,cc._kpts)
    nkpts = len(kpts)
    diff = 0.0
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp,kr,kq]
                for p in range(integrals.shape[3]):
                    for q in range(integrals.shape[4]):
                        for r in range(integrals.shape[5]):
                            for s in range(integrals.shape[6]):
                                pqrs = integrals[kp,kq,kr,p,q,r,s]
                                pqsr = integrals[kp,kq,ks,p,q,s,r]
                                cdiff = numpy.linalg.norm(pqrs+pqsr).real
                                print "AS diff = %.15g" % cdiff, pqrs, pqsr, kp, kq, kr, ks, p, q, r, s
                                diff = max(diff,cdiff)
    print "antisymmetrization : max diff = %.15g" % diff

