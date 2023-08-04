#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
QCISD for real integrals
8-fold permutation symmetry has been used
(ij|kl) = (ji|kl) = (kl|ij) = ...
'''


import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.ccsd import CCSD, _add_vvvv, _flops, _ChemistsERIs
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


# t1: ia
# t2: ijab
def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(QCISD) = %.15g', eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = numpy.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(QCISD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('QCISD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('QCISD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(mycc, t1, t2, eris):
    assert (isinstance(eris, _ChemistsERIs))

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    t1new = numpy.zeros_like(t1)
    t2new = _add_vvvv(mycc, 0*t1, t2, eris, t2sym='jiba')
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc] - numpy.diag(mo_e_o)
    fvv = fock[nocc:,nocc:] - numpy.diag(mo_e_v)

    if mycc.incore_complete:
        fswap = None
    else:
        fswap = lib.H5TmpFile()
    fwVOov, fwVooV = _add_ovvv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap)
    time1 = log.timer_debug1('ovvv', *time1)

    woooo = numpy.asarray(eris.oooo).transpose(0,2,1,3).copy()

    unit = nocc**2*nvir*7 + nocc**3 + nocc*nvir**2
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-nocc**4)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    for p0, p1 in lib.prange(0, nvir, blksize):
        wVOov = fwVOov[p0:p1]
        wVooV = fwVooV[p0:p1]
        eris_ovoo = eris.ovoo[:,p0:p1]
        eris_oovv = numpy.empty((nocc,nocc,p1-p0,nvir))
        def load_oovv(p0, p1):
            eris_oovv[:] = eris.oovv[:,:,p0:p1]
        with lib.call_in_background(load_oovv, sync=not mycc.async_io) as prefetch_oovv:
            #:eris_oovv = eris.oovv[:,:,p0:p1]
            prefetch_oovv(p0, p1)
            wVOov -= lib.einsum('jbik,ka->bjia', eris_ovoo, t1)
            t2new[:,:,p0:p1] += wVOov.transpose(1,2,0,3)
            eris_ovoo = None
        load_oovv = prefetch_oovv = None

        wVOov *= 0 # QCI

        eris_ovvo = numpy.empty((nocc,p1-p0,nvir,nocc))
        def load_ovvo(p0, p1):
            eris_ovvo[:] = eris.ovvo[:,p0:p1]
        with lib.call_in_background(load_ovvo, sync=not mycc.async_io) as prefetch_ovvo:
            #:eris_ovvo = eris.ovvo[:,p0:p1]
            prefetch_ovvo(p0, p1)
            t1new[:,p0:p1] -= numpy.einsum('jb,jiab->ia', t1, eris_oovv)
            wVooV -= eris_oovv.transpose(2,0,1,3)
            wVOov += wVooV*.5  #: bjia + bija*.5
        load_ovvo = prefetch_ovvo = None

        t2new[:,:,p0:p1] += (eris_ovvo*0.5).transpose(0,3,1,2)
        eris_voov = eris_ovvo.conj().transpose(1,0,3,2)
        t1new[:,p0:p1] += 2*numpy.einsum('jb,aijb->ia', t1, eris_voov)
        eris_ovvo = None
        eris_oovv = tmp = None

        fov[:,p0:p1] += numpy.einsum('kc,aikc->ia', t1, eris_voov) * 2
        fov[:,p0:p1] -= numpy.einsum('kc,akic->ia', t1, eris_voov)

        tau = t2[:,:,p0:p1]
        theta  = tau.transpose(1,0,2,3) * 2
        theta -= tau
        fvv -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo += lib.einsum('aikb,kjab->ij', eris_voov, theta)
        theta = None
        woooo += lib.einsum('ijab,aklb->ijkl', tau, eris_voov)
        tau = None

        def update_wVooV(q0, q1, tau):
            wVooV[:] += lib.einsum('bkic,jkca->bija', eris_voov[:,:,:,q0:q1], tau)
        with lib.call_in_background(update_wVooV, sync=not mycc.async_io) as update_wVooV:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tau  = t2[:,:,q0:q1] * .5
                #:wVooV += lib.einsum('bkic,jkca->bija', eris_voov[:,:,:,q0:q1], tau)
                update_wVooV(q0, q1, tau)
        tau = update_wVooV = None
        def update_t2(q0, q1, tmp):
            t2new[:,:,q0:q1] += tmp.transpose(2,0,1,3)
            tmp *= .5
            t2new[:,:,q0:q1] += tmp.transpose(0,2,1,3)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tmp = lib.einsum('jkca,ckib->jaib', t2[:,:,p0:p1,q0:q1], wVooV)
                #:t2new[:,:,q0:q1] += tmp.transpose(2,0,1,3)
                #:tmp *= .5
                #:t2new[:,:,q0:q1] += tmp.transpose(0,2,1,3)
                update_t2(q0, q1, tmp)
                tmp = None

        wVOov += eris_voov
        eris_VOov = -.5 * eris_voov.transpose(0,2,1,3)
        eris_VOov += eris_voov
        eris_voov = None
        def update_wVOov(q0, q1, tau):
            wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau)
        with lib.call_in_background(update_wVOov, sync=not mycc.async_io) as update_wVOov:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tau  = t2[:,:,q0:q1].transpose(1,3,0,2) * 2
                tau -= t2[:,:,q0:q1].transpose(0,3,1,2)
                #:wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau)
                update_wVOov(q0, q1, tau)
                tau = None
        def update_t2(q0, q1, theta):
            t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                theta  = t2[:,:,p0:p1,q0:q1] * 2
                theta -= t2[:,:,p0:p1,q0:q1].transpose(1,0,2,3)
                #:t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
                update_t2(q0, q1, theta)
                theta = None
        eris_VOov = wVOov = wVooV = update_wVOov = None
        time1 = log.timer_debug1('voov [%d:%d]'%(p0, p1), *time1)
    fwVOov = fwVooV = fswap = None

    for p0, p1 in lib.prange(0, nvir, blksize):
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        t1new += numpy.einsum('jb,ijba->ia', fov[:,p0:p1], theta)
        t1new -= lib.einsum('jbki,kjba->ia', eris.ovoo[:,p0:p1], theta)

        tau = t2[:,:,p0:p1]
        t2new[:,:,p0:p1] += .5 * lib.einsum('ijkl,klab->ijab', woooo, tau)
        theta = tau = None

    t2new += lib.einsum('ijac,bc->ijab', t2, fvv)
    t2new -= lib.einsum('ki,kjab->ijab', foo, t2)

    eia = mo_e_o[:,None] - mo_e_v
    t1new += numpy.einsum('ib,ab->ia', t1, fvv)
    t1new -= numpy.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2new[i,:i] += t2new[:i,i].transpose(0,2,1)
            t2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            t2new[:i,i] = t2new[i,:i].transpose(0,2,1)
        t2new[i,i] = t2new[i,i] + t2new[i,i].T
        t2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new

def _add_ovvv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap):
    time1 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir+1) // 2

    if fswap is None:
        wVOov = numpy.zeros((nvir,nocc,nocc,nvir))
    else:
        wVOov = fswap.create_dataset('wVOov', (nvir,nocc,nocc,nvir), 'f8')
    wooVV = numpy.zeros((nocc,nocc*nvir_pair))

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2*3 + nocc**2*nvir + 2
    blksize = min(nvir, max(BLKMIN, int((max_memory*.95e6/8-wooVV.size)/unit)))
    if not mycc.direct:
        unit = nocc*nvir**2*3 + nocc**2*nvir + 2 + nocc*nvir**2 + nocc*nvir
        blksize = min(nvir, max(BLKMIN, int((max_memory*.95e6/8-wooVV.size-nocc**2*nvir)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    def load_ovvv(buf, p0):
        if p0 < nvir:
            p1 = min(nvir, p0+blksize)
            buf[:p1-p0] = eris.ovvv[:,p0:p1].transpose(1,0,2)

    with lib.call_in_background(load_ovvv, sync=not mycc.async_io) as prefetch:
        buf = numpy.empty((blksize,nocc,nvir_pair))
        buf_prefetch = numpy.empty((blksize,nocc,nvir_pair))

        load_ovvv(buf_prefetch, 0)
        for p0, p1 in lib.prange(0, nvir, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, p1)

            eris_vovv = buf[:p1-p0]
            eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,nvir_pair))
            eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)

            wVOov[p0:p1] = lib.einsum('biac,jc->bija', eris_vovv, t1)

            theta = t2[:,:,p0:p1].transpose(1,2,0,3) * 2
            theta -= t2[:,:,p0:p1].transpose(0,2,1,3)
            t1new += lib.einsum('icjb,cjba->ia', theta, eris_vovv)
            theta = None
            time1 = log.timer_debug1('vovv [%d:%d]'%(p0, p1), *time1)

    if fswap is None:
        wooVV = lib.unpack_tril(wooVV.reshape(nocc**2,nvir_pair))
        return wVOov, wooVV.reshape(nocc,nocc,nvir,nvir).transpose(2,1,0,3)
    else:
        fswap.create_dataset('wVooV', (nvir,nocc,nocc,nvir), 'f8')
        wooVV = wooVV.reshape(nocc,nocc,nvir_pair)
        tril2sq = lib.square_mat_in_trilu_indices(nvir)
        for p0, p1 in lib.prange(0, nvir, blksize):
            fswap['wVooV'][p0:p1] = wooVV[:,:,tril2sq[p0:p1]].transpose(2,1,0,3)
        return fswap['wVOov'], fswap['wVooV']

def energy(mycc, t1=None, t2=None, eris=None):
    '''CCSD correlation energy'''
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvo = eris.ovvo[:,p0:p1]
        tau = t2[:,:,p0:p1]
        e += 2 * numpy.einsum('ijab,iabj', tau, eris_ovvo)
        e -=     numpy.einsum('jiab,iabj', tau, eris_ovvo)
    if abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in QCISD energy %s', e)
    return e.real

def as_scanner(cc):
    '''Generating a scanner/solver for QCISD PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total QCISD energy.

    '''
    if isinstance(cc, lib.SinglePointScanner):
        return cc

    logger.info(cc, 'Set %s as a scanner', cc.__class__)

    class QCISD_Scanner(cc.__class__, lib.SinglePointScanner):
        def __init__(self, cc):
            self.__dict__.update(cc.__dict__)
            self._scf = cc._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            if self.t2 is not None:
                last_size = self.vector_size()
            else:
                last_size = 0

            self.reset(mol)

            mf_scanner = self._scf
            mf_scanner(mol)
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
            if last_size != self.vector_size():
                self.t1 = self.t2 = None
            self.kernel(self.t1, self.t2, **kwargs)
            return self.e_tot
    return QCISD_Scanner(cc)


class QCISD(CCSD):
    '''restricted QCISD
    '''

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('QCISD nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if (log.verbose >= logger.DEBUG1 and
            self.__class__ == QCISD):
            nocc = self.nocc
            nvir = self.nmo - self.nocc
            flops = _flops(nocc, nvir)
            log.debug1('total FLOPs %s', flops)
        return self

    energy = energy
    _add_vvvv = _add_vvvv
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None):
        return self.qcisd(t1, t2, eris)
    def qcisd(self, t1=None, t2=None, eris=None):
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    as_scanner = as_scanner

    def qcisd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import qcisd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return qcisd_t.kernel(self, eris, t1, t2, self.verbose)


RQCISD = QCISD


if __name__ == '__main__':
    from pyscf import scf

    mol = gto.Mole()
    #mol.atom = [
    #    [8 , (0. , 0.     , 0.)],
    #    [1 , (0. , -0.757 , 0.587)],
    #    [1 , (0. , 0.757  , 0.587)]]
    mol.atom = [['Ne', (0,0,0)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 7
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = QCISD(mf, frozen=1)
    ecc, t1, t2 = mycc.kernel()
    et = mycc.qcisd_t()
    print("QCISD(T) =", mycc.e_tot+et)

    mol = gto.Mole()
    mol.atom = """C  0.000  0.000  0.000
                  H  0.637  0.637  0.637
                  H -0.637 -0.637  0.637
                  H -0.637  0.637 -0.637
                  H  0.637 -0.637 -0.637"""
    mol.basis = 'cc-pvdz'
    mol.verbose = 7
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = QCISD(mf, frozen=1)
    ecc, t1, t2 = mycc.kernel()
    print(mycc.e_tot - -40.383989)
    et = mycc.qcisd_t()
    print(mycc.e_tot+et - -40.387679)
