#!/usr/bin/env python

import time
from functools import reduce
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.cc import rccsd
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
_dgemm = lib.numpy_helper._dgemm

class RCCSD(ccsd.CCSD):
    def ao2mo(self, mo_coeff=None):
        return _make_eris_df(self, mo_coeff)

    def _add_vvvv_tril(self, t1, t2, eris, out=None, with_ovvv=False):
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        assert(not self.direct)
        time0 = time.clock(), time.time()
        nocc, nvir = t1.shape
        t2new_tril = numpy.ndarray((nocc*(nocc+1)//2,nvir,nvir), buffer=out)
        t2new_tril[:] = 0

        def contract_rec_(t2new_tril, tau, eri, i0, i1, j0, j1):
            nao = tau.shape[-1]
            ic = i1 - i0
            jc = j1 - j0
            #: t2tril[:,j0:j1] += numpy.einsum('xcd,cdab->xab', tau[:,i0:i1], eri)
            _dgemm('N', 'N', nocc*(nocc+1)//2, jc*nao, ic*nao,
                   tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
                   t2new_tril.reshape(-1,nao*nao), 1, 1, i0*nao, 0, j0*nao)

            #: t2tril[:,i0:i1] += numpy.einsum('xcd,abcd->xab', tau[:,j0:j1], eri)
            _dgemm('N', 'T', nocc*(nocc+1)//2, ic*nao, jc*nao,
                   tau.reshape(-1,nao*nao), eri.reshape(-1,jc*nao),
                   t2new_tril.reshape(-1,nao*nao), 1, 1, j0*nao, 0, i0*nao)

        def contract_tril_(t2new_tril, tau, eri, a0, a):
            nvir = tau.shape[-1]
            #: t2new[i,:i+1, a] += numpy.einsum('xcd,cdb->xb', tau[:,a0:a+1], eri)
            _dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (a+1-a0)*nvir,
                   tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
                   t2new_tril.reshape(-1,nvir*nvir), 1, 1, a0*nvir, 0, a*nvir)

            #: t2new[i,:i+1,a0:a] += numpy.einsum('xd,abd->xab', tau[:,a], eri[:a])
            if a > a0:
                _dgemm('N', 'T', nocc*(nocc+1)//2, (a-a0)*nvir, nvir,
                       tau.reshape(-1,nvir*nvir), eri.reshape(-1,nvir),
                       t2new_tril.reshape(-1,nvir*nvir), 1, 1, a*nvir, 0, a0*nvir)

        nvir_pair = nvir * (nvir+1) // 2
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        naux = eris.naux
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p0 = 0
        for i in range(nocc):
            tau[p0:p0+i+1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p0+i+1] += t2[i,:i+1]
            p0 += i + 1
        time0 = logger.timer_debug1(self, 'vvvv-tau', *time0)

#TODO: check if vvL can be entirely load into memory
        max_memory = max(0, self.max_memory - lib.current_memory()[0])
        dmax = min(nvir, max(ccsd.BLKMIN, numpy.sqrt(max_memory*.8e6/8/nvir**2/2)))
        vvblk = min(nvir, max(ccsd.BLKMIN, (max_memory*1e6/8 - dmax**2*(nvir**2*1.5+naux))/naux))
        dmax = int(dmax)
        vvblk = int(vvblk)
        eribuf = numpy.empty((dmax,dmax,nvir_pair))
        loadbuf = numpy.empty((dmax,dmax,nvir,nvir))

        for i0, i1 in lib.prange(0, nvir, dmax):
            di = i1 - i0
            for j0, j1 in lib.prange(0, i0, dmax):
                dj = j1 - j0

                ijL = numpy.empty((di,dj,naux))
                for i in range(i0, i1):
                    ioff = i*(i+1)//2
                    ijL[i-i0] = eris.vvL[ioff+j0:ioff+j1]
                ijL = ijL.reshape(-1,naux)
                eri = numpy.ndarray(((i1-i0)*(j1-j0),nvir_pair), buffer=eribuf)
                for p0, p1 in lib.prange(0, nvir_pair, vvblk):
                    vvL = _cp(eris.vvL[p0:p1])
                    eri[:,p0:p1] = lib.ddot(ijL, vvL.T)
                    vvL = None

                tmp = numpy.ndarray((i1-i0,nvir,j1-j0,nvir), buffer=loadbuf)
                _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                       eri.ctypes.data_as(ctypes.c_void_p),
                                       (ctypes.c_int*4)(i0, i1, j0, j1),
                                       ctypes.c_int(nvir))
                contract_rec_(t2new_tril, tau, tmp, i0, i1, j0, j1)
                time0 = logger.timer_debug1(self, 'vvvv [%d:%d,%d:%d]' %
                                            (i0,i1,j0,j1), *time0)

            ijL = []
            for i in range(i0, i1):
                ioff = i*(i+1)//2
                ijL.append(eris.vvL[ioff+i0:ioff+i+1])
            ijL = numpy.vstack(ijL).reshape(-1,naux)
            eri = numpy.ndarray((di*(di+1)//2,nvir_pair), buffer=eribuf)
            for p0, p1 in lib.prange(0, nvir_pair, vvblk):
                vvL = _cp(eris.vvL[p0:p1])
                eri[:,p0:p1] = lib.ddot(ijL, vvL.T)
                vvL = None
            for i in range(di):
                p0, p1 = i*(i+1)//2, (i+1)*(i+2)//2
                tmp = lib.unpack_tril(eri[p0:p1], out=loadbuf)
                contract_tril_(t2new_tril, tau, tmp, i0, i0+i)
            time0 = logger.timer_debug1(self, 'vvvv [%d:%d,%d:%d]' %
                                        (i0,i1,i0,i1), *time0)
        eribuf = loadbuf = eri = tmp = None
        return t2new_tril

class _ERIs:
    def __init__(self, cc):
        self.mo_coeff = None
        self.fock = None
        self.nocc = cc.nocc

        self.oooo = None
        self.vooo = None
        self.voov = None
        self.vvoo = None
        self.vovv = None
        self.vvvv = None

def _make_eris_df(cc, mo_coeff=None):
    eris = _ERIs(cc)
    if mo_coeff is None:
        eris.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
    else:
        eris.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
    eris.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nocc_pair = nocc*(nocc+1)//2
    nvir_pair = nvir*(nvir+1)//2
    with_df = cc._scf.with_df
    naux = eris.naux = with_df.get_naoaux()

    eris.feri = lib.H5TmpFile()
    chunks = (nvir_pair, min(naux,with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair,naux), 'f8', chunks=chunks)

    Loo = numpy.empty((naux,nocc,nocc))
    Lvo = numpy.empty((naux,nvir,nocc))
    fswap = lib.H5TmpFile()
    mo = numpy.asarray(eris.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lvo[p0:p1] = Lpq[:,nocc:,:nocc]
        Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])
        eris.vvL[:,p0:p1] = Lvv.T
    Lpq = Lvv = None
    Loo = Loo.reshape(naux,nocc**2)
    Lvo = Lvo.reshape(naux,nvir*nocc)

    eris.feri['oooo'] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.feri['vooo'] = lib.ddot(Lvo.T, Loo).reshape(nvir,nocc,nocc,nocc)
    vovo = lib.ddot(Lvo.T, Lvo).reshape(nvir,nocc,nvir,nocc)
    eris.feri['vovo'] = vovo
    eris.feri['voov'] = vovo.transpose(0,1,3,2)
    vovo = None
    eris.oooo = eris.feri['oooo']
    eris.vooo = eris.feri['vooo']
    eris.vovo = eris.feri['vovo']
    eris.voov = eris.feri['voov']

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocc**2*nvir_pair)/(nocc**2+naux)))
    vvoo_tril = numpy.empty((nvir_pair,nocc,nocc))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        vvL = _cp(eris.vvL[p0:p1])
        vvoo_tril[p0:p1] = lib.ddot(vvL, Loo).reshape(p1-p0,nocc,nocc)
    vvoo = numpy.empty((nvir,nvir,nocc,nocc))
    idx,idy = numpy.tril_indices(nvir)
    vvoo[idx,idy] = vvoo_tril
    vvoo[idy,idx] = vvoo_tril.transpose(0,2,1)
    eris.feri['vvoo'] = vvoo
    eris.vvoo = eris.feri['vvoo']
    vvoo = vvoo_tril = vvL = Loo = None

    Lvo = Lvo.reshape(naux,nvir,nocc)
    vblk = max(nocc, int((max_memory*.15e6/8)/(nocc*nvir_pair)))
    vvblk = min(nvir_pair, max(4, int((max_memory*.8e6/8)/(vblk*nocc+naux))))
    eris.vovv = eris.feri.create_dataset('vovv', (nvir,nocc,nvir_pair), 'f8',
                                         chunks=(nvir,nocc,vvblk))
    for q0, q1 in lib.prange(0, nvir_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvir, vblk):
            tmpLvo = _cp(Lvo[:,p0:p1]).reshape(naux,-1)
            eris.vovv[p0:p1,:,q0:q1] = lib.ddot(tmpLvo.T, vvL.T).reshape(p1-p0,nocc,q1-q0)
        vvL = None
    return eris

def _cp(a):
    return numpy.array(a, copy=False, order='C')

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).density_fit('weigend').run()
    mycc = RCCSD(mf).run()
    print(mycc.e_corr - -0.21337100025961622)

    print("IP energies... (right eigenvector)")
    part = None
    e,v = mycc.ipccsd(nroots=3, partition=part)
    print(e)
    print(e[0] - 0.43364287418576897)
    print(e[1] - 0.5188001071775572 )
    print(e[2] - 0.67851590275796392)

    print("IP energies... (left eigenvector)")
    e,lv = mycc.ipccsd(nroots=3,left=True,partition=part)
    print(e)
    print(e[0] - 0.43364286531878882)
    print(e[1] - 0.51879999865136994)
    print(e[2] - 0.67851587320495355)

    print("EA energies... (right eigenvector)")
    e,v = mycc.eaccsd(nroots=3, partition=part)
    print(e)
    print(e[0] - 0.16730125785810035)
    print(e[1] - 0.23999823045518162)
    print(e[2] - 0.50960183439619933)

    print("EA energies... (left eigenvector)")
    e,lv = mycc.eaccsd(nroots=3, left=True, partition=part)
    print(e)
    print(e[0] - 0.16730137808538076)
    print(e[1] - 0.23999845448276602)
    print(e[2] - 0.50960182130968001)

    e, v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.27575637238275519)
    print(e[1] - 0.27575637238275519)
    print(e[2] - 0.27575637238275519)
    print(e[3] - 0.30068967373840394)
