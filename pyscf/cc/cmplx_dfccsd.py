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


import ctypes
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__
from pyscf.cc.rccsd import RCCSD
from pyscf.cc import df_rintermediates as imd

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def _contract_vvvv_t2(mycc, mol, Lvv, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
       vvvv = einsum('Lac,Lbd->acbd',Lvv,Lvv)
    '''
    dtype = mycc.mo_coeff.dtype
    dsize = mycc.mo_coeff.itemsize
    _dot = lib.dot
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(mol, verbose)

    naux = Lvv.shape[0]
    nocci, noccj, nvira, nvirb = t2.shape
    x2 = t2.reshape(-1,nvira * nvirb)
    nocc2 = nocci * noccj
    nvir2 = nvira * nvirb
    Ht2 = numpy.ndarray(t2.shape, buffer=out, dtype=dtype)
    Ht2[:] = 0
    mem_avail = mycc.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6/dsize
    Lblk = min(naux, int(0.4*mem_avail / nvir2))
    A = 2 * nvir2 + nocc2
    B = 2 * Lblk * nvirb
    C = 0.7 * (mem_avail - nvir2 * Lblk)
    delta = B**2 + 4 * A * C
    vblk = int((-B + numpy.sqrt(delta)) / (2 * A))
    vblk = min(nvirb, vblk)
    assert Lblk > 0, "enlarge mem"
    assert vblk > 0, "enlarge mem"

    for L0, L1 in lib.prange(0, naux, Lblk):
        Bvv = _cp(Lvv[L0:L1])
        for a0, a1 in lib.prange(0, nvirb, vblk):
            Ba = Bvv[:, a0*nvirb:a1*nvirb].T.copy()
            for b0, b1 in lib.prange(0, nvirb, vblk):
                Bb = Bvv[:, b0*nvirb:b1*nvirb].copy()
                eri = _dot(Ba, Bb)
                eri = eri.reshape((a1-a0),nvirb,(b1-b0),nvirb).transpose(0,2,1,3)
                eri = eri.reshape(-1,nvir2).T
                eri = _dot(x2, eri)
                Ht2[:, :, a0:a1, b0:b1] += eri.reshape(nocci, noccj, a1-a0, b1-b0)
                eri = None
                Bb = None
            Ba = None
        time0 = log.timer_debug1("vvvv_t2 Lblk/naux [%d:%d] / %d" %(L0,L1,naux), *time0)
    return Ht2


def _contract_ovvv_t2(mycc, eris, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ikcd,kcad->ia', t2, ovvv)
    '''
    dtype = mycc.mo_coeff.dtype
    dsize = mycc.mo_coeff.itemsize
    _dot = lib.dot

    nocci, noccj, nvira, nvirb = t2.shape
    nvir3 = nvira**3
    mem_avail = mycc.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    occblk = min(nocci, int(0.8*(mem_avail-nocci*nvira) / (2*nvir3 + nocci*nvira**2)))
    assert occblk > 0, "enlarge mem"
    ia = 0
    for i0, i1 in lib.prange(0, nocci, occblk):
        x2 = _cp(t2[:,i0:i1]).reshape(nocci, -1)
        ovvv = eris.get_ovvv(slice(i0,i1)).transpose(0,1,3,2)
        ovvv = ovvv.reshape(-1, nvira)
        ia += _dot(x2, ovvv)
        x2 = None
        ovvv = None
    return ia

def _contract_ovvv_t(mycc, eris, t1, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,kcbd->kbij', t2, ovvv)
       t1Ht2 = numpy.einsum('ka,kbij->ijab', t1, ovoo)
    '''
    dtype = mycc.mo_coeff.dtype
    dsize = mycc.mo_coeff.itemsize
    _dot = lib.dot

    nocci, noccj, nvira, nvirb = t2.shape
    nvir3 = nvira**3
    x2 = t2.reshape(-1,nvira * nvirb).T.copy()

    Ht2 = numpy.ndarray([nocci*nvira, nocci*noccj], buffer=out, dtype=dtype)
    Ht2[:] = 0

    mem_avail = mycc.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    occblk = min(nocci, int(0.8*mem_avail / (2*nvir3 + nvira*nocci**2 )))
    assert occblk > 0, "enlarge mem"
    for i0, i1 in lib.prange(0, nocci, occblk):
        ovvv = eris.get_ovvv(slice(i0,i1)).transpose(0,2,1,3)
        ovvv = ovvv.reshape(-1,nvira * nvirb)
        Ht2[i0*nvira:i1*nvira] = _dot(ovvv, x2)
        ovvv = None
    tHt = _dot(t1.T, Ht2.reshape(nocci,-1))
    tHt = tHt.reshape(nvira * nvirb, -1).T
    Ht2 = None
    return tHt.reshape(t2.shape)

def _contract_ovvv_t1(mycc, eris, t1, out=None, verbose=None):
    '''
    'iacb,jc->ijab' = 'iabc,jc->iabj' -> ijab
    '''
    dtype = mycc.mo_coeff.dtype
    dsize = mycc.mo_coeff.itemsize
    _dot = lib.dot
    nocci, nvira = t1.shape
    nvir3 = nvira**3
    Ht1 = numpy.ndarray([nocci, nvira, nvira, nocci], buffer=out, dtype=dtype)
    Ht1[:] = 0

    mem_avail = mycc.max_memory - lib.current_memory()[0]
    mem_avail *= 1e6 / dsize
    occblk = min(nocci, int(0.7*(mem_avail - nocci * nvira) / (2*nvir3+2*nocci*nvira**2) ))
    assert occblk > 0, "enlarge mem"
    for i0, i1 in lib.prange(0, nocci, occblk):
        ovvv = eris.get_ovvv(slice(i0,i1)).transpose(0,1,3,2).conj().reshape(-1, nvira)
        tmp = _dot(ovvv,t1.T)
        tmp = tmp.reshape(i1-i0, nvira, nvira, nocci)
        Ht1[i0:i1] = tmp
        tmp = None
        ovvv = None
    return Ht1.transpose(0,3,1,2)


class _ChemistsERIs(ccsd._ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert (not direct)
        return _contract_vvvv_t2(mycc, self.mol, self.Lvv, t2, out, verbose)

    def get_ovvv(self, *slices):
        return numpy.asarray(self.ovvv[slices])

def _make_df_eris(cc, mo_coeff=None):
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    dtype = eris.mo_coeff.dtype
    dsize = eris.mo_coeff.itemsize
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    eris.nvir = nvir
    nvir_pair = nvir*(nvir+1)//2
    with_df = cc.with_df
    naux = eris.naux = with_df.get_naoaux()
    nao = eris.mo_coeff.shape[0]
    nao_pair = nao**2
    _dot = lib.dot

    if cc._feri is None:
        if isinstance(cc._eris_to_save, str):
            eris.feri = h5py.File(cc._eris_to_save, 'w')
            print("save dferi to", cc._eris_to_save)
        else:
            eris.feri = lib.H5TmpFile()
        eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), dtype=dtype)
        eris.ovoo = eris.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), dtype=dtype, chunks=(nocc,1,nocc,nocc))
        eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), dtype=dtype, chunks=(nocc,1,nocc,nvir))
        eris.ovvo = eris.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), dtype=dtype, chunks=(nocc,1,nvir,nocc))
        eris.oovv = eris.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype=dtype, chunks=(nocc,nocc,1,nvir))
        # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
        blksize = min(naux, int(4e9/dsize/nvir**2))
        eris.Lvv = eris.feri.create_dataset('Lvv', (naux,nvir*nvir), dtype=dtype, chunks=(blksize, nvir*nvir))
        eris.ovL = eris.feri.create_dataset('ovL', (nocc*nvir,naux), dtype=dtype, chunks=(nocc*nvir, blksize))

        Loo = numpy.empty((naux,nocc,nocc), dtype=dtype)
        ovL = numpy.empty((naux,nocc,nvir), dtype=dtype)
        mo = numpy.asarray(eris.mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None

        mem_avail = cc.max_memory - lib.current_memory()[0]
        mem_auxblk = 3 * (nao_pair) * dsize/1e6
        aux_blksize = min(naux, max(1, int(numpy.floor(mem_avail*0.5 / mem_auxblk))))
        buf = numpy.empty(aux_blksize*nocc*nvir, dtype=dtype)

        if dtype == numpy.double:
            for Lpq in with_df.loop(blksize=aux_blksize):
                p0, p1 = p1, p1+Lpq.shape[0]
                out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
                out = out.reshape(p1-p0,nmo,nmo)
                Loo[p0:p1] = out[:,:nocc,:nocc]
                ovL[p0:p1] = out[:,:nocc,nocc:]
                eris.Lvv[p0:p1] = out[:,nocc:,nocc:].reshape(-1,nvir*nvir)
        else:
            kpti_kptj = numpy.zeros((2,3))
            if hasattr (with_df, 'kpts'):
                kpti_kptj[0] = with_df.kpts
                kpti_kptj[1] = with_df.kpts
                for LpqR, LpqI, sign in with_df.sr_loop(blksize=aux_blksize, kpti_kptj=kpti_kptj, compact=False):
                    Lpq = LpqR+LpqI*1j
                    LpqR = LpqI = None
                    p0, p1 = p1, p1+Lpq.shape[0]
                    out = _ao2mo.r_e2(Lpq, mo, ijslice, [], None, aosym='s1')
                    out = out.reshape(p1-p0,nmo,nmo)
                    Loo[p0:p1] = out[:,:nocc,:nocc]
                    ovL[p0:p1] = out[:,:nocc,nocc:]
                    eris.Lvv[p0:p1] = out[:,nocc:,nocc:].reshape(-1,nvir*nvir)
            else:
                for k, Lpq in enumerate(with_df.loop()):
                    Lpq = lib.unpack_tril(Lpq).astype(numpy.complex128)
                    out = _ao2mo.r_e2(Lpq, mo, ijslice, [], None)
                    p0, p1 = p1, p1 + Lpq.shape[0]
                    out = out.reshape(p1-p0,nmo,nmo)
                    Loo[p0:p1] = out[:,:nocc,:nocc]
                    ovL[p0:p1] = out[:,:nocc,nocc:]
                    eris.Lvv[p0:p1] = out[:,nocc:,nocc:].reshape(-1,nvir*nvir)

        Lpq = out = None
        Loo = Loo.reshape(naux,nocc**2)
        Lvo = ovL.transpose(0,2,1).conj()
        Lvo = Lvo.reshape(naux,nocc*nvir)
        ovL = ovL.reshape(naux,nocc*nvir)
        ovL = ovL.T
        eris.oooo[:] = _dot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
        eris.ovoo[:] = _dot(ovL, Loo).reshape(nocc,nvir,nocc,nocc)
        eris.ovov[:] = _dot(ovL, ovL.T).reshape(nocc,nvir,nocc,nvir)
        eris.ovvo[:] = _dot(ovL, Lvo).reshape(nocc,nvir,nvir,nocc)
        eris.ovL[:] = ovL
        Lvo = None

        eris.oovv[:] = 0 + 0j
        for p0, p1 in lib.prange(0, naux, blksize):
            Lvv = _cp(eris.Lvv[p0:p1])
            eris.oovv[:] += _dot(Loo[p0:p1].T, Lvv).reshape(nocc,nocc,nvir,nvir)
        Loo = Lvv = None

        eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), dtype=dtype,
                                            chunks=(1,nvir,nvir,nvir))
        eris.ovvv[:] = 0 + 0j
        mem_avail = cc.max_memory - lib.current_memory()[0]
        mem_avail *= 1e6/dsize
        oblksize = min(nocc, int(0.5*(mem_avail - blksize * nvir**2)/ (nvir**3+ nvir*blksize) ))
        assert oblksize > 0, "enlarge mem"
        for i0, i1 in lib.prange(0, nocc, oblksize):
            for p0, p1 in lib.prange(0, naux, blksize):
                Lvv = _cp(eris.Lvv[p0:p1])
                eris.ovvv[i0:i1] += _dot(_cp(ovL[i0*nvir:i1*nvir,p0:p1]), Lvv).reshape(i1-i0,nvir,nvir,nvir)
        eris.feri.flush()

    elif isinstance(cc._feri, str):
        print("load dferi from", cc._feri)
        eris.feri = h5py.File(cc._feri, 'r')
        eris.oooo = eris.feri['oooo']
        eris.ovoo = eris.feri['ovoo']
        eris.ovov = eris.feri['ovov']
        eris.ovvo = eris.feri['ovvo']
        eris.oovv = eris.feri['oovv']
        eris.ovvv = eris.feri['ovvv']
        eris.Lvv = eris.feri['Lvv']
        eris.ovL = eris.feri['ovL']
    return eris

def _cp(a):
    return numpy.array(a, copy=False, order='C')
   
def update_amps(cc, t1, t2, eris):
    import numpy as np
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    # rewrite ovvv, vvvv part
    assert (isinstance(eris, ccsd._ChemistsERIs))
    eris.max_memory = cc.max_memory
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo[numpy.diag_indices(nocc)] -= mo_e_o
    Fvv[numpy.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new  =-2*lib.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   lib.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -lib.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*lib.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -lib.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   lib.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2*lib.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -lib.einsum('kiac,kc->ia', eris.oovv, t1)
    tau = t2 + lib.einsum('ia,jb->ijab', t1, t1)
    tmp = 2 * tau.transpose(0,1,3,2) - tau
    t1new += _contract_ovvv_t2(cc, eris, tmp)
    tmp = None

    eris_ovoo = numpy.asarray(eris.ovoo, order='C')
    tmp = -2 * tau.transpose(1,0,2,3) + tau
    t1new += lib.einsum('kcli,klac->ia', eris_ovoo, tmp)
    tmp = None

    # T2 equation
    tmp = lib.einsum('kibc,jc->kbij', eris.oovv, t1)
    tmp = -lib.einsum('kbij,ka->ijab', tmp, t1)
    tmp += _contract_ovvv_t1(cc, eris, t1)

    t2new = tmp + tmp.transpose(1,0,3,2)
    tmp = None

    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    eris_ovoo = None

    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    tmp2 = None

    t2new -= tmp + tmp.transpose(1,0,3,2)
    tmp = None

    t2new += numpy.asarray(eris.ovov).conj().transpose(0,2,1,3)
    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo[numpy.diag_indices(nocc)] -= mo_e_o
    Lvv[numpy.diag_indices(nvir)] -= mo_e_v

    Woooo = imd.cc_Woooo(t1, t2, eris)
    t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
    Woooo = None
    ### new
    t2new += eris._contract_vvvv_t2(cc, tau)
    t2new -= _contract_ovvv_t(cc, eris, t1, tau)

    tmp = _contract_ovvv_t(cc, eris, t1, tau.transpose(0,1,3,2))
    t2new -= tmp.transpose(0,1,3,2)
    tmp = None

    tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = None

    tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = None

    tau = 2*t2 - t2.transpose(0,1,3,2)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    tmp = lib.einsum('akic,kjcb->ijab', Wvoov, tau)
    Wvoov = None
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = None

    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
    tmp += lib.einsum('akci,kjcb->ijab', Wvovo, t2)
    Wvovo = None
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = None

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

class cRCCSD(RCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])
        self._feri = None
        self._eris_to_save = None

    update_amps = update_amps

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return ccsd.RCCSD.reset(self, mol)

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)

