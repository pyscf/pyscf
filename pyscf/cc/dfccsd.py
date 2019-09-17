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
import ctypes
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

class RCCSD(ccsd.CCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=False, t2sym=None):
        assert(not self.direct)
        return ccsd.CCSD._add_vvvv(self, t1, t2, eris, out, with_ovvv, t2sym)


def _contract_vvvv_t2(mycc, mol, vvL, t2, out=None, verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    _dgemm = lib.numpy_helper._dgemm
    time0 = time.clock(), time.time()
    log = logger.new_logger(mol, verbose)

    naux = vvL.shape[-1]
    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    nvir2 = nvira * nvirb
    Ht2 = numpy.ndarray(x2.shape, buffer=out)
    Ht2[:] = 0

    max_memory = max(MEMORYMIN, mycc.max_memory - lib.current_memory()[0])
    def contract_blk_(eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        #:Ht2[:,j0:j1] += numpy.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm('N', 'N', nocc2, jc*nvirb, ic*nvirb,
               x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
               Ht2.reshape(-1,nvir2), 1, 1, i0*nvirb, 0, j0*nvirb)

        if i0 > j0:
            #:Ht2[:,i0:i1] += numpy.einsum('xef,abef->xab', x2[:,j0:j1], eri)
            _dgemm('N', 'T', nocc2, ic*nvirb, jc*nvirb,
                   x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
                   Ht2.reshape(-1,nvir2), 1, 1, j0*nvirb, 0, i0*nvirb)

#TODO: check if vvL can be entirely loaded into memory
    nvir_pair = nvirb * (nvirb+1) // 2
    dmax = numpy.sqrt(max_memory*.7e6/8/nvirb**2/2)
    dmax = int(min((nvira+3)//4, max(ccsd.BLKMIN, dmax)))
    vvblk = (max_memory*1e6/8 - dmax**2*(nvirb**2*1.5+naux))/naux
    vvblk = int(min((nvira+3)//4, max(ccsd.BLKMIN, vvblk/naux)))
    eribuf = numpy.empty((dmax,dmax,nvir_pair))
    loadbuf = numpy.empty((dmax,dmax,nvirb,nvirb))
    tril2sq = lib.square_mat_in_trilu_indices(nvira)

    for i0, i1 in lib.prange(0, nvira, dmax):
        off0 = i0*(i0+1)//2
        off1 = i1*(i1+1)//2
        vvL0 = _cp(vvL[off0:off1])
        for j0, j1 in lib.prange(0, i1, dmax):
            ijL = vvL0[tril2sq[i0:i1,j0:j1] - off0].reshape(-1,naux)
            eri = numpy.ndarray(((i1-i0)*(j1-j0),nvir_pair), buffer=eribuf)
            for p0, p1 in lib.prange(0, nvir_pair, vvblk):
                vvL1 = _cp(vvL[p0:p1])
                eri[:,p0:p1] = lib.ddot(ijL, vvL1.T)
                vvL1 = None

            tmp = numpy.ndarray((i1-i0,nvirb,j1-j0,nvirb), buffer=loadbuf)
            _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                   eri.ctypes.data_as(ctypes.c_void_p),
                                   (ctypes.c_int*4)(i0, i1, j0, j1),
                                   ctypes.c_int(nvirb))
            contract_blk_(tmp, i0, i1, j0, j1)
            time0 = log.timer_debug1('vvvv [%d:%d,%d:%d]'%(i0,i1,j0,j1), *time0)
    return Ht2.reshape(t2.shape)


class _ChemistsERIs(ccsd._ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert(not direct)
        return _contract_vvvv_t2(mycc, self.mol, self.vvL, t2, out, verbose)

def _make_df_eris(cc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nocc_pair = nocc*(nocc+1)//2
    nvir_pair = nvir*(nvir+1)//2
    with_df = cc.with_df
    naux = eris.naux = with_df.get_naoaux()

    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.oovv = eris.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
    chunks = (min(nvir_pair,int(4e8/with_df.blockdim)), min(naux,with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair,naux), 'f8', chunks=chunks)

    Loo = numpy.empty((naux,nocc,nocc))
    Lov = numpy.empty((naux,nocc,nvir))
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
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])
        eris.vvL[:,p0:p1] = Lvv.T
    Lpq = Lvv = None
    Loo = Loo.reshape(naux,nocc**2)
    Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov[:] = ovov
    eris.ovvo[:] = ovov.transpose(0,1,3,2)
    ovov = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocc**2*nvir_pair)/(nocc**2+naux)))
    oovv_tril = numpy.empty((nocc*nocc,nvir_pair))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        oovv_tril[:,p0:p1] = lib.ddot(Loo.T, _cp(eris.vvL[p0:p1]).T)
    eris.oovv[:] = lib.unpack_tril(oovv_tril).reshape(nocc,nocc,nvir,nvir)
    oovv_tril = Loo = None

    Lov = Lov.reshape(naux,nocc,nvir)
    vblk = max(nocc, int((max_memory*.15e6/8)/(nocc*nvir_pair)))
    vvblk = int(min(nvir_pair, 4e8/nocc, max(4, (max_memory*.8e6/8)/(vblk*nocc+naux))))
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8',
                                         chunks=(nocc,1,vvblk))
    for q0, q1 in lib.prange(0, nvir_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvir, vblk):
            tmpLov = _cp(Lov[:,:,p0:p1]).reshape(naux,-1)
            eris.ovvv[:,p0:p1,q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(nocc,p1-p0,q1-q0)
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
