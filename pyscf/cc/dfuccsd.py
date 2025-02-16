#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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


import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import df
from pyscf.cc import uccsd
from pyscf.cc import ccsd
from pyscf.cc import dfccsd
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

class UCCSD(uccsd.UCCSD):
    _keys = {'with_df'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return uccsd.UCCSD.reset(self, mol)

    def ao2mo(self, mo_coeff=None):
        return _make_df_eris(self, mo_coeff)

    def _add_vvvv(self, t1, t2, eris, out=None, with_ovvv=False, t2sym=None):
        assert (not self.direct)
        return uccsd.UCCSD._add_vvvv(self, t1, t2, eris, out, with_ovvv, t2sym)

    def _add_vvVV(self, t1, t2, eris, out=None):
        assert (not self.direct)
        return uccsd.UCCSD._add_vvVV(self, t1, t2, eris, out)

    to_gpu = lib.to_gpu

class _ChemistsERIs(uccsd._ChemistsERIs):
    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert (not direct)
        return dfccsd._contract_vvvv_t2(mycc, self.mol, self.vvL, self.vvL, t2, out, verbose)
    def _contract_VVVV_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert (not direct)
        return dfccsd._contract_vvvv_t2(mycc, self.mol, self.VVL, self.VVL, t2, out, verbose)
    def _contract_vvVV_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert (not direct)
        return dfccsd._contract_vvvv_t2(mycc, self.mol, self.vvL, self.VVL, t2, out, verbose)

def _cp(a):
    return np.asarray(a, order='C')

def _make_df_eris(mycc, mo_coeff=None):
    assert mycc._scf.istype('UHF')
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    moa, mob = eris.mo_coeff
    nocca, noccb = eris.nocc
    nao = moa.shape[0]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    nvira = nmoa - nocca
    nvirb = nmob - noccb
    nvira_pair = nvira * (nvira + 1) // 2
    nvirb_pair = nvirb * (nvirb + 1) // 2
    with_df = mycc.with_df
    naux = eris.naux = with_df.get_naoaux()

    # --- Three-center integrals
    # (L|aa)
    Loo = np.empty((naux, nocca, nocca))
    Lov = np.empty((naux, nocca, nvira))
    Lvo = np.empty((naux, nvira, nocca))
    # (L|bb)
    LOO = np.empty((naux, noccb, noccb))
    LOV = np.empty((naux, noccb, nvirb))
    LVO = np.empty((naux, nvirb, noccb))

    # --- Four-center integrals
    eris.feri = lib.H5TmpFile()
    # (aa|aa)
    eris.oooo = eris.feri.create_dataset('oooo', (nocca,nocca,nocca,nocca), 'f8')
    eris.oovv = eris.feri.create_dataset('oovv', (nocca,nocca,nvira,nvira), 'f8', chunks=(nocca,nocca,1,nvira))
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocca,nvira,nocca,nocca), 'f8', chunks=(nocca,1,nocca,nocca))
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocca,nvira,nvira,nocca), 'f8', chunks=(nocca,1,nvira,nocca))
    eris.ovov = eris.feri.create_dataset('ovov', (nocca,nvira,nocca,nvira), 'f8', chunks=(nocca,1,nocca,nvira))
    # (bb|bb)
    eris.OOOO = eris.feri.create_dataset('OOOO', (noccb,noccb,noccb,noccb), 'f8')
    eris.OOVV = eris.feri.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), 'f8', chunks=(noccb,noccb,1,nvirb))
    eris.OVOO = eris.feri.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), 'f8', chunks=(noccb,1,noccb,noccb))
    eris.OVVO = eris.feri.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), 'f8', chunks=(noccb,1,nvirb,noccb))
    eris.OVOV = eris.feri.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), 'f8', chunks=(noccb,1,noccb,nvirb))
    # (aa|bb)
    eris.ooOO = eris.feri.create_dataset('ooOO', (nocca,nocca,noccb,noccb), 'f8')
    eris.ooVV = eris.feri.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), 'f8', chunks=(nocca,nocca,1,nvirb))
    eris.ovOO = eris.feri.create_dataset('ovOO', (nocca,nvira,noccb,noccb), 'f8', chunks=(nocca,1,noccb,noccb))
    eris.ovVO = eris.feri.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), 'f8', chunks=(nocca,1,nvirb,noccb))
    eris.ovOV = eris.feri.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), 'f8', chunks=(nocca,1,noccb,nvirb))
    # (bb|aa)
    eris.OOvv = eris.feri.create_dataset('OOvv', (noccb,noccb,nvira,nvira), 'f8', chunks=(noccb,noccb,1,nvira))
    eris.OVoo = eris.feri.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), 'f8', chunks=(noccb,1,nocca,nocca))
    eris.OVvo = eris.feri.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), 'f8', chunks=(noccb,1,nvira,nocca))

    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
    chunks = (min(nvira_pair, int(4e8 / with_df.blockdim)), min(naux, with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvira_pair,naux), 'f8', chunks=chunks)
    chunks = (min(nvirb_pair, int(4e8 / with_df.blockdim)), min(naux, with_df.blockdim))
    eris.VVL = eris.feri.create_dataset('VVL', (nvirb_pair,naux), 'f8', chunks=chunks)

    # Transform three-center integrals to MO basis
    p1 = 0
    for eri1 in with_df.loop():
        eri1 = lib.unpack_tril(eri1).reshape(-1, nao, nao)
        # (L|aa)
        Lpq = lib.einsum('Lab,ap,bq->Lpq', eri1, moa, moa)
        p0, p1 = p1, p1 + Lpq.shape[0]
        blk = np.s_[p0:p1]
        Loo[blk] = Lpq[:, :nocca, :nocca]
        Lov[blk] = Lpq[:, :nocca, nocca:]
        Lvo[blk] = Lpq[:, nocca:, :nocca]
        eris.vvL[:, p0:p1] = lib.pack_tril(Lpq[:, nocca:, nocca:]).T
        # (L|bb)
        Lpq = None
        Lpq = lib.einsum('Lab,ap,bq->Lpq', eri1, mob, mob)
        LOO[blk] = Lpq[:, :noccb, :noccb]
        LOV[blk] = Lpq[:, :noccb, noccb:]
        LVO[blk] = Lpq[:, noccb:, :noccb]
        eris.VVL[:, p0:p1] = lib.pack_tril(Lpq[:, noccb:, noccb:]).T
        Lpq = None

    Loo = Loo.reshape(naux, nocca * nocca)
    Lov = Lov.reshape(naux, nocca * nvira)
    Lvo = Lvo.reshape(naux, nocca * nvira)
    LOO = LOO.reshape(naux, noccb * noccb)
    LOV = LOV.reshape(naux, noccb * nvirb)
    LVO = LVO.reshape(naux, noccb * nvirb)

    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocca,nocca,nocca,nocca)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocca,nvira,nocca,nocca)
    eris.ovvo[:] = lib.ddot(Lov.T, Lvo).reshape(nocca,nvira,nvira,nocca)
    eris.ovov[:] = lib.ddot(Lov.T, Lov).reshape(nocca,nvira,nocca,nvira)

    eris.OVoo[:] = lib.ddot(LOV.T, Loo).reshape(noccb,nvirb,nocca,nocca)
    eris.OVvo[:] = lib.ddot(LOV.T, Lvo).reshape(noccb,nvirb,nvira,nocca)

    Lvo = None

    eris.OOOO[:] = lib.ddot(LOO.T, LOO).reshape(noccb,noccb,noccb,noccb)
    eris.OVOO[:] = lib.ddot(LOV.T, LOO).reshape(noccb,nvirb,noccb,noccb)
    eris.OVVO[:] = lib.ddot(LOV.T, LVO).reshape(noccb,nvirb,nvirb,noccb)
    eris.OVOV[:] = lib.ddot(LOV.T, LOV).reshape(noccb,nvirb,noccb,nvirb)

    eris.ooOO[:] = lib.ddot(Loo.T, LOO).reshape(nocca,nocca,noccb,noccb)
    eris.ovOO[:] = lib.ddot(Lov.T, LOO).reshape(nocca,nvira,noccb,noccb)
    eris.ovVO[:] = lib.ddot(Lov.T, LVO).reshape(nocca,nvira,nvirb,noccb)
    eris.ovOV[:] = lib.ddot(Lov.T, LOV).reshape(nocca,nvira,noccb,nvirb)

    LVO = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)

    # eris.oovv
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocca**2*nvira_pair)/(nocca**2+naux)))
    oovv_tril = np.empty((nocca * nocca, nvira_pair))
    for p0, p1 in lib.prange(0, nvira_pair, blksize):
        oovv_tril[:, p0:p1] = lib.ddot(Loo.T, _cp(eris.vvL[p0:p1]).T)
    eris.oovv[:] = lib.unpack_tril(oovv_tril).reshape(nocca, nocca, nvira, nvira)
    oovv_tril = None

    # eris.ooVV
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocca**2*nvirb_pair)/(nocca**2+naux)))
    oovv_tril = np.empty((nocca * nocca, nvirb_pair))
    for p0, p1 in lib.prange(0, nvirb_pair, blksize):
        oovv_tril[:, p0:p1] = lib.ddot(Loo.T, _cp(eris.VVL[p0:p1]).T)
    eris.ooVV[:] = lib.unpack_tril(oovv_tril).reshape(nocca, nocca, nvirb, nvirb)
    oovv_tril = Loo = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)

    # eris.OOvv
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-noccb**2*nvira_pair)/(noccb**2+naux)))
    oovv_tril = np.empty((noccb * noccb, nvira_pair))
    for p0, p1 in lib.prange(0, nvira_pair, blksize):
        oovv_tril[:, p0:p1] = lib.ddot(LOO.T, _cp(eris.vvL[p0:p1]).T)
    eris.OOvv[:] = lib.unpack_tril(oovv_tril).reshape(noccb, noccb, nvira, nvira)
    oovv_tril = None

    # eris.OOVV
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-noccb**2*nvirb_pair)/(noccb**2+naux)))
    oovv_tril = np.empty((noccb * noccb, nvirb_pair))
    for p0, p1 in lib.prange(0, nvirb_pair, blksize):
        oovv_tril[:, p0:p1] = lib.ddot(LOO.T, _cp(eris.VVL[p0:p1]).T)
    eris.OOVV[:] = lib.unpack_tril(oovv_tril).reshape(noccb, noccb, nvirb, nvirb)
    oovv_tril = LOO = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)

    Lov = Lov.reshape(naux, nocca, nvira)
    LOV = LOV.reshape(naux, noccb, nvirb)

    # eris.ovvv
    vblk = max(nocca, int((max_memory*.15e6/8)/(nocca*nvira_pair)))
    vvblk = int(min(nvira_pair, 4e8/nocca, max(4, (max_memory*.8e6/8)/(vblk*nocca+naux))))
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocca, nvira, nvira_pair), 'f8', chunks=(nocca, 1, vvblk))
    for q0, q1 in lib.prange(0, nvira_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvira, vblk):
            tmpLov = Lov[:, :, p0:p1].reshape(naux, -1)
            eris.ovvv[:, p0:p1, q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(nocca, p1 - p0, q1 - q0)
        vvL = None

    # eris.ovVV
    vblk = max(nocca, int((max_memory*.15e6/8)/(nocca*nvirb_pair)))
    vvblk = int(min(nvirb_pair, 4e8/nocca, max(4, (max_memory*.8e6/8)/(vblk*nocca+naux))))
    eris.ovVV = eris.feri.create_dataset('ovVV', (nocca, nvira, nvirb_pair), 'f8', chunks=(nocca, 1, vvblk))
    for q0, q1 in lib.prange(0, nvirb_pair, vvblk):
        vvL = _cp(eris.VVL[q0:q1])
        for p0, p1 in lib.prange(0, nvira, vblk):
            tmpLov = Lov[:, :, p0:p1].reshape(naux, -1)
            eris.ovVV[:, p0:p1, q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(nocca, p1 - p0, q1 - q0)
        vvL = None
    Lov = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)

    # eris.OVvv
    vblk = max(noccb, int((max_memory*.15e6/8)/(noccb*nvira_pair)))
    vvblk = int(min(nvira_pair, 4e8/noccb, max(4, (max_memory*.8e6/8)/(vblk*noccb+naux))))
    eris.OVvv = eris.feri.create_dataset('OVvv', (noccb, nvirb, nvira_pair), 'f8', chunks=(noccb, 1, vvblk))
    for q0, q1 in lib.prange(0, nvira_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvirb, vblk):
            tmpLov = LOV[:, :, p0:p1].reshape(naux, -1)
            eris.OVvv[:, p0:p1, q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(noccb, p1 - p0, q1 - q0)
        vvL = None

    # eris.OVVV
    vblk = max(noccb, int((max_memory*.15e6/8)/(noccb*nvirb_pair)))
    vvblk = int(min(nvirb_pair, 4e8/noccb, max(4, (max_memory*.8e6/8)/(vblk*noccb+naux))))
    eris.OVVV = eris.feri.create_dataset('OVVV', (noccb, nvirb, nvirb_pair), 'f8', chunks=(noccb, 1, vvblk))
    for q0, q1 in lib.prange(0, nvirb_pair, vvblk):
        vvL = _cp(eris.VVL[q0:q1])
        for p0, p1 in lib.prange(0, nvirb, vblk):
            tmpLov = LOV[:, :, p0:p1].reshape(naux, -1)
            eris.OVVV[:, p0:p1, q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(noccb, p1 - p0, q1 - q0)
        vvL = None
    LOV = None

    log.timer('DF-UCCSD integral transformation', *cput0)
    return eris

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
    mf = scf.UHF(mol).density_fit('weigend').run()
    mycc = UCCSD(mf).run()
    print(mycc.e_corr - -0.2133709727796199)

    print("IP energies... (right eigenvector)")
    e,v = mycc.ipccsd(nroots=8)
    print(e)
    print(e[0] - 0.4336428577342009)
    print(e[2] - 0.5188000951518845)
    print(e[4] - 0.6785158684375829)

    print("EA energies... (right eigenvector)")
    e,v = mycc.eaccsd(nroots=8)
    print(e)
    print(e[0] - 0.1673013569134136)
    print(e[2] - 0.2399984284491973)
    print(e[4] - 0.5096018470162480)

    e, v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757563806054133)
    print(e[1] - 0.2757563806171079)
    print(e[2] - 0.2757563806183815)
    print(e[3] - 0.3006896721085447)
