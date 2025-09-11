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

'''
density fitting MP2,  3-center integrals incore.
'''

import h5py
import ctypes
import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.mp import mp2
from pyscf.mp.mp2 import make_rdm1, make_rdm2, _mo_splitter
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_with_t2', True)
THRESH_LINDEP = getattr(__config__, 'mp_dfmp2_thresh_lindep', 1e-10)

libmp = lib.load_library('libmp')


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):

    log = logger.new_logger(mp, verbose)

    if eris is None: eris = mp.ao2mo()

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    occ_energy, vir_energy = mp.split_mo_energy()[1:3]
    moevv = np.asarray(vir_energy[:,None] + vir_energy, order='C')

    mem_avail = mp.max_memory - lib.current_memory()[0]

    if with_t2:
        t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=eris.dtype)
        t2_ptr = t2.ctypes.data_as(ctypes.c_void_p)
        mem_avail -= t2.size * eris.dsize / 1e6
    else:
        t2 = None
        t2_ptr = lib.c_null_ptr()

    if mem_avail < 0:
        log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
        raise MemoryError

    emp2_ss = emp2_os = 0

    drv = libmp.MP2_contract_d

    # determine occ blksize
    if isinstance(eris.ovL, np.ndarray):    # incore ovL
        occ_blksize = nocc
    else:   # outcore ovL
        # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
        occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvir**2)/(2*naux*nvir)))
        occ_blksize = min(nocc, max(1, occ_blksize))

    log.debug('occ blksize for %s loop: %d/%d', mp.__class__.__name__, occ_blksize, nocc)

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc,occ_blksize)):
        nocci = i1-i0
        iaL = eris.get_occ_blk(i0,i1)
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc,occ_blksize)):
            noccj = j1-j0
            if ibatch == jbatch:
                jbL = iaL
            else:
                jbL = eris.get_occ_blk(j0,j1)

            ed = np.zeros(1, dtype=np.float64)
            ex = np.zeros(1, dtype=np.float64)
            moeoo_block = np.asarray(
                occ_energy[i0:i1,None] + occ_energy[j0:j1], order='C')
            s2symm = 1
            t2_ex = 0
            drv(
                ed.ctypes.data_as(ctypes.c_void_p),
                ex.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(s2symm),
                iaL.ctypes.data_as(ctypes.c_void_p),
                jbL.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(i0), ctypes.c_int(j0),
                ctypes.c_int(nocci), ctypes.c_int(noccj),
                ctypes.c_int(nocc), ctypes.c_int(nvir), ctypes.c_int(naux),
                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                moevv.ctypes.data_as(ctypes.c_void_p),
                t2_ptr, ctypes.c_int(t2_ex)
            )
            emp2_ss += ed[0] + ex[0]
            emp2_os += ed[0]

            jbL = None
        iaL = None

        cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i0,i1,nocc), *cput1)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFRMP2(mp2.RMP2):
    _keys = {'with_df', 'mo_energy', 'force_outcore'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
        mp2.MP2Base.__init__(self, mf, frozen, mo_coeff, mo_occ)

        self.mo_energy = get_mo_energy(mf, self.mo_coeff, self.mo_occ, mo_energy)

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        # DEBUG:
        self.force_outcore = False

    kernel = mp2.RMP2.kernel

    def split_mo_coeff(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        masks = _mo_splitter(self)
        return [mo_coeff[:,m] for m in masks]

    def split_mo_energy(self, mo_energy=None):
        if mo_energy is None: mo_energy = self.mo_energy
        masks = _mo_splitter(self)
        return [mo_energy[m] for m in masks]

    def split_mo_occ(self, mo_occ=None):
        if mo_occ is None: mo_occ = self.mo_occ
        masks = _mo_splitter(self)
        return [mo_occ[m] for m in masks]

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return mp2.MP2.reset(self, mol)

    def ao2mo(self, mo_coeff=None, ovL=None, ovL_to_save=None):
        return _make_df_eris(self, mo_coeff, ovL, ovL_to_save)

    def make_rdm1(self, t2=None, ao_repr=False, with_frozen=True):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm1(self, t2, ao_repr=ao_repr, with_frozen=with_frozen)

    def make_rdm2(self, t2=None, ao_repr=False):
        if t2 is None:
            t2 = self.t2
        assert t2 is not None
        return make_rdm2(self, t2, ao_repr=ao_repr)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    Gradients = NotImplemented

MP2 = DFMP2 = DFRMP2

from pyscf import scf
scf.hf.RHF.DFMP2 = lib.class_as_method(DFMP2)
scf.rohf.ROHF.DFMP2 = NotImplemented

del (WITH_T2)


def get_mo_energy(mf, mo_coeff, mo_occ, mo_energy=None):
    if mo_energy is not None:
        return mo_energy
    if mo_coeff is mf.mo_coeff and mf.converged:
        return mf.mo_energy

    # rebuild fock
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(dm=dm)
    fockao = mf.get_fock(vhf=vhf, dm=dm)
    if np.asarray(dm).ndim == 2:    # RHF
        return np.diag(reduce(lib.dot, (mo_coeff.T.conj(), fockao, mo_coeff))).real
    else:
        return [np.diag(reduce(lib.dot, (mo_coeff[i].T.conj(), fockao[i], mo_coeff[i]))).real
                for i in [0,1]]


def _make_df_eris(mp, mo_coeff=None, ovL=None, ovL_to_save=None, verbose=None):
    log = logger.new_logger(mp, verbose)

    with_df = getattr(mp, 'with_df', None)
    assert( with_df is not None )

    if with_df._cderi is None:
        if getattr(with_df, 'cell', None) is not None:  # PBC
            log.warn('PBC mean-field does not support direct DFMP2. Caching AO 3c integrals now.')
            with_df.build()
            naux = with_df.get_naoaux()
        else:
            log.debug('Caching ovL-type integrals directly')
            if with_df.auxmol is None:
                with_df.auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
            naux = with_df.auxmol.nao_nr()
    else:
        log.debug('Caching ovL-type integrals by transforming saved AO 3c integrals.')
        naux = with_df.get_naoaux()

    if mo_coeff is None: mo_coeff = mp.mo_coeff
    occ_coeff, vir_coeff = mp.split_mo_coeff()[1:3]

    # determine incore or outcore
    nocc = occ_coeff.shape[1]
    nvir = vir_coeff.shape[1]

    if ovL is not None:
        if isinstance(ovL, np.ndarray):
            outcore = False
        elif isinstance(ovL, str):
            outcore = True
        else:
            log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                      type(ovL))
            raise TypeError
    else:
        mem_now = mp.max_memory - lib.current_memory()[0]
        mem_df = nocc*nvir*naux*8/1024**2.
        log.debug('ao2mo est mem= %.2f MB  avail mem= %.2f MB', mem_df, mem_now)
        # DEBUG:
        if mp.force_outcore:
            outcore = True
        else:
            outcore = (ovL_to_save is not None) or (mem_now*0.8 < mem_df)
    log.debug('ovL-type integrals are cached %s', 'outcore' if outcore else 'incore')

    if outcore:
        eris = _DFOUTCOREERIS(with_df, occ_coeff, vir_coeff, mp.max_memory,
                              ovL=ovL, ovL_to_save=ovL_to_save,
                              verbose=log.verbose, stdout=log.stdout)
    else:
        eris = _DFINCOREERIS(with_df, occ_coeff, vir_coeff, mp.max_memory,
                             ovL=ovL,
                             verbose=log.verbose, stdout=log.stdout)
    eris.build()

    return eris


class _DFINCOREERIS:
    def __init__(self, with_df, occ_coeff, vir_coeff, max_memory, ovL=None,
                 verbose=None, stdout=None):
        self.with_df = with_df
        self.occ_coeff = occ_coeff
        self.vir_coeff = vir_coeff

        self.max_memory = max_memory
        self.verbose = verbose
        self.stdout = stdout

        self.dtype = self.occ_coeff.dtype
        assert( self.dtype == np.float64 )  # FIXME: support complex
        self.dsize = 8

        self.ovL = ovL

    @property
    def nocc(self):
        return self.occ_coeff.shape[1]
    @property
    def nvir(self):
        return self.vir_coeff.shape[1]
    @property
    def naux(self):
        return self.ovL.shape[-1]

    def build(self):
        log = logger.new_logger(self)
        if self.ovL is None:
            if self.with_df._cderi is None:
                self.ovL = _init_mp_df_eris_direct(self.with_df, self.occ_coeff, self.vir_coeff,
                                                   self.max_memory, log=log)
            else:
                self.ovL = _init_mp_df_eris(self.with_df, self.occ_coeff, self.vir_coeff,
                                            self.max_memory, log=log)

    def get_occ_blk(self, i0,i1):
        nvir, naux = self.nvir, self.naux
        return np.asarray(self.ovL[i0*nvir:i1*nvir], order='C').reshape(i1-i0,nvir,naux)
    def get_ov_blk(self, ia0,ia1):
        return np.asarray(self.ovL[ia0:ia1], order='C')


class _DFOUTCOREERIS(_DFINCOREERIS):
    def __init__(self, with_df, occ_coeff, vir_coeff, max_memory, ovL=None, ovL_to_save=None,
                 verbose=None, stdout=None):
        _DFINCOREERIS.__init__(self, with_df, occ_coeff, vir_coeff, max_memory, None,
                               verbose, stdout)

        self._ovL = ovL
        self._ovL_to_save = ovL_to_save

    def build(self):
        log = logger.new_logger(self)
        with_df = self.with_df
        if self._ovL is None:
            if isinstance(self._ovL_to_save, str):
                self.feri = lib.H5FileWrap(self._ovL_to_save, 'w')
            else:
                self.feri = lib.H5TmpFile()
            log.debug('ovL is saved to %s', self.feri.filename)
            if with_df._cderi is None:
                _init_mp_df_eris_direct(with_df, self.occ_coeff, self.vir_coeff, self.max_memory,
                                        h5obj=self.feri, log=log)
            else:
                _init_mp_df_eris(with_df, self.occ_coeff, self.vir_coeff, self.max_memory,
                                 h5obj=self.feri, log=log)
            self.ovL = self.feri['ovL']
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.debug('ovL is read from %s', self.feri.filename)
            assert( 'ovL' in self.feri )
            self.ovL = self.feri['ovL']
        else:
            raise RuntimeError

def _init_mp_df_eris(with_df, occ_coeff, vir_coeff, max_memory, h5obj=None, log=None):
    if log is None: log = logger.new_logger(with_df)

    nao,nocc = occ_coeff.shape
    nvir = vir_coeff.shape[1]
    nmo = nocc + nvir
    nao_pair = nao**2
    naux = with_df.get_naoaux()

    dtype = occ_coeff.dtype
    assert( dtype == np.float64 )
    dsize = 8

    mo = np.asarray(np.hstack((occ_coeff,vir_coeff)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    if h5obj is None:   # incore
        ovL = np.empty((nocc*nvir,naux), dtype=dtype)
    else:
        ovL_shape = (nocc*nvir,naux)
        ovL = h5obj.create_dataset('ovL', ovL_shape, dtype=dtype, chunks=(1,*ovL_shape[1:]))

    mem_avail = max_memory - lib.current_memory()[0]

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao_pair+nocc*nvir) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.5 / mem_auxblk))))
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize*nocc*nvir, dtype=dtype)
        ijslice = (0,nocc,nocc,nmo)

        p1 = 0
        for Lpq in with_df.loop(blksize=aux_blksize):
            p0, p1 = p1, p1+Lpq.shape[0]
            out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
            ovL[:,p0:p1] = out.T
            Lpq = out = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux*nvir * dsize/1e6
        occ_blksize = min(nocc, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        mem_auxblk = (occ_blksize*nvir+nao_pair) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, nocc)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux*occ_blksize*nvir, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvir, dtype=dtype)

        for i0,i1 in lib.prange(0,nocc,occ_blksize):
            nocci = i1-i0
            ijslice = (i0,i1,nocc,nmo)
            p1 = 0
            OvL = np.ndarray((nocci*nvir,naux), dtype=dtype, buffer=buf)
            for Lpq in with_df.loop(blksize=aux_blksize):
                p0, p1 = p1, p1+Lpq.shape[0]
                out = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf2)
                OvL[:,p0:p1] = out.T
                Lpq = out = None
            ovL[i0*nvir:i1*nvir] = OvL  # this avoids slow operations like ovL[i0:i1,:,p0:p1] = ...
            OvL = None
        buf = buf2 = None

    return ovL


def _init_mp_df_eris_direct(with_df, occ_coeff, vir_coeff, max_memory, h5obj=None, log=None):
    from pyscf import gto
    from pyscf.df.incore import fill_2c2e
    from pyscf.ao2mo.outcore import balance_partition

    if log is None: log = logger.new_logger(with_df)

    mol = with_df.mol
    auxmol = with_df.auxmol
    nbas = mol.nbas
    nao,nocc = occ_coeff.shape
    nvir = vir_coeff.shape[1]
    nmo = nocc + nvir
    nao_pair = nao*(nao+1)//2
    naoaux = auxmol.nao_nr()

    dtype = occ_coeff.dtype
    assert( dtype == np.float64 )
    dsize = 8

    mo = np.asarray(np.hstack((occ_coeff,vir_coeff)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    tspans = np.zeros((5,2))
    tnames = ['j2c', 'j3c', 'xform', 'save', 'fit']
    tick = (logger.process_clock(), logger.perf_counter())
    # precompute for fitting
    j2c = fill_2c2e(mol, auxmol)
    try:
        m2c = scipy.linalg.cholesky(j2c, lower=True)
        tag = 'cd'
    except scipy.linalg.LinAlgError:
        e, u = np.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        keep = abs(e) > THRESH_LINDEP
        log.debug('cond(j2c) = %g', cond)
        log.debug('keep %d/%d cderi vectors', np.count_nonzero(keep), keep.size)
        e = e[keep]
        u = u[:,keep]
        m2c = lib.dot(u*e**-0.5, u.T.conj())
        tag = 'eig'
    j2c = None
    naux = m2c.shape[1]
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[0] += np.asarray(tock) - np.asarray(tick)

    mem_avail = max_memory - lib.current_memory()[0]

    incore = h5obj is None
    if incore:
        ovL = np.empty((nocc*nvir,naoaux), dtype=dtype)
        mem_avail -= ovL.size * dsize / 1e6
    else:
        ovL_shape = (nocc*nvir,naux)
        ovL = h5obj.create_dataset('ovL', ovL_shape, dtype=dtype, chunks=(1,*ovL_shape[1:]))
        h5tmp = lib.H5TmpFile()
        Lov0_shape = (naoaux,nocc*nvir)
        Lov0 = h5tmp.create_dataset('Lov0', Lov0_shape, dtype=dtype, chunks=(1,*Lov0_shape[1:]))

    # buffer
    mem_blk = nao_pair*2 * dsize / 1e6
    aux_blksize = max(1, min(naoaux, int(np.floor(mem_avail*0.7 / mem_blk))))
    auxshl_range = balance_partition(auxmol.ao_loc, aux_blksize)
    auxlen = max([x[2] for x in auxshl_range])
    log.info('mem_avail = %.2f  mem_blk = %.2f  auxlen = %d', mem_avail, mem_blk, auxlen)
    buf0 = np.empty(auxlen*nao_pair, dtype=dtype)
    buf0T = np.empty(auxlen*nao_pair, dtype=dtype)

    # precompute for j3c
    comp = 1
    aosym = 's2ij'
    int3c = gto.moleintor.ascint3(mol._add_suffix('int3c2e'))
    atm_f, bas_f, env_f = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                            auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc_f = gto.moleintor.make_loc(bas_f, int3c)
    cintopt = gto.moleintor.make_cintopt(atm_f, bas_f, env_f, int3c)

    def calc_j3c_ao(kshl0, kshl1):
        shls_slice = (0, nbas, 0, nbas, nbas+kshl0, nbas+kshl1)
        pqL = gto.moleintor.getints3c(int3c, atm_f, bas_f, env_f, shls_slice, comp,
                                      aosym, ao_loc_f, cintopt, out=buf0)
        Lpq = lib.transpose(pqL, out=buf0T)
        pqL = None
        return Lpq

    # transform
    k1 = 0
    for auxshl_rg in auxshl_range:
        kshl0, kshl1, dk = auxshl_rg
        k0, k1 = k1, k1+dk
        log.debug('kshl = [%d:%d/%d]  [%d:%d/%d]', kshl0, kshl1, auxmol.nbas, k0, k1, naoaux)
        tick = (logger.process_clock(), logger.perf_counter())
        lpq = calc_j3c_ao(kshl0, kshl1)
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[1] += np.asarray(tock) - np.asarray(tick)
        lov = _ao2mo.nr_e2(lpq, mo, ijslice, aosym='s2', out=buf0)
        tick = (logger.process_clock(), logger.perf_counter())
        tspans[2] += np.asarray(tick) - np.asarray(tock)
        if incore:
            ovl = lib.transpose(lov, out=buf0T)
            ovL[:,k0:k1] = ovl
            ovl = None
        else:
            Lov0[k0:k1] = lov
        lpq = lov = None
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[3] += np.asarray(tock) - np.asarray(tick)
    buf0 = buf0T = None

    tick = (logger.process_clock(), logger.perf_counter())
    # fit
    if tag == 'cd': drv = getattr(libmp, 'trisolve_parallel_grp', None)
    if incore:
        if tag == 'cd':
            if drv is None:
                ovL = scipy.linalg.solve_triangular(m2c, ovL.T, lower=True,
                                                    overwrite_b=True, check_finite=False).T
            else:
                assert m2c.flags.f_contiguous
                grpfac = 10
                drv(
                    m2c.ctypes.data_as(ctypes.c_void_p),
                    ovL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nocc*nvir),
                    ctypes.c_int(grpfac)
                )
        else:
            nvxao = nvir*naoaux
            nvx = nvir*naux
            mem_blk = nvx * dsize/1e6
            occ_blksize = max(1, min(nocc, int(np.floor(mem_avail*0.5/mem_blk))))
            buf = np.empty(occ_blksize*nvx, dtype=dtype)
            ovL = ovL.reshape(-1)
            for i0,i1 in lib.prange(0,nocc,occ_blksize):
                nocci = i1-i0
                out = np.ndarray((nocci*nvir,naux), dtype=dtype, buffer=buf)
                lib.dot(ovL[i0*nvxao:i1*nvxao].reshape(nocci*nvir,naoaux), m2c, c=out)
                ovL[i0*nvx:i1*nvx] = out.reshape(-1)
            ovL = ovL[:nocc*nvx].reshape(nocc*nvir,naux)
            buf = None
    else:
        nvxao = nvir*naoaux
        nvx = nvir*naux
        mem_blk = nvxao * dsize / 1e6
        occ_blksize = max(1, min(nocc, int(np.floor(mem_avail*0.4/mem_blk))))
        for i0,i1 in lib.prange(0, nocc, occ_blksize):
            nocci = i1-i0
            ivL = np.asarray(Lov0[:,i0*nvir:i1*nvir].T, order='C')
            if tag == 'cd':
                if drv is None:
                    ivL = scipy.linalg.solve_triangular(m2c, ivL.T, lower=True,
                                                        overwrite_b=True, check_finite=False).T
                else:
                    assert m2c.flags.f_contiguous
                    grpfac = 10
                    drv(
                        m2c.ctypes.data_as(ctypes.c_void_p),
                        ivL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux),
                        ctypes.c_int(nocci*nvir),
                        ctypes.c_int(grpfac)
                    )
            else:
                ivL = lib.dot(ivL.reshape(nocci*nvir,naoaux), m2c)
            ovL[i0*nvir:i1*nvir] = ivL
        del h5tmp['Lov0']
        h5tmp.close()
        Lov0 = None
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[4] += np.asarray(tock) - np.asarray(tick)

    for tspan,tname in zip(tspans,tnames):
        log.debug('ao2mo CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan)
    log.info('')

    return ovL
