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
from pyscf.mp import dfmp2, ump2
from pyscf.mp.ump2 import make_rdm1, make_rdm2, _mo_splitter
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfump2_with_t2', True)
THRESH_LINDEP = getattr(__config__, 'mp_dfump2_thresh_lindep', 1e-10)

libmp = dfmp2.libmp


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):

    log = logger.new_logger(mp, verbose)

    if eris is None: eris = mp.ao2mo()

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
    nvirmax = max(nvir)
    split_mo_energy = mp.split_mo_energy()
    occ_energy = [x[1] for x in split_mo_energy]
    vir_energy = [x[2] for x in split_mo_energy]

    mem_avail = mp.max_memory - lib.current_memory()[0]

    if with_t2:
        t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
              np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
              np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
        t2_ptr = [x.ctypes.data_as(ctypes.c_void_p) for x in t2]
        mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
    else:
        t2 = None
        t2_ptr = [lib.c_null_ptr()] * 3

    if mem_avail < 0:
        log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
        raise MemoryError

    emp2_ss = emp2_os = 0

    drv = libmp.MP2_contract_d

    # determine occ blksize
    if isinstance(eris.ovL[0], np.ndarray):    # incore ovL
        occ_blksize = nocc
    else:   # outcore ovL
        # 3*V^2 (for C driver) + 2*[O]XV (for iaL & jaL) = mem
        occ_blksize = int(np.floor((mem_avail*0.6*1e6/eris.dsize - 3*nvirmax**2)/(2*naux*nvirmax)))
        occ_blksize = [min(nocc[s], max(1, occ_blksize)) for s in [0,1]]

    log.debug('occ blksize for %s loop: %d/%d %d/%d', mp.__class__.__name__,
              occ_blksize[0], nocc[0], occ_blksize[1], nocc[1])

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    # same spin
    drv = libmp.MP2_contract_d
    for s in [0,1]:
        s_t2 = 0 if s == 0 else 2
        moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
        for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
            nocci = i1-i0
            iaL = eris.get_occ_blk(s,i0,i1)
            for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[s],occ_blksize[s])):
                noccj = j1-j0
                if ibatch == jbatch:
                    jbL = iaL
                else:
                    jbL = eris.get_occ_blk(s,j0,j1)

                ed = np.zeros(1, dtype=np.float64)
                ex = np.zeros(1, dtype=np.float64)
                moeoo_block = np.asarray(
                    occ_energy[s][i0:i1,None] + occ_energy[s][j0:j1], order='C')
                s2symm = 1
                t2_ex = True
                drv(
                    ed.ctypes.data_as(ctypes.c_void_p),
                    ex.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(s2symm),
                    iaL.ctypes.data_as(ctypes.c_void_p),
                    jbL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(i0), ctypes.c_int(j0),
                    ctypes.c_int(nocci), ctypes.c_int(noccj),
                    ctypes.c_int(nocc[s]), ctypes.c_int(nvir[s]), ctypes.c_int(naux),
                    moeoo_block.ctypes.data_as(ctypes.c_void_p),
                    moevv.ctypes.data_as(ctypes.c_void_p),
                    t2_ptr[s_t2], ctypes.c_int(t2_ex)
                )
                emp2_ss += (ed[0] + ex[0]) * 0.5

                jbL = None
            iaL = None

            cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i0,i1,nocc[s]),
                                     *cput1)

    # opposite spin
    sa, sb = 0, 1
    drv = libmp.MP2_OS_contract_d
    moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
    for ibatch,(i0,i1) in enumerate(lib.prange(0,nocc[sa],occ_blksize[sa])):
        nocci = i1-i0
        iaL = eris.get_occ_blk(sa,i0,i1)
        for jbatch,(j0,j1) in enumerate(lib.prange(0,nocc[sb],occ_blksize[sb])):
            noccj = j1-j0
            jbL = eris.get_occ_blk(sb,j0,j1)

            ed = np.zeros(1, dtype=np.float64)
            moeoo_block = np.asarray(
                occ_energy[sa][i0:i1,None] + occ_energy[sb][j0:j1], order='C')
            drv(
                ed.ctypes.data_as(ctypes.c_void_p),
                iaL.ctypes.data_as(ctypes.c_void_p),
                jbL.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(i0), ctypes.c_int(j0),
                ctypes.c_int(nocci), ctypes.c_int(noccj),
                ctypes.c_int(nocc[sa]), ctypes.c_int(nocc[sb]),
                ctypes.c_int(nvir[sa]), ctypes.c_int(nvir[sb]),
                ctypes.c_int(naux),
                moeoo_block.ctypes.data_as(ctypes.c_void_p),
                moevv.ctypes.data_as(ctypes.c_void_p),
                t2_ptr[1]
            )
            emp2_os += ed[0]

            jbL = None
        iaL = None

        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i0,i1,nocc[sa]),
                                 *cput1)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFUMP2(ump2.UMP2):
    _keys = dfmp2.DFRMP2._keys

    __init__ = dfmp2.DFRMP2.__init__

    get_nocc = ump2.get_nocc
    get_nmo = ump2.get_nmo
    get_frozen_mask = ump2.get_frozen_mask

    kernel = ump2.UMP2.kernel

    make_fno = ump2.make_fno
    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    reset = dfmp2.DFRMP2.reset

    def split_mo_coeff(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        masks = _mo_splitter(self)
        return [[mo_coeff[s][:,m] for m in masks[s]] for s in [0,1]]

    def split_mo_energy(self, mo_energy=None):
        if mo_energy is None: mo_energy = self.mo_energy
        masks = _mo_splitter(self)
        return [[mo_energy[s][m] for m in masks[s]] for s in [0,1]]

    def split_mo_occ(self, mo_occ=None):
        if mo_occ is None: mo_occ = self.mo_occ
        masks = _mo_splitter(self)
        return [[mo_occ[s][m] for m in masks[s]] for s in [0,1]]

    def ao2mo(self, mo_coeff=None, ovL=None, ovL_to_save=None):
        return _make_df_eris(self, mo_coeff, ovL, ovL_to_save)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    Gradients = NotImplemented

MP2 = UMP2 = DFUMP2

from pyscf import scf
scf.uhf.UHF.DFMP2 = lib.class_as_method(DFUMP2)
del (WITH_T2)


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
    split_mo_coeff = mp.split_mo_coeff()
    occ_coeff = [x[1] for x in split_mo_coeff]
    vir_coeff = [x[2] for x in split_mo_coeff]

    # determine incore or outcore
    nocc = np.asarray([x.shape[1] for x in occ_coeff])
    nvir = np.asarray([x.shape[1] for x in vir_coeff])

    if ovL is not None:
        if isinstance(ovL, (np.ndarray,list,tuple)):
            outcore = False
        elif isinstance(ovL, str):
            outcore = True
        else:
            log.error('Unknown data type %s for input `ovL` (should be np.ndarray or str).',
                      type(ovL))
            raise TypeError
    else:
        mem_now = mp.max_memory - lib.current_memory()[0]
        mem_df = sum(nocc*nvir)*naux*8/1024**2.
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

        self.dtype = np.result_type(*self.occ_coeff)
        assert( self.dtype == np.float64 )  # FIXME: support complex
        self.dsize = 8

        self.ovL = ovL

    @property
    def nocc(self):
        return [x.shape[1] for x in self.occ_coeff]
    @property
    def nvir(self):
        return [x.shape[1] for x in self.vir_coeff]
    @property
    def naux(self):
        return self.ovL[0].shape[-1]

    def build(self):
        log = logger.new_logger(self)
        if self.ovL is None:
            if self.with_df._cderi is None:
                self.ovL = _init_mp_df_eris_direct(self.with_df, self.occ_coeff, self.vir_coeff,
                                                   self.max_memory, log=log)
            else:
                self.ovL = _init_mp_df_eris(self.with_df, self.occ_coeff, self.vir_coeff,
                                            self.max_memory, log=log)

    def get_occ_blk(self, s,i0,i1):
        nvir, naux = self.nvir[s], self.naux
        return np.asarray(self.ovL[s][i0*nvir:i1*nvir], order='C').reshape(i1-i0,nvir,naux)
    def get_ov_blk(self, s,ia0,ia1):
        return np.asarray(self.ovL[s][ia0:ia1], order='C')


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
            self.ovL = [self.feri[f'ovL{s}'] for s in [0,1]]
        elif isinstance(self._ovL, str):
            self.feri = h5py.File(self._ovL, 'r')
            log.debug('ovL is read from %s', self.feri.filename)
            assert( 'ovL0' in self.feri and 'ovL1' in self.feri )
            self.ovL = [self.feri[f'ovL{s}'] for s in [0,1]]
        else:
            raise RuntimeError

def _init_mp_df_eris(with_df, occ_coeff, vir_coeff, max_memory, h5obj=None, log=None):
    if log is None: log = logger.new_logger(with_df)

    nao = occ_coeff[0].shape[0]
    nocc = np.asarray([x.shape[1] for x in occ_coeff])
    nvir = np.asarray([x.shape[1] for x in vir_coeff])
    noccmax = max(nocc)
    nvirmax = max(nvir)
    nmo = [nocc[s] + nvir[s] for s in [0,1]]
    nao_pair = nao**2
    naux = with_df.get_naoaux()

    dtype = np.result_type(*occ_coeff)
    assert( dtype == np.float64 )
    dsize = 8

    mo = [np.asarray(np.hstack((occ_coeff[s],vir_coeff[s])), order='F') for s in [0,1]]

    mem_avail = max_memory - lib.current_memory()[0]

    if h5obj is None:   # incore
        ovL = [np.empty((nocc[s]*nvir[s],naux), dtype=dtype) for s in [0,1]]
        mem_avail -= sum(nocc*nvir)*naux * dsize/1e6
    else:
        ovL = []
        for s in [0,1]:
            ovL_shape = (nocc[s]*nvir[s],naux)
            ovL.append(h5obj.create_dataset(f'ovL{s}', ovL_shape, dtype=dtype,
                                            chunks=(1,*ovL_shape[1:])))

    if isinstance(ovL, np.ndarray):
        # incore: batching aux (OV + Nao_pair) * [X] = M
        mem_auxblk = (nao_pair+noccmax*nvirmax) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.5 / mem_auxblk))))
        log.debug('aux blksize for incore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(aux_blksize*noccmax*nvirmax, dtype=dtype)
        ijslice = [(0, nocc[s], nocc[s], nmo[s]) for s in [0,1]]

        p1 = 0
        for Lpq in with_df.loop(blksize=aux_blksize):
            p0, p1 = p1, p1+Lpq.shape[0]
            for s in [0,1]:
                out = _ao2mo.nr_e2(Lpq, mo[s], ijslice[s], aosym='s2', out=buf)
                ovL[s][:,p0:p1] = out.T
                out = None
            Lpq = None
        buf = None
    else:
        # outcore: batching occ [O]XV and aux ([O]V + Nao_pair)*[X]
        mem_occblk = naux*nvirmax * dsize/1e6
        occ_blksize = min(noccmax, max(1, int(np.floor(mem_avail*0.6 / mem_occblk))))
        mem_auxblk = (occ_blksize*nvirmax+nao_pair) * dsize/1e6
        aux_blksize = min(naux, max(1, int(np.floor(mem_avail*0.3 / mem_auxblk))))
        log.debug('occ blksize for outcore ao2mo: %d/%d', occ_blksize, noccmax)
        log.debug('aux blksize for outcore ao2mo: %d/%d', aux_blksize, naux)
        buf = np.empty(naux*occ_blksize*nvirmax, dtype=dtype)
        buf2 = np.empty(aux_blksize*occ_blksize*nvirmax, dtype=dtype)

        for s in [0,1]:
            for i0,i1 in lib.prange(0,nocc[s],occ_blksize):
                nocci = i1-i0
                ijslice = (i0,i1,nocc[s],nmo[s])
                p1 = 0
                OvL = np.ndarray((nocci*nvir[s],naux), dtype=dtype, buffer=buf)
                for Lpq in with_df.loop(blksize=aux_blksize):
                    p0, p1 = p1, p1+Lpq.shape[0]
                    out = _ao2mo.nr_e2(Lpq, mo[s], ijslice, aosym='s2', out=buf2)
                    OvL[:,p0:p1] = out.T
                    Lpq = out = None
                # this avoids slow operations like ovL[i0:i1,:,p0:p1] = ...
                ovL[s][i0*nvir[s]:i1*nvir[s]] = OvL
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
    nao = occ_coeff[0].shape[0]
    nocc = np.asarray([x.shape[1] for x in occ_coeff])
    nvir = np.asarray([x.shape[1] for x in vir_coeff])
    noccmax = max(nocc)
    nvirmax = max(nvir)
    nmo = [nocc[s] + nvir[s] for s in [0,1]]
    nao_pair = nao*(nao+1)//2
    naoaux = auxmol.nao_nr()

    dtype = np.result_type(*occ_coeff)
    assert( dtype == np.float64 )
    dsize = 8

    mo = [np.asarray(np.hstack((occ_coeff[s],vir_coeff[s])), order='F') for s in [0,1]]
    ijslice = [(0, nocc[s], nocc[s], nmo[s]) for s in [0,1]]

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
        ovL = [np.empty((nocc[s]*nvir[s],naux), dtype=dtype) for s in [0,1]]
        mem_avail -= sum([x.size for x in ovL]) * dsize / 1e6
    else:
        ovL = []
        for s in [0,1]:
            ovL_shape = (nocc[s]*nvir[s],naux)
            ovL.append( h5obj.create_dataset(f'ovL{s}', ovL_shape, dtype=dtype,
                                             chunks=(1,*ovL_shape[1:])) )
        h5tmp = lib.H5TmpFile()
        Lov0 = []
        for s in [0,1]:
            Lov0_shape = (naoaux,nocc[s]*nvir[s])
            Lov0.append( h5tmp.create_dataset(f'Lov0{s}', Lov0_shape, dtype=dtype,
                                              chunks=(1,*Lov0_shape[1:])) )

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
        for s in [0,1]:
            lov = _ao2mo.nr_e2(lpq, mo[s], ijslice[s], aosym='s2', out=buf0)
            tick = (logger.process_clock(), logger.perf_counter())
            tspans[2] += np.asarray(tick) - np.asarray(tock)
            if incore:
                ovl = lib.transpose(lov, out=buf0T)
                ovL[s][:,k0:k1] = ovl
                ovl = None
            else:
                Lov0[s][k0:k1] = lov
            lov = None
            tock = (logger.process_clock(), logger.perf_counter())
            tspans[3] += np.asarray(tock) - np.asarray(tick)
        lpq = None
    buf0 = buf0T = None

    tick = (logger.process_clock(), logger.perf_counter())
    # fit
    if tag == 'cd': drv = getattr(libmp, 'trisolve_parallel_grp', None)
    if incore:
        if tag == 'cd':
            if drv is None:
                for s in [0,1]:
                    scipy.linalg.solve_triangular(m2c, ovL[s].T, lower=True,
                                                  overwrite_b=True, check_finite=False).T
            else:
                assert m2c.flags.f_contiguous
                grpfac = 10
                for s in [0,1]:
                    drv(
                        m2c.ctypes.data_as(ctypes.c_void_p),
                        ovL[s].ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux),
                        ctypes.c_int(nocc[s]*nvir[s]),
                        ctypes.c_int(grpfac)
                    )
        else:
            mem_blk = nvirmax*naux * dsize/1e6
            occ_blksize0 = max(1, min(noccmax, int(np.floor(mem_avail*0.5/mem_blk))))
            buf = np.empty(occ_blksize0*nvirmax*naux, dtype=dtype)
            for s in [0,1]:
                occ_blksize = min(nocc[s], occ_blksize0)
                ovL[s] = ovL[s].reshape(-1)
                nvxao = nvir[s]*naoaux
                nvx = nvir[s]*naux
                for i0,i1 in lib.prange(0,nocc[s],occ_blksize):
                    nocci = i1-i0
                    out = np.ndarray((nocci*nvir[s],naux), dtype=dtype, buffer=buf)
                    lib.dot(ovL[s][i0*nvxao:i1*nvxao].reshape(nocci*nvir[s],naoaux), m2c, c=out)
                    ovL[s][i0*nvx:i1*nvx] = out.reshape(-1)
                ovL[s] = ovL[s][:nocc[s]*nvx].reshape(nocc[s]*nvir[s],naux)
            buf = None
    else:
        mem_blk = nvirmax*naoaux * dsize / 1e6
        occ_blksize0 = max(1, min(noccmax, int(np.floor(mem_avail*0.4/mem_blk))))
        for s in [0,1]:
            occ_blksize = min(nocc[s], occ_blksize0)
            nvxao = nvir[s]*naoaux
            nvx = nvir[s]*naux
            for i0,i1 in lib.prange(0, nocc[s], occ_blksize):
                nocci = i1-i0
                ivL = np.asarray(Lov0[s][:,i0*nvir[s]:i1*nvir[s]].T, order='C')
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
                            ctypes.c_int(nocci*nvir[s]),
                            ctypes.c_int(grpfac)
                        )
                else:
                    ivL = lib.dot(ivL.reshape(nocci*nvir[s],naoaux), m2c)
                ovL[s][i0*nvir[s]:i1*nvir[s]] = ivL

        for s in [0,1]:
            del h5tmp[f'Lov0{s}']
        h5tmp.close()
        Lov0 = None
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[4] += np.asarray(tock) - np.asarray(tick)

    for tspan,tname in zip(tspans,tnames):
        log.debug('ao2mo CPU time for %-10s  %9.2f sec  wall time %9.2f sec', tname, *tspan)
    log.info('')

    return ovL
