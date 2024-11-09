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

'''
ao2mo of scalar integrals to complex MO integrals for GHF orbitals
'''

import time
import tempfile
import numpy
import h5py
import ctypes
import _ctypes
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import outcore
from pyscf import __config__
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3

libao2mo = lib.load_library('libao2mo')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libao2mo._handle, name))

IOBLK_SIZE = getattr(__config__, 'ao2mo_outcore_ioblk_size', 256)  # 256 MB
IOBUF_WORDS = getattr(__config__, 'ao2mo_outcore_iobuf_words', 1e8)  # 1.6 GB
IOBUF_ROW_MIN = getattr(__config__, 'ao2mo_outcore_row_min', 160)
MAX_MEMORY = getattr(__config__, 'ao2mo_outcore_max_memory', 4000)  # 4GB

def full(mol, mo_coeff, erifile, dataname='eri_mo',
         intor='int2e_sph', motype='ghf', aosym='s1', comp=None,
         max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.WARN):
    general(mol, (mo_coeff,)*4, erifile, dataname,
            intor, motype, aosym, comp, max_memory, ioblk_size, verbose)
    return erifile

def general(mol, mo_coeffs, erifile, dataname='eri_mo',
            intor='int2e_sph', motype='ghf', aosym='s1', comp=None,
            max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.debug):
    time_0pass = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    intor, comp = gto.moleintor._get_intor_and_comp(mol._add_suffix(intor), comp)
    klsame = iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]//2

    if motype == 'j-spinor':
        ca, cb = mol.sph2spinor_coeff()
        mo_alph = [numpy.dot(ca, moi) for moi in mo_coeffs]
        mo_beta = [numpy.dot(cb, moi) for moi in mo_coeffs]
    elif motype == 'ghf':
        nao = mo_coeffs[0].shape[0] // 2
        mo_alph = [moi[:nao,:] for moi in mo_coeffs]
        mo_beta = [moi[nao:,:] for moi in mo_coeffs]
    else:
        raise AssertionError(f'Unknown motype {motype}. should be one of "j-spinor" or "ghf".')

    aosym = outcore._stand_sym_code(aosym)
    if aosym in ('s1', 's2ij', 'a2ij'):
        nao_pair = nao * nao
    else:
        nao_pair = _count_naopair(mol, nao)

    nij_pair = nmoi*nmoj
    nkl_pair = nmok*nmol

    if klsame and aosym in ('s4', 's2kl', 'a2kl', 'a4ij', 'a4kl', 'a4'):
        log.debug('k-mo == l-mo')
        mokla = numpy.asarray(mo_alph[2], dtype=numpy.complex128, order='F')
        moklb = numpy.asarray(mo_beta[2], dtype=numpy.complex128, order='F')
        klshape = (0, nmok, 0, nmok)
    else:
        mokla = numpy.asarray(numpy.hstack((mo_alph[2],mo_alph[3])),
                             dtype=numpy.complex128, order='F')
        moklb = numpy.asarray(numpy.hstack((mo_beta[2],mo_beta[3])),
                             dtype=numpy.complex128, order='F')
        klshape = (0, nmok, nmok, nmok+nmol)

    if isinstance(erifile, str):
        if h5py.is_hdf5(erifile):
            feri = lib.H5FileWrap(erifile, 'a')
            if dataname in feri:
                del(feri[dataname])
        else:
            feri = lib.H5FileWrap(erifile, 'w')
    else:
        assert(isinstance(erifile, h5py.Group))
        feri = erifile

    if comp == 1:
        chunks = (nmoj,nmol)
        shape = (nij_pair, nkl_pair)
    else:
        chunks = (1,nmoj,nmol)
        shape = (comp, nij_pair, nkl_pair)

    if nij_pair == 0 or nkl_pair == 0:
        feri.create_dataset(dataname, shape, 'c16')
        if isinstance(erifile, str):
            feri.close()
        return erifile
    else:
        h5d_eri = feri.create_dataset(dataname, shape, 'c16', chunks=chunks)

    log.debug('MO integrals %s are saved in %s/%s', intor, erifile, dataname)
    log.debug('num. MO ints = %.8g, required disk %.8g MB',
              float(nij_pair)*nkl_pair*comp, nij_pair*nkl_pair*comp*16/1e6)

# transform e1
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    half_e1(mol, (mo_alph, mo_beta), swapfile.name, intor, aosym, comp,
            max_memory, ioblk_size, log)
    time_1pass = log.timer('AO->MO transformation for %s 1 pass'%intor,
                           *time_0pass)

    e2buflen = guess_e2bufsize(ioblk_size, nij_pair, nao_pair)[0]

    log.debug('step2: kl-pair (ao %d, mo %d), mem %.8g MB, '
              'ioblock (r/w) %.8g/%.8g MB',
              nao_pair, nkl_pair, e2buflen*nao_pair*16/1e6,
              e2buflen*nij_pair*16/1e6, e2buflen*nkl_pair*16/1e6)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    ijmoblks = int(numpy.ceil(float(nij_pair)/e2buflen)) * comp
    #ao_loc = numpy.asarray(mol.ao_loc_2c(), dtype=numpy.int32)
    ao_loc = numpy.asarray(mol.ao_loc, dtype=numpy.int32)
    tao = numpy.asarray(mol.tmap(), dtype=numpy.int32)
    ti0 = time_1pass
    buf = numpy.empty((e2buflen, nao_pair), dtype=numpy.complex128)
    istep = 0
    for row0, row1 in prange(0, nij_pair, e2buflen):
        nrow = row1 - row0

        for icomp in range(comp):
            istep += 1
            tioi = 0
            log.debug('step 2 [%d/%d], [%d,%d:%d], row = %d',
                      istep, ijmoblks, icomp, row0, row1, nrow)

            col0 = 0
            for ic in range(klaoblks):
                dat = fswap['%d/%d'%(icomp,ic)]
                col1 = col0 + dat.shape[1]
                buf[:nrow,col0:col1] = dat[row0:row1]
                col0 = col1
            ti2 = log.timer('step 2 [%d/%d], load buf'%(istep,ijmoblks), *ti0)
            tioi += ti2[1]-ti0[1]
            pbuf = _ao2mo.r_e2(buf[:nrow], mokla, klshape, tao, ao_loc, aosym)
            pbuf2 = _ao2mo.r_e2(buf[:nrow], moklb, klshape, tao, ao_loc, aosym)

            tw1 = logger.perf_counter()
            if comp == 1:
                h5d_eri[row0:row1] = pbuf+pbuf2
            else:
                h5d_eri[icomp,row0:row1] = pbuf+pbuf2
            tioi += logger.perf_counter()-tw1

            ti1 = (logger.process_clock(), logger.perf_counter())
            log.debug('step 2 [%d/%d] CPU time: %9.2f, Wall time: %9.2f, I/O time: %9.2f',
                      istep, ijmoblks, ti1[0]-ti0[0], ti1[1]-ti0[1], tioi)
            ti0 = ti1
    buf = pbuf = None
    fswap.close()
    if isinstance(erifile, str):
        feri.close()

    log.timer('AO->MO transformation for %s 2 pass'%intor, *time_1pass)
    log.timer('AO->MO transformation for %s '%intor, *time_0pass)
    return erifile

def full_iofree(mol, mo_coeff, dataname='eri_mo', intor='int2e_sph',
                motype='ghf', aosym='s1', comp=None, verbose=logger.debug,
                **kwargs):
    erifile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    general(mol, (mo_coeff,)*4, erifile.name, dataname='eri_mo',
            intor=intor, motype=motype, aosym=aosym, comp=comp,
            verbose=verbose)
    with h5py.File(erifile.name, 'r') as feri:
        return numpy.asarray(feri['eri_mo'])

def general_iofree(mol, mo_coeffs, dataname='eri_mo', intor='int2e_sph',
                   motype='ghf', aosym='s1', comp=None, verbose=logger.debug,
                   **kwargs):
    erifile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    general(mol, mo_coeffs, erifile.name, dataname='eri_mo',
            intor=intor, motype=motype, aosym=aosym, comp=comp,
            verbose=verbose)
    with h5py.File(erifile.name, 'r') as feri:
        return numpy.asarray(feri['eri_mo'])

# swapfile will be overwritten if exists.
# mo_coeffs contains two sets of mos, one is alpha back transformed, one is beta back transformed.
def half_e1(mol, mo_coeffs, swapfile,
            intor='int2e_sph', aosym='s1', comp=None,
            max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.WARN,
            ao2mopt=None):
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)

    mo_alph = mo_coeffs[0]
    mo_beta = mo_coeffs[1]
    ijsame = iden_coeffs(mo_alph[0], mo_alph[1])

    nmoi = mo_alph[0].shape[1]
    nmoj = mo_alph[1].shape[1]
    nao = mo_alph[0].shape[0]
    aosym = outcore._stand_sym_code(aosym)
    if aosym in ('s1', 's2kl', 'a2kl'):
        nao_pair = nao * nao
    else:
        nao_pair = _count_naopair(mol, nao)
    nij_pair = nmoi * nmoj

    if ijsame and aosym in ('s4', 's2ij', 'a2ij', 'a4ij', 'a4kl', 'a4'):
        log.debug('i-mo == j-mo')
        moija = numpy.asarray(mo_alph[0], order='F')
        moijb = numpy.asarray(mo_beta[0], order='F')
        ijshape = (0, nmoi, 0, nmoi)
    else:
        moija = numpy.asarray(numpy.hstack((mo_alph[0],mo_alph[1])), order='F')
        moijb = numpy.asarray(numpy.hstack((mo_beta[0],mo_beta[1])), order='F')
        ijshape = (0, nmoi, nmoi, nmoi+nmoj)

    e1buflen, mem_words, iobuf_words, ioblk_words = \
            guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp)
# The buffer to hold AO integrals in C code
    aobuflen = int((mem_words - iobuf_words) // (nao*nao*comp))
    shranges = outcore.guess_shell_ranges(mol, (aosym not in ('s1', 's2ij', 'a2ij')),
                                          aobuflen, e1buflen, mol.ao_loc, False)

    log.debug('step1: tmpfile %.8g MB', nij_pair*nao_pair*16/1e6)
    log.debug('step1: (ij,kl) = (%d,%d), mem cache %.8g MB, iobuf %.8g MB',
              nij_pair, nao_pair, mem_words*16/1e6, iobuf_words*16/1e6)

    fswap = lib.H5FileWrap(swapfile, 'w')
    for icomp in range(comp):
        fswap.create_group(str(icomp))  # for h5py old version

    tao = numpy.asarray(mol.tmap(), dtype=numpy.int32)

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.half_e1', *time0)
    nstep = len(shranges)
    for istep,sh_range in enumerate(shranges):
        log.debug('step 1 [%d/%d], AO [%d:%d], len(buf) = %d',
                  istep+1, nstep, *(sh_range[:3]))
        buflen = sh_range[2]
        iobuf = numpy.empty((comp,buflen,nij_pair), dtype=numpy.complex128)
        nmic = len(sh_range[3])
        p0 = 0
        for imic, aoshs in enumerate(sh_range[3]):
            log.debug1('      fill iobuf micro [%d/%d], AO [%d:%d], len(aobuf) = %d',
                       imic+1, nmic, *aoshs)
            buf = r_e1(intor, moija, moijb, ijshape, aoshs,
                       mol._atm, mol._bas, mol._env,
                       tao, aosym, comp, ao2mopt)
            iobuf[:,p0:p0+aoshs[2]] = buf
            p0 += aoshs[2]
        ti2 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)

        e2buflen, chunks = guess_e2bufsize(ioblk_size, nij_pair, buflen)
        for icomp in range(comp):
            dset = fswap.create_dataset('%d/%d'%(icomp,istep),
                                        (nij_pair,iobuf.shape[1]), 'c16',
                                        chunks=None)
            for col0, col1 in prange(0, nij_pair, e2buflen):
                dset[col0:col1] = lib.transpose(iobuf[icomp,:,col0:col1])
        ti0 = log.timer('transposing to disk', *ti2)
    fswap.close()
    return swapfile

# if out is not None, transform AO to MO in-place
def r_e1(intor, mo_a, mo_b, orbs_slice, sh_range, atm, bas, env,
         tao, aosym='s1', comp=1, ao2mopt=None, out=None):
    assert(aosym in ('s4', 's2ij', 's2kl', 's1', 'a2ij', 'a2kl', 'a4ij',
                     'a4kl', 'a4'))
    intor = ascint3(intor)
    mo_a = numpy.asarray(mo_a, dtype=numpy.complex128, order='F')
    mo_b = numpy.asarray(mo_b, dtype=numpy.complex128, order='F')
    i0, i1, j0, j1 = orbs_slice
    icount = i1 - i0
    jcount = j1 - j0
    ij_count = icount * jcount

    c_atm = numpy.asarray(atm, dtype=numpy.int32)
    c_bas = numpy.asarray(bas, dtype=numpy.int32)
    c_env = numpy.asarray(env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    klsh0, klsh1, nkl = sh_range

    #if icount <= jcount:
    fmmm = _fpointer('AO2MOmmm_nrr_iltj')
    #else:
    #   fmmm = _fpointer('AO2MOmmm_nrr_igtj')

    out = numpy.ndarray((2*comp,nkl,ij_count), dtype=numpy.complex128,
                        buffer=out)
    if out.size == 0:
        return out

    if ao2mopt is not None:
        cao2mopt = ao2mopt._this
        cintopt = ao2mopt._cintopt
        intor = ao2mopt._intor
    else:
        cao2mopt = lib.c_null_ptr()
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    cintor = _fpointer(intor)

    tao = numpy.asarray(tao, dtype=numpy.int32)
    ao_loc = make_loc(c_bas, 'int2e_sph')

    fdrv = getattr(libao2mo, 'AO2MOnrr_e1_drv')
    fill = _fpointer('AO2MOfill_nrr_' + aosym)
    ftrans = _fpointer('AO2MOtranse1_nrr_' + aosym)
    fdrv(cintor, fill, ftrans, fmmm,
         out.ctypes.data_as(ctypes.c_void_p),
         mo_a.ctypes.data_as(ctypes.c_void_p),
         mo_b.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(klsh0), ctypes.c_int(klsh1-klsh0),
         ctypes.c_int(nkl), ctypes.c_int(comp),
         (ctypes.c_int*4)(*orbs_slice), tao.ctypes.data_as(ctypes.c_void_p),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cao2mopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))
    for i in range(comp):
        out[i,:,:] += out[2*i+1,:,:]
    return out[:comp,:,:]

def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) \
            or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp):
    mem_words = max_memory * 1e6 / 16
# part of the max_memory is used to hold the AO integrals.  The iobuf is the
# buffer to temporary hold the transformed integrals before streaming to disk.
# iobuf is then divided to small blocks (ioblk_words) and streamed to disk.
    if mem_words > IOBUF_WORDS * 2:
        iobuf_words = int(IOBUF_WORDS)
    else:
        iobuf_words = int(mem_words // 2)
    ioblk_words = int(min(ioblk_size*1e6/16, iobuf_words))

    e1buflen = int(min(iobuf_words//(comp*nij_pair), nao_pair))
    return e1buflen, mem_words, iobuf_words, ioblk_words

def guess_e2bufsize(ioblk_size, nrows, ncols):
    e2buflen = int(min(ioblk_size*1e6/16/ncols, nrows))
    e2buflen = max(e2buflen//IOBUF_ROW_MIN, 1) * IOBUF_ROW_MIN
    chunks = (IOBUF_ROW_MIN, ncols)
    return e2buflen, chunks

def _count_naopair(mol, nao):
    ao_loc = mol.ao_loc_2c()
    nao_pair = 0
    for i in range(mol.nbas):
        di = int(ao_loc[i+1] - ao_loc[i])
        for j in range(i+1):
            dj = int(ao_loc[j+1] - ao_loc[j])
            nao_pair += di * dj
    return nao_pair

del(MAX_MEMORY)
