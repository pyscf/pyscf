#!/usr/bin/env python
# Copyright 2018-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Bryan Lau <blau1270@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

"""
Created on Thu May 17 11:05:22 2018

@author: Bryan Lau


A module that will do on-disk transformation of two electron integrals, and
also return specific slices of (o)ccupied and (v)irtual ones needed for post HF

Comparing to the full in-memory transformation (see incore.py) which holds all
intermediates in memory, this version uses less memory but performs slow due
to IO overhead.
"""

import time
import ctypes
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf.ao2mo import _ao2mo

IOBLK_SIZE = 128  # MB

def general(eri, mo_coeffs, erifile, dataname='eri_mo',
            ioblk_size=IOBLK_SIZE, compact=True, verbose=logger.NOTE):
    '''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals on disk.
    Args:
        eri : 8-fold reduced eri vector
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str or h5py File or h5py Group object
            To store the transformed integrals, in HDF5 format.
    Kwargs
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals


    Pseudocode / algorithm:
        u = mu
        v = nu
        l = lambda
        o = sigma

        Assume eri's are 8-fold reduced.
        nij/nkl_pair = npair or i*j/k*l if only transforming a subset

        First half transform:
            Initialize half_eri of size (nij_pair,npair)
                For lo = 1 -> npair
                    Unpack row lo
                    Unpack row lo to matrix E_{uv}^{lo}
                    Transform C_ui^+*E*C_nj -> E_{ij}^{lo}
                    Ravel or pack E_{ij}^{lo}
                    Save E_{ij}^{lo} -> half_eri[:,lo]

        Second half transform:
            Initialize h5d_eri of size (nij_pair,nkl_pair)
                For ij = 1 -> nij_pair
                    Load and unpack half_eri[ij,:] -> E_{lo}^{ij}
                    Transform C_{lk}E_{lo}^{ij}C_{ol} -> E_{kl}^{ij}
                    Repack E_{kl}^{ij}
                    Save E_{kl}^{ij} -> h5d_eri[ij,:]

        Each matrix is indexed by the composite index ij x kl, where ij/kl is
        either npair or ixj/kxl, if only a subset of MOs are being transformed.
        Since entire rows or columns need to be read in, the arrays are chunked
        such that IOBLK_SIZE = row/col x chunking col/row. For example, for the
        first half transform, we would save in nij_pair x IOBLK_SIZE/nij_pair,
        then load in IOBLK_SIZE/nkl_pair x npair for the second half transform.

        ------ kl ----->
        |jxl
        |
        ij
        |
        |
        v

        As a first guess, the chunking size is jxl. If the super-rows/cols are
        larger than IOBLK_SIZE, then the chunk rectangle jxl is trimmed
        accordingly. The pathological limiting case is where the dimensions
        nao_pair, nij_pair, or nkl_pair are so large that the arrays are
        chunked 1x1, in which case IOBLK_SIZE needs to be increased.

    '''
    log = logger.new_logger(None, verbose)
    log.info('******** ao2mo disk, custom eri ********')

    eri_ao = numpy.asarray(eri, order='C')
    nao, nmoi = mo_coeffs[0].shape
    nmoj = mo_coeffs[1].shape[1]
    nao_pair = nao*(nao+1)//2
    ijmosym, nij_pair, moij, ijshape = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
    klmosym, nkl_pair, mokl, klshape = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
    ijshape = (ijshape[0], ijshape[1]-ijshape[0],
               ijshape[2], ijshape[3]-ijshape[2])
    dtype = numpy.result_type(eri, *mo_coeffs)
    typesize = dtype.itemsize/1e6 # in MB

    if nij_pair == 0:
        return numpy.empty((nij_pair,nkl_pair))

    ij_red = ijmosym == 's1'
    kl_red = klmosym == 's1'

    if isinstance(erifile, str):
        if h5py.is_hdf5(erifile):
            feri = h5py.File(erifile, 'a')
            if dataname in feri:
                del(feri[dataname])
        else:
            feri = h5py.File(erifile,'w',libver='latest')
    else:
        assert(isinstance(erifile, h5py.Group))
        feri = erifile

    h5d_eri = feri.create_dataset(dataname,(nij_pair,nkl_pair), dtype.char)
    feri_swap = lib.H5TmpFile(libver='latest')
    chunk_size = min(nao_pair, max(4, int(ioblk_size*1e6/8/nao_pair)))

    log.debug('Memory information:')
    log.debug('  IOBLK_SIZE (MB): {}  chunk_size: {}'
              .format(ioblk_size, chunk_size))
    log.debug('  Final disk eri size (MB): {:.3g}'
              .format(nij_pair*nkl_pair*typesize))
    log.debug('  Half transformed eri size (MB): {:.3g}'
              .format(nij_pair*nao_pair*typesize))
    log.debug('  RAM buffer (MB): {:.3g}'
             .format(nij_pair*IOBLK_SIZE*typesize*2))

    if eri_ao.size == nao_pair**2: # 4-fold symmetry
        # half_e1 first transforms the indices which are contiguous in memory
        # transpose the 4-fold integrals to make ij the contiguous indices
        eri_ao = lib.transpose(eri_ao)
        ftrans = _ao2mo.libao2mo.AO2MOtranse1_incore_s4
    elif eri_ao.size == nao_pair*(nao_pair+1)//2:
        ftrans = _ao2mo.libao2mo.AO2MOtranse1_incore_s8
    else:
        raise NotImplementedError

    if ijmosym == 's2':
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_s2
    elif nmoi <= nmoj:
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
    else:
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_igtj
    fdrv = getattr(_ao2mo.libao2mo, 'AO2MOnr_e1incore_drv')

    def save(piece, buf):
        feri_swap[str(piece)] = buf.T

    # transform \mu\nu -> ij
    cput0 = time.clock(), time.time()
    with lib.call_in_background(save) as async_write:
        for istep, (p0, p1) in enumerate(lib.prange(0, nao_pair, chunk_size)):
            if dtype == numpy.double:
                buf = numpy.empty((p1-p0, nij_pair))
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     eri_ao.ctypes.data_as(ctypes.c_void_p),
                     moij.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(p0), ctypes.c_int(p1-p0),
                     ctypes.c_int(nao),
                     ctypes.c_int(ijshape[0]), ctypes.c_int(ijshape[1]),
                     ctypes.c_int(ijshape[2]), ctypes.c_int(ijshape[3]))
            else:  # complex
                tmp = numpy.empty((p1-p0, nao_pair))
                if eri_ao.size == nao_pair**2: # 4-fold symmetry
                    tmp = eri_ao[p0:p1]
                else: # 8-fold symmetry
                    for i in range(p0, p1):
                        tmp[i-p0] = lib.unpack_row(eri_ao, i)
                tmp = lib.unpack_tril(tmp, filltriu=lib.SYMMETRIC)
                buf = lib.einsum('xpq,pi,qj->xij', tmp, mo_coeffs[0].conj(), mo_coeffs[1])
                if ij_red:
                    buf = buf.reshape(p1-p0,-1) # grabs by row
                else:
                    buf = lib.pack_tril(buf)

            async_write(istep, buf)

    log.timer('(uv|lo) -> (ij|lo)', *cput0)

    # transform \lambda\sigma -> kl
    cput1 = time.clock(), time.time()
    Cklam = mo_coeffs[2].conj()
    buf_read = numpy.empty((chunk_size,nao_pair), dtype=dtype)
    buf_prefetch = numpy.empty_like(buf_read)

    def load(start, stop, buf):
        if start < stop:
            _load_from_h5g(feri_swap, start, stop, buf)

    def save(start, stop, buf):
        if start < stop:
            h5d_eri[start:stop] = buf[:stop-start]

    with lib.call_in_background(save,load) as (async_write, prefetch):
        for p0, p1 in lib.prange(0, nij_pair, chunk_size):
            if p0 == 0:
                load(p0, p1, buf_prefetch)

            buf_read, buf_prefetch = buf_prefetch, buf_read
            prefetch(p1, min(p1+chunk_size, nij_pair), buf_prefetch)

            lo = lib.unpack_tril(buf_read[:p1-p0], filltriu=lib.SYMMETRIC)
            lo = lib.einsum('xpq,pi,qj->xij', lo, Cklam, mo_coeffs[3])
            if kl_red:
                kl = lo.reshape(p1-p0,-1)
            else:
                kl = lib.pack_tril(lo)
            async_write(p0, p1, kl)

    log.timer('(ij|lo) -> (ij|kl)', *cput1)

    if isinstance(erifile, str):
        feri.close()
    return erifile

if __name__ == '__main__':
    import tempfile
    from pyscf import gto, scf, ao2mo
    # set verbose to 7 to get detailed timing info, otherwise 0
    verbose = 0

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '6311g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    mf.verbose = verbose
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[0]

    # compare custom outcore eri with incore eri
    nocc = numpy.count_nonzero(mf.mo_occ)
    nvir = nmo - nocc

    print('Full incore transformation (pyscf)...')
    start_time = time.time()
    eri_incore = ao2mo.incore.full(mf._eri, mo_coeff)
    onnn = eri_incore[:nocc*nmo].copy()
    print('    Time elapsed (s): ',time.time() - start_time)

    print('Parital incore transformation (pyscf)...')
    start_time = time.time()
    orbo = mo_coeff[:,:nocc]
    onnn2 = ao2mo.incore.general(mf._eri, (orbo,mo_coeff,mo_coeff,mo_coeff))
    print('    Time elapsed (s): ',time.time() - start_time)

    tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)

    print('\n\nCustom outcore transformation ...')
    orbo = mo_coeff[:,:nocc]
    start_time = time.time()
    general(mf._eri, (orbo,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa',
            verbose=verbose)
    stop_time = time.time() - start_time
    print('    Time elapsed (s): ',stop_time)
    print('\n\nPyscf outcore transformation ...')
    start_time = time.time()
    ao2mo.outcore.general(mol, (orbo,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'ab',
                          verbose=verbose)
    stop_time2 = time.time() - start_time
    print('    Time elapsed (s): ',stop_time2)
    print('How worse is the custom implemenation?',stop_time/stop_time2)
    with h5py.File(tmpfile2.name, 'r') as f:
        print('\n\nIncore (pyscf) vs outcore (custom)?',numpy.allclose(onnn2,f['aa']))
        print('Outcore (pyscf) vs outcore (custom)?',numpy.allclose(f['ab'],f['aa']))

    print('\n\nCustom full outcore transformation ...')
    start_time = time.time()
    general(mf._eri, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa',
            verbose=verbose)
    stop_time = time.time() - start_time
    print('    Time elapsed (s): ',stop_time)
    print('\n\nPyscf full outcore transformation ...')
    start_time = time.time()
    ao2mo.outcore.full(mol, mo_coeff, tmpfile2.name, 'ab',verbose=verbose)
    stop_time2 = time.time() - start_time
    print('    Time elapsed (s): ',stop_time2)
    print('    How worse is the custom implemenation?',stop_time/stop_time2)
    with h5py.File(tmpfile2.name, 'r') as f:
        print('\n\nIncore (pyscf) vs outcore (custom)?',numpy.allclose(eri_incore,f['aa']))
        print('Outcore (pyscf) vs outcore (custom)?',numpy.allclose(f['ab'],f['aa']))

    tmpfile2.close()
