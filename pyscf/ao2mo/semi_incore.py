#!/usr/bin/env python
# Copyright 2018 The PySCF Developers. All Rights Reserved.
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
# Author: Bryan Lau
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
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger

IOBLK_SIZE = 128

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

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]

    nao_pair = nao*(nao+1) // 2
    if compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
        ij_red = False
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        ij_red = True
        nij_pair = nmoi*nmoj
    if compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3]):
        kl_red = False
        nkl_pair = nmok*(nmok+1) // 2
    else:
        kl_red = True
        nkl_pair = nmok*nmol

    dtype = numpy.result_type(eri, *mo_coeffs)
    typesize = dtype.itemsize/1e6 # in MB
    chunks_half = (max(1, numpy.minimum(int(ioblk_size//(nao_pair*typesize)),nmoj)),
                   max(1, numpy.minimum(int(ioblk_size//(nij_pair*typesize)),nmol)))
    '''
    ideally, the final transformed eris should have a chunk of nmoj x nmol to
    optimize read operations. However, I'm chunking the row size so that the
    write operations during the transform can be done as fast as possible.
    '''
    chunks_full = (numpy.minimum(int(ioblk_size//(nkl_pair*typesize)),nmoj),nmol)

    if isinstance(erifile, str):
        if h5py.is_hdf5(erifile):
            feri = h5py.File(erifile)
            if dataname in feri:
                del(feri[dataname])
        else:
            feri = h5py.File(erifile,'w',libver='latest')
    else:
        assert(isinstance(erifile, h5py.Group))
        feri = erifile

    h5d_eri = feri.create_dataset(dataname,(nij_pair,nkl_pair),
                                  dtype.char, chunks=chunks_full)

    feri_swap = lib.H5TmpFile(libver='latest')
    half_eri = feri_swap.create_dataset(dataname,(nij_pair,nao_pair),
                                        dtype.char, chunks=chunks_half)

    log.debug('Memory information:')
    log.debug('  IOBLK_SIZE (MB): {}'.format(ioblk_size))
    log.debug('  jxl {}x{}, half eri chunk dim  {}x{}'.format(nmoj,nmol,chunks_half[0],chunks_half[1]))
    log.debug('  jxl {}x{}, full eri chunk dim {}x{}'.format(nmoj,nmol,chunks_full[0],chunks_full[1]))
    log.debug('  Final disk eri size (MB): {:.3g}, chunked {:.3g}'
              .format(nij_pair*nkl_pair*typesize,numpy.prod(chunks_full)*typesize))
    log.debug('  Half transformed eri size (MB): {:.3g}, chunked {:.3g}'
              .format(nij_pair*nao_pair*typesize,numpy.prod(chunks_half)*typesize))
    log.debug('  RAM buffer for half transform (MB): {:.3g}'
             .format(nij_pair*chunks_half[1]*typesize*2))
    log.debug('  RAM buffer for full transform (MB): {:.3g}'
             .format(typesize*chunks_full[0]*nkl_pair*2 + chunks_half[0]*nao_pair*typesize*2))

    def save1(piece,buf):
        start = piece*chunks_half[1]
        stop = (piece+1)*chunks_half[1]
        if stop > nao_pair:
            stop = nao_pair
        half_eri[:,start:stop] = buf[:,:stop-start]
        return

    def load2(piece):
        start = piece*chunks_half[0]
        stop = (piece+1)*chunks_half[0]
        if stop > nij_pair:
            stop = nij_pair
            if start >= nij_pair:
                start = stop - 1
        return half_eri[start:stop,:]

    def prefetch2(piece):
        start = piece*chunks_half[0]
        stop = (piece+1)*chunks_half[0]
        if stop > nij_pair:
            stop = nij_pair
            if start >= nij_pair:
                start = stop - 1
        buf_prefetch[:stop-start,:] = half_eri[start:stop,:]
        return

    def save2(piece,buf):
        start = piece*chunks_full[0]
        stop = (piece+1)*chunks_full[0]
        if stop > nij_pair:
            stop = nij_pair
        h5d_eri[start:stop,:] = buf[:stop-start,:]
        return

    # transform \mu\nu -> ij
    cput0 = time.clock(), time.time()
    Cimu = mo_coeffs[0].conj().transpose()
    buf_write = numpy.empty((nij_pair,chunks_half[1]))
    buf_out = numpy.empty_like(buf_write)
    wpiece = 0
    with lib.call_in_background(save1) as async_write:
        for lo in range(nao_pair):
            if lo % chunks_half[1] == 0 and lo > 0:
                #save1(wpiece,buf_write)
                buf_out, buf_write = buf_write, buf_out
                async_write(wpiece,buf_out)
                wpiece += 1
            buf = lib.unpack_row(eri,lo)
            uv = lib.unpack_tril(buf)
            uv = Cimu.dot(uv).dot(mo_coeffs[1])
            if ij_red:
                ij = numpy.ravel(uv) # grabs by row
            else:
                ij = lib.pack_tril(uv)
            buf_write[:,lo % chunks_half[1]] = ij
    # final write operation & cleanup
    save1(wpiece,buf_write)
    log.timer('(uv|lo) -> (ij|lo)', *cput0)
    uv = None
    ij = None
    buf = None

    # transform \lambda\sigma -> kl
    cput1 = time.clock(), time.time()
    Cklam = mo_coeffs[2].conj().transpose()
    buf_write = numpy.empty((chunks_full[0],nkl_pair))
    buf_out = numpy.empty_like(buf_write)
    buf_read = numpy.empty((chunks_half[0],nao_pair))
    buf_prefetch = numpy.empty_like(buf_read)
    rpiece = 0
    wpiece = 0
    with lib.call_in_background(save2,prefetch2) as (async_write,prefetch):
        buf_read = load2(rpiece)
        prefetch(rpiece+1)
        for ij in range(nij_pair):
            if ij % chunks_full[0] == 0 and ij > 0:
                #save2(wpiece,buf_write)
                buf_out, buf_write = buf_write, buf_out
                async_write(wpiece,buf_out)
                wpiece += 1
            if ij % chunks_half[0] == 0 and ij > 0:
                #buf_read = load2(rpiece)
                buf_read, buf_prefetch = buf_prefetch, buf_read
                rpiece += 1
                prefetch(rpiece+1)
            lo = lib.unpack_tril(buf_read[ij % chunks_half[0],:])
            lo = Cklam.dot(lo).dot(mo_coeffs[3])
            if kl_red:
                kl = numpy.ravel(lo)
            else:
                kl = lib.pack_tril(lo)
            buf_write[ij % chunks_full[0],:] = kl
    save2(wpiece,buf_write)
    log.timer('(ij|lo) -> (ij|kl)', *cput1)

    if isinstance(erifile, str):
        feri.close()
    return erifile

def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

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
    with h5py.File(tmpfile2.name) as f:
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
    with h5py.File(tmpfile2.name) as f:
        print('\n\nIncore (pyscf) vs outcore (custom)?',numpy.allclose(eri_incore,f['aa']))
        print('Outcore (pyscf) vs outcore (custom)?',numpy.allclose(f['ab'],f['aa']))

    tmpfile2.close()
