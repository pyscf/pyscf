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
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

import numpy as np
import ctypes
from pyscf import lib
from pyscf.agf2 import mpi_helper

#TODO: move _get_blksize etc. here

libagf2 = lib.load_library('libagf2')


def cholesky_build(vv, vev, gf_occ, gf_vir, eps=1e-20):
    ''' Constructs the truncated auxiliaries from :attr:`vv` and :attr:`vev`.
        Performs a Cholesky decomposition via :func:`numpy.linalg.cholesky`,
        for a positive-definite or positive-semidefinite matrix. For the
        latter, the null space is removed.

        The :attr:`vv` matrix of :func:`build_se_part` can be positive-
        semidefinite when :attr:`gf_occ.naux` < :attr:`gf_occ.nphys` for
        the occupied self-energy, or :attr:`gf_vir.naux` < :attr:`gf_vir.nphys`
        for the virtual self-energy.
    '''

    ##NOTE: test this
    ##FIXME: doesn't seem to work when situation arises due to frozen core - might have to rethink this
    ##FIXME: won't work for UAGF2 if we use this solution (need to change naux)
    #if gf_occ.nphys >= (gf_occ.naux**2 * gf_vir.naux):
    #    # remove the null space from vv and vev
    #    zero_rows = np.all(np.absolute(vv) < eps, axis=1)
    #    zero_cols = np.all(np.absolute(vv) < eps, axis=0)
    #    null_space = np.logical_and(zero_rows, zero_cols)

    #    vv = vv[~null_space][:,~null_space]
    #    vev = vev[~null_space][:,~null_space]
    #    np.set_printoptions(precision=4, linewidth=150)

    #NOTE: this seems like an unscientific solution... rescue the above?
    try:
        b = np.linalg.cholesky(vv).T
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(vv)
        w[w < eps] = eps
        vv_posdef = np.dot(np.dot(v, np.diag(w)), v.T.conj())
        b = np.linalg.cholesky(vv_posdef).T

    b_inv = np.linalg.inv(b)

    m = np.dot(np.dot(b_inv.T, vev), b_inv)
    e, c = np.linalg.eigh(m)
    c = np.dot(b.T, c[:gf_occ.nphys])

    if c.shape[0] < gf_occ.nphys:
        c_full = np.zeros((gf_occ.nphys, c.shape[1]), dtype=c.dtype)
        c_full[~null_space] = c
        c = c_full

    return e, c


def build_mats_ragf2_incore(qeri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Wraps AGF2ee_vv_vev_islice
    '''

    fdrv = getattr(libagf2, 'AGF2ee_vv_vev_islice')

    qeri = np.asarray(qeri, order='C')
    e_i = np.asarray(gf_occ.energy, order='C') 
    e_a = np.asarray(gf_vir.energy, order='C')

    nmo = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    assert qeri.size == (nmo * nocc * nocc * nvir)

    vv = np.zeros((nmo*nmo))
    vev = np.zeros((nmo*nmo))

    rank, size = mpi_helper.rank, mpi_helper.size
    istart = rank * nocc // size
    iend = nocc if rank == (size-1) else (rank+1) * nocc // size

    fdrv(qeri.ctypes.data_as(ctypes.c_void_p),
         e_i.ctypes.data_as(ctypes.c_void_p),
         e_a.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(os_factor),
         ctypes.c_double(ss_factor),
         ctypes.c_int(nmo),
         ctypes.c_int(nocc),
         ctypes.c_int(nvir),
         ctypes.c_int(istart),
         ctypes.c_int(iend),
         vv.ctypes.data_as(ctypes.c_void_p),
         vev.ctypes.data_as(ctypes.c_void_p))

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    vv = vv.reshape(nmo, nmo)
    vev = vev.reshape(nmo, nmo)

    return vv, vev


def build_mats_ragf2_outcore(qeri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Python version of AGF2ee_vv_vev_islice to support outcore
    '''

    nmo = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    assert qeri.size == (nmo * nocc * nocc * nvir)

    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)
    eja = eja.ravel()

    for i in mpi_helper.nrange(gf_occ.naux):
        xija = qeri[:,i].reshape(nmo, -1)
        xjia = qeri[:,:,i].reshape(nmo, -1)

        eija = eja + gf_occ.energy[i]

        vv = lib.dot(xija, xija.T, alpha=fpos, beta=1, c=vv)
        vv = lib.dot(xija, xjia.T, alpha=fneg, beta=1, c=vv)

        exija = xija * eija[None]

        vev = lib.dot(exija, xija.T, alpha=fpos, beta=1, c=vev)
        vev = lib.dot(exija, xjia.T, alpha=fneg, beta=1, c=vev)

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    return vv, vev


def build_mats_dfragf2_incore(qxi, qja, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Wrapper for AGF2df_vv_vev_islice
    '''

    fdrv = getattr(libagf2, 'AGF2df_vv_vev_islice')

    qxi = np.asarray(qxi, order='C')
    qja = np.asarray(qja, order='C')
    e_i = np.asarray(gf_occ.energy, order='C') 
    e_a = np.asarray(gf_vir.energy, order='C')

    naux = qxi.shape[0]
    nmo = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    assert qxi.size == (naux * nmo * nocc)
    assert qja.size == (naux * nocc * nvir)

    vv = np.zeros((nmo*nmo))
    vev = np.zeros((nmo*nmo))

    rank, size = mpi_helper.rank, mpi_helper.size
    istart = rank * nocc // size
    iend = nocc if rank == (size-1) else (rank+1) * nocc // size

    fdrv(qxi.ctypes.data_as(ctypes.c_void_p),
         qja.ctypes.data_as(ctypes.c_void_p),
         e_i.ctypes.data_as(ctypes.c_void_p),
         e_a.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(os_factor),
         ctypes.c_double(ss_factor),
         ctypes.c_int(nmo),
         ctypes.c_int(nocc),
         ctypes.c_int(nvir),
         ctypes.c_int(naux),
         ctypes.c_int(istart),
         ctypes.c_int(iend),
         vv.ctypes.data_as(ctypes.c_void_p),
         vev.ctypes.data_as(ctypes.c_void_p))

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    vv = vv.reshape(nmo, nmo)
    vev = vev.reshape(nmo, nmo)

    return vv, vev


def build_mats_dfragf2_outcore(qxi, qja, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Python version of AGF2df_vv_vev_islice to support outcore
    '''

    naux = qxi.shape[0]
    nmo = gf_occ.nphys
    nocc = gf_occ.naux
    nvir = gf_vir.naux

    assert qxi.size == (naux * nmo * nocc)
    assert qja.size == (naux * nocc * nvir)

    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)
    eja = eja.ravel()

    buf = (np.zeros((nmo, nocc*nvir)), np.zeros((nmo*nocc, nvir)))

    for i in mpi_helper.nrange(gf_occ.naux):
        qx = qxi.reshape(naux, nmo, nocc)[:,:,i]
        xija = lib.dot(qx.T, qja, c=buf[0])
        xjia = lib.dot(qxi.T, qja[:,i*nvir:(i+1)*nvir], c=buf[1])
        xjia = xjia.reshape(nmo, nocc*nvir)

        eija = eja + gf_occ.energy[i]

        vv = lib.dot(xija, xija.T, alpha=fpos, beta=1, c=vv)
        vv = lib.dot(xija, xjia.T, alpha=fneg, beta=1, c=vv)

        exija = xija * eija[None]

        vev = lib.dot(exija, xija.T, alpha=fpos, beta=1, c=vev)
        vev = lib.dot(exija, xjia.T, alpha=fneg, beta=1, c=vev)

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    return vv, vev


def build_mats_uagf2_incore(qeri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Wraps AGF2uee_vv_vev_islice
    '''

    fdrv = getattr(libagf2, 'AGF2uee_vv_vev_islice')

    qeri_a = np.asarray(qeri[0], order='C')
    qeri_b = np.asarray(qeri[1], order='C')
    e_i = np.asarray(gf_occ[0].energy, order='C')
    e_I = np.asarray(gf_occ[1].energy, order='C')
    e_a = np.asarray(gf_vir[0].energy, order='C')
    e_A = np.asarray(gf_vir[1].energy, order='C')

    nmo = gf_occ[0].nphys
    noa, nob = gf_occ[0].naux, gf_occ[1].naux
    nva, nvb = gf_vir[0].naux, gf_vir[1].naux

    assert qeri_a.size == (nmo * noa * noa * nva)
    assert qeri_b.size == (nmo * noa * nob * nvb)

    vv = np.zeros((nmo*nmo))
    vev = np.zeros((nmo*nmo))

    rank, size = mpi_helper.rank, mpi_helper.size
    istart = rank * noa // size
    iend = noa if rank == (size-1) else (rank+1) * noa // size

    fdrv(qeri_a.ctypes.data_as(ctypes.c_void_p),
         qeri_b.ctypes.data_as(ctypes.c_void_p),
         e_i.ctypes.data_as(ctypes.c_void_p),
         e_I.ctypes.data_as(ctypes.c_void_p),
         e_a.ctypes.data_as(ctypes.c_void_p),
         e_A.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(os_factor),
         ctypes.c_double(ss_factor),
         ctypes.c_int(nmo),
         ctypes.c_int(noa),
         ctypes.c_int(nob),
         ctypes.c_int(nva),
         ctypes.c_int(nvb),
         ctypes.c_int(istart),
         ctypes.c_int(iend),
         vv.ctypes.data_as(ctypes.c_void_p),
         vev.ctypes.data_as(ctypes.c_void_p))

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    vv = vv.reshape(nmo, nmo)
    vev = vev.reshape(nmo, nmo)

    return vv, vev


def build_mats_uagf2_outcore(qeri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Python version of AGF2uee_vv_vev_islice to support outcore
    '''

    nmo = gf_occ[0].nphys
    noa, nob = gf_occ[0].naux, gf_occ[1].naux
    nva, nvb = gf_vir[0].naux, gf_vir[1].naux

    assert qeri[0].size == (nmo * noa * noa * nva)
    assert qeri[1].size == (nmo * noa * nob * nvb)

    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    fposa = ss_factor
    fnega = -ss_factor
    fposb = os_factor

    eja_a = lib.direct_sum('j,a->ja', gf_occ[0].energy, -gf_vir[0].energy).ravel()
    eja_b = lib.direct_sum('j,a->ja', gf_occ[1].energy, -gf_vir[1].energy).ravel()

    for i in mpi_helper.nrange(noa):
        xija_aa = qeri[0][:,i].reshape(nmo, -1)
        xija_ab = qeri[1][:,i].reshape(nmo, -1)
        xjia_aa = qeri[0][:,:,i].reshape(nmo, -1)

        eija_aa = eja_a + gf_occ[0].energy[i]
        eija_ab = eja_b + gf_occ[0].energy[i]

        vv = lib.dot(xija_aa, xija_aa.T, alpha=fposa, beta=1, c=vv)
        vv = lib.dot(xija_aa, xjia_aa.T, alpha=fnega, beta=1, c=vv)
        vv = lib.dot(xija_ab, xija_ab.T, alpha=fposb, beta=1, c=vv)

        exija_aa = xija_aa * eija_aa[None]
        exija_ab = xija_ab * eija_ab[None]

        vev = lib.dot(exija_aa, xija_aa.T, alpha=fposa, beta=1, c=vev)
        vev = lib.dot(exija_aa, xjia_aa.T, alpha=fnega, beta=1, c=vev)
        vev = lib.dot(exija_ab, xija_ab.T, alpha=fposb, beta=1, c=vev)

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    return vv, vev

    
def build_mats_dfuagf2_incore(qxi, qja, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Wrapper for AGF2udf_vv_vev_islice
    '''

    fdrv = getattr(libagf2, 'AGF2udf_vv_vev_islice')

    qxi_a, qxi_b = qxi
    qja_a, qja_b = qja

    qxi = np.asarray(qxi_a, order='C')
    qja = np.asarray(qja_a, order='C')
    qJA = np.asarray(qja_b, order='C')
    e_i = np.asarray(gf_occ[0].energy, order='C')
    e_I = np.asarray(gf_occ[1].energy, order='C')
    e_a = np.asarray(gf_vir[0].energy, order='C')
    e_A = np.asarray(gf_vir[1].energy, order='C')

    nmo = gf_occ[0].nphys
    noa, nob = gf_occ[0].naux, gf_occ[1].naux
    nva, nvb = gf_vir[0].naux, gf_vir[1].naux
    naux = qxi.shape[0]

    assert qxi.size == (naux * nmo * noa)
    assert qja.size == (naux * noa * nva)
    assert qJA.size == (naux * nob * nvb)

    vv = np.zeros((nmo*nmo))
    vev = np.zeros((nmo*nmo))

    rank, size = mpi_helper.rank, mpi_helper.size
    istart = rank * noa // size
    iend = noa if rank == (size-1) else (rank+1) * noa // size

    fdrv(qxi.ctypes.data_as(ctypes.c_void_p),
         qja.ctypes.data_as(ctypes.c_void_p),
         qJA.ctypes.data_as(ctypes.c_void_p),
         e_i.ctypes.data_as(ctypes.c_void_p),
         e_I.ctypes.data_as(ctypes.c_void_p),
         e_a.ctypes.data_as(ctypes.c_void_p),
         e_A.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(os_factor),
         ctypes.c_double(ss_factor),
         ctypes.c_int(nmo),
         ctypes.c_int(noa),
         ctypes.c_int(nob),
         ctypes.c_int(nva),
         ctypes.c_int(nvb),
         ctypes.c_int(naux),
         ctypes.c_int(istart),
         ctypes.c_int(iend),
         vv.ctypes.data_as(ctypes.c_void_p),
         vev.ctypes.data_as(ctypes.c_void_p))

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    vv = vv.reshape(nmo, nmo)
    vev = vev.reshape(nmo, nmo)

    return vv, vev


def build_mats_dfuagf2_outcore(qxi, qja, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Python version of AGF2udf_vv_vev_islice to support outcore
    '''

    nmo = gf_occ[0].nphys
    noa, nob = gf_occ[0].naux, gf_occ[1].naux
    nva, nvb = gf_vir[0].naux, gf_vir[1].naux
    naux = qxi[0].shape[0]

    qxi_a, qxi_b = qxi
    qja_a, qja_b = qja

    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    fposa = ss_factor
    fnega = -ss_factor
    fposb = os_factor

    eja_a = lib.direct_sum('j,a->ja', gf_occ[0].energy, -gf_vir[0].energy).ravel()
    eja_b = lib.direct_sum('j,a->ja', gf_occ[1].energy, -gf_vir[1].energy).ravel()

    buf = (np.zeros((nmo, noa*nva)), 
           np.zeros((nmo, nob*nvb)),
           np.zeros((nmo*noa, nva))) 

    for i in mpi_helper.nrange(noa):
        qx_a = qxi_a.reshape(naux, nmo, noa)[:,:,i]
        xija_aa = lib.dot(qx_a.T, qja_a, c=buf[0])
        xija_ab = lib.dot(qx_a.T, qja_b, c=buf[1])
        xjia_aa = lib.dot(qxi_a.T, qja_a[:,i*nva:(i+1)*nva], c=buf[2])
        xjia_aa = xjia_aa.reshape(nmo, -1)

        eija_aa = eja_a + gf_occ[0].energy[i]
        eija_ab = eja_b + gf_occ[0].energy[i]

        vv = lib.dot(xija_aa, xija_aa.T, alpha=fposa, beta=1, c=vv)
        vv = lib.dot(xija_aa, xjia_aa.T, alpha=fnega, beta=1, c=vv)
        vv = lib.dot(xija_ab, xija_ab.T, alpha=fposb, beta=1, c=vv)

        exija_aa = xija_aa * eija_aa[None]
        exija_ab = xija_ab * eija_ab[None]

        vev = lib.dot(exija_aa, xija_aa.T, alpha=fposa, beta=1, c=vev)
        vev = lib.dot(exija_aa, xjia_aa.T, alpha=fnega, beta=1, c=vev)
        vev = lib.dot(exija_ab, xija_ab.T, alpha=fposb, beta=1, c=vev)

    vv = mpi_helper.reduce(vv)
    vev = mpi_helper.reduce(vev)

    return vv, vev


def get_blksize(max_memory_total, *sizes):
    ''' Gets a block size such that the sum of the product of
        :attr:`sizes` with :attr:`blksize` is less than avail
        memory.

        If multiple tuples are provided, the maximum is used.
    '''

    if isinstance(sizes[0], tuple):
        sum_of_sizes = max([sum(x) for x in sizes])
    else:
        sum_of_sizes = sum(sizes)

    mem_avail = max_memory_total - lib.current_memory()[0]
    mem_avail *= 8e6 # MB -> bits
    sum_of_sizes *= 64 # 64 bits -> bits

    return int(mem_avail / sum_of_sizes)
