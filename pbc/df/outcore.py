#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.ao2mo.outcore import balance_segs
from pyscf.pbc.lib.kpt_misc import is_zero, gamma_point, unique, KPT_DIFF_TOL

libpbc = lib.load_library('libpbc')


def aux_e2(cell, auxcell, erifile, intor='cint3c2e_sph', aosym='s2ij', comp=1,
           kptij_lst=None, dataname='eri_mo', shls_slice=None, max_memory=2000,
           verbose=0):
    '''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    On diks, the integrals are stored as (kptij_idx, naux, nao_pair)

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    if h5py.is_hdf5(erifile):
        feri = h5py.File(erifile)
        if dataname in feri:
            del(feri[dataname])
        if dataname+'-kptij' in feri:
            del(feri[dataname+'-kptij'])
    else:
        feri = h5py.File(erifile, 'w')

    if kptij_lst is None:
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]

    nbas = cell.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, auxcell.nbas)

    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    aux_loc = auxcell.ao_loc_nr('ssc' in intor)[:shls_slice[5]+1]
    ao_loc = numpy.asarray(numpy.hstack([ao_loc[:-1], ao_loc[-1]+aux_loc]),
                           dtype=numpy.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naux = aux_loc[shls_slice[5]] - ao_loc[shls_slice[4]]

    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    for k, kptij in enumerate(kptij_lst):
        key = '%s/%d' % (dataname, k)
        if gamma_point(kptij):
            dtype = 'f8'
        else:
            dtype = 'c16'
        if aosym_ks2[k]:
            nao_pair = ni * (ni+1) // 2
        else:
            nao_pair = ni * nj
        if comp == 1:
            shape = (naux,nao_pair)
        else:
            shape = (comp,naux,nao_pair)
        chunks = (min(256,naux), min(256,nao_pair))  # 512 KB
        feri.create_dataset(key, shape, dtype, chunks=chunks)
    if naux == 0:
        feri.close()
        return erifile

    if aosym[:2] == 's2':
        nao_pair = ni * (ni+1) // 2
    else:
        nao_pair = ni * nj

    Ls = cell.get_lattice_Ls()
    nimgs = len(Ls)
    if gamma_point(kptij_lst):
        kk_type = 'g'
        dtype = numpy.double
        nkpts = nkptij = 1
        kptij_idx = numpy.array([0], dtype=numpy.int32)
        expkL = numpy.ones(1)
    elif j_only:
        kk_type = 'k'
        dtype = numpy.complex128
        kpts = kptij_idx = kpti
        expkL = numpy.exp(1j*numpy.dot(kpts, Ls.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = numpy.complex128
        kpts = unique(numpy.vstack([kpti,kptj]))[0]
        expkL = numpy.exp(1j*numpy.dot(kpts, Ls.T))
        wherei = numpy.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        wherej = numpy.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = numpy.asarray(wherei*nkpts+wherej, dtype=numpy.int32)
        nkptij = len(kptij_idx)
        nao_pair = ni * nj

    fill = 'PBCnr3c_fill_%s%s' % (kk_type, aosym[:2])
    cintopt = _vhf.make_cintopt(atm, bas, env, intor)

    buflen = max(8, int(max_memory*1e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty(ni*nj*buflen, dtype=dtype)

    naux0 = 0
    for istep, auxrange in enumerate(auxranges):
        sh0, sh1, nrow = auxrange
        sh0, sh1 = sh0+shls_slice[4], sh1+shls_slice[4]
        nrow = aux_loc[sh1] - aux_loc[sh0]
        sub_slice = (shls_slice[0], shls_slice[1],
                     nbas+shls_slice[2], nbas+shls_slice[3],
                     nbas*2+sh0, nbas*2+sh1)
        mat = numpy.ndarray((nkptij,comp,nao_pair,naux), dtype=dtype, buffer=buf)
        libpbc.PBCnr3c_drv(getattr(libpbc, intor), getattr(libpbc, fill),
                           mat.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(nkptij), ctypes.c_int(nkpts),
                           ctypes.c_int(comp), ctypes.c_int(nimgs),
                           Ls.ctypes.data_as(ctypes.c_void_p),
                           expkL.ctypes.data_as(ctypes.c_void_p),
                           kptij_idx.ctypes.data_as(ctypes.c_void_p),
                           (ctypes.c_int*6)(*sub_slice),
                           ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
                           atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
                           bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas),
                           env.ctypes.data_as(ctypes.c_void_p))

        for k, kptij in enumerate(kptij_lst):
            h5dat = feri['%s/%d'%(dataname,k)]
            for icomp, v in enumerate(mat[k]):
                v = lib.transpose(v, out=buf1)
                if gamma_point(kptij):
                    v = v.real
                if aosym_ks2[k] and v.shape[1] == ni**2:
                    v = lib.pack_tril(v.reshape(-1,ni,ni))
                if comp == 1:
                    h5dat[naux0:naux0+nrow] = v
                else:
                    h5dat[icomp,naux0:naux0+nrow] = v
        naux0 += nrow

    feri.close()
    return erifile


