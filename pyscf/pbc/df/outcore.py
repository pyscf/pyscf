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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.ao2mo.outcore import balance_segs
from pyscf.pbc.lib.kpts_helper import gamma_point, unique, KPT_DIFF_TOL
from pyscf.pbc.df.incore import wrap_int3c

libpbc = lib.load_library('libpbc')


def aux_e1(cell, auxcell, erifile, intor='int3c2e', aosym='s2ij', comp=None,
           kptij_lst=None, dataname='eri_mo', shls_slice=None, max_memory=2000,
           verbose=0):
    r'''3-center AO integrals (L|ij) with double lattice sum:
    \sum_{lm} (L[0]|i[l]j[m]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, naux, nao_pair) or four-index
    integral tensor (kptij_idx, comp, naux, nao_pair) are stored on disk.

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = h5py.File(erifile, 'a')
    else:
        feri = h5py.File(erifile, 'w')
    if dataname in feri:
        del(feri[dataname])
    if dataname+'-kptij' in feri:
        del(feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    naux = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'
    for k, kptij in enumerate(kptij_lst):
        key = '%s/%d' % (dataname, k)
        if gamma_point(kptij):
            dtype = 'f8'
        else:
            dtype = 'c16'
        if aosym_ks2[k]:
            nao_pair = nii
        else:
            nao_pair = nij
        if comp == 1:
            shape = (naux,nao_pair)
        else:
            shape = (comp,naux,nao_pair)
        feri.create_dataset(key, shape, dtype)
    if naux == 0:
        feri.close()
        return erifile

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*1e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty(ni*nj*buflen, dtype=dtype)

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst)

    naux0 = 0
    for istep, auxrange in enumerate(auxranges):
        sh0, sh1, nrow = auxrange
        sub_slice = (shls_slice[0], shls_slice[1],
                     shls_slice[2], shls_slice[3],
                     shls_slice[4]+sh0, shls_slice[4]+sh1)
        mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
        mat = int3c(sub_slice, mat)

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

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


def _aux_e2(cell, auxcell, erifile, intor='int3c2e', aosym='s2ij', comp=None,
            kptij_lst=None, dataname='eri_mo', shls_slice=None, max_memory=2000,
            verbose=0):
    r'''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    Three-index integral tensor (kptij_idx, nao_pair, naux) or four-index
    integral tensor (kptij_idx, comp, nao_pair, naux) are stored on disk.

    **This function should be only used by df and mdf initialization function
    _make_j3c**

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    intor, comp = gto.moleintor._get_intor_and_comp(cell._add_suffix(intor), comp)

    if isinstance(erifile, h5py.Group):
        feri = erifile
    elif h5py.is_hdf5(erifile):
        feri = h5py.File(erifile, 'a')
    else:
        feri = h5py.File(erifile, 'w')
    if dataname in feri:
        del(feri[dataname])
    if dataname+'-kptij' in feri:
        del(feri[dataname+'-kptij'])

    if kptij_lst is None:
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    nkptij = len(kptij_lst)

    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)
    nij = ni * nj

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_ks2 = abs(kpti-kptj).sum(axis=1) < KPT_DIFF_TOL
    j_only = numpy.all(aosym_ks2)
    #aosym_ks2 &= (aosym[:2] == 's2' and shls_slice[:2] == shls_slice[2:4])
    aosym_ks2 &= aosym[:2] == 's2'

    if j_only and aosym[:2] == 's2':
        assert(shls_slice[2] == 0)
        nao_pair = nii
    else:
        nao_pair = nij

    if gamma_point(kptij_lst):
        dtype = numpy.double
    else:
        dtype = numpy.complex128

    buflen = max(8, int(max_memory*.47e6/16/(nkptij*ni*nj*comp)))
    auxdims = aux_loc[shls_slice[4]+1:shls_slice[5]+1] - aux_loc[shls_slice[4]:shls_slice[5]]
    auxranges = balance_segs(auxdims, buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty(nkptij*comp*ni*nj*buflen, dtype=dtype)
    buf1 = numpy.empty_like(buf)

    int3c = wrap_int3c(cell, auxcell, intor, aosym, comp, kptij_lst)

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
# sorted_ij_idx: Sort and group the kptij_lst according to the ordering in
# df._make_j3c to reduce the data fragment in the hdf5 file.  When datasets
# are written to hdf5, they are saved sequentially. If the integral data are
# saved as the order of kptij_lst, removing the datasets in df._make_j3c will
# lead to holes that can not be reused.
    sorted_ij_idx = numpy.hstack([numpy.where(uniq_inverse == k)[0]
                                  for k, kpt in enumerate(uniq_kpts)])
    tril_idx = numpy.tril_indices(ni)
    tril_idx = tril_idx[0] * ni + tril_idx[1]
    def save(istep, mat):
        for k in sorted_ij_idx:
            v = mat[k]
            if gamma_point(kptij_lst[k]):
                v = v.real
            if aosym_ks2[k] and nao_pair == ni**2:
                v = v[:,tril_idx]
            feri['%s/%d/%d' % (dataname,k,istep)] = v

    with lib.call_in_background(save) as bsave:
        for istep, auxrange in enumerate(auxranges):
            sh0, sh1, nrow = auxrange
            sub_slice = (shls_slice[0], shls_slice[1],
                         shls_slice[2], shls_slice[3],
                         shls_slice[4]+sh0, shls_slice[4]+sh1)
            mat = numpy.ndarray((nkptij,comp,nao_pair,nrow), dtype=dtype, buffer=buf)
            bsave(istep, int3c(sub_slice, mat))
            buf, buf1 = buf1, buf

    if not isinstance(erifile, h5py.Group):
        feri.close()
    return erifile


